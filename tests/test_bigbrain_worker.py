# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Tests for tools/bigbrain/worker.py -- no exllamav3, no GPU.

Two layers:

* Slice selection + rebase (plain stand-ins; no torch). The filter+rebase is
  the part that, if wrong, silently loads the wrong weights or crashes the
  cache index.
* The per-request recurrent-state contract and the pipeline itself, against a
  FAKE model that reproduces exllamav3 1.6.1's contract on CPU torch:
    - Cache.get_new_state() hands out a slot handle (GDNState) whose tensors
      live in the cache's per-layer pools (GDNLayerState: conv ring +
      recurrent row), cleared on creation (cache/cache.py:326,
      modules/gated_delta_net.py:116-163, :343-346);
    - prepare_for_recurrence (cache/recurrent_util.py:18-81): past_len > 0
      with no params["recurrent_states"] raises ValueError, and the states'
      .position must equal past_len;
    - advance_recurrent_states (recurrent_util.py:84-105): position += seqlen;
    - a GDN layer called WITHOUT recurrent_states runs stateless (zero initial
      state, nothing saved: gated_delta_net.py:1193-1214) -- so a decode step
      that loses the state gives silently WRONG output, which these tests
      catch by comparing against a monolithic reference;
    - TransformerBlock under params["prefill"] skips the MLP and returns x
      (modules/transformer.py:166-168), as prefill_ls drives it.
  Torch-dependent tests skip when torch is absent.
"""
from __future__ import annotations

import inspect
import shlex
import sys
import threading
import time
import types
from collections import deque
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from tools.bigbrain.pipe import BigBrainLink
from tools.bigbrain.worker import (
    BigBrainRemoteError,
    BigBrainStateError,
    SamplingSettings,
    SliceWorker,
    _select_and_rebase,
    _select_slice,
    build_arg_parser,
    make_cache,
    read_dump,
    write_dump,
)


# --------------------------------------------------------------------------
# slice selection + rebase (no torch)
# --------------------------------------------------------------------------

def _mock_block(idx):
    """A TransformerBlock stand-in: layer_idx + an inner Attention with its own
    layer_idx (both must be rebased together)."""
    return SimpleNamespace(layer_idx=idx, key=f"model.layers.{idx}",
                           attn=SimpleNamespace(layer_idx=idx))


def _mock_model(n_blocks=8):
    """[Embedding, Block x N, RMSNorm, lm_head] -- the flat list shape
    (architecture/llama.py). Prolog/epilog modules have layer_idx None."""
    return SimpleNamespace(modules=[
        SimpleNamespace(layer_idx=None, key="model.embed_tokens"),
        *[_mock_block(i) for i in range(n_blocks)],
        SimpleNamespace(layer_idx=None, key="model.norm"),
        SimpleNamespace(layer_idx=None, key="lm_head"),
    ])


def test_first_node_keeps_prolog_and_rebases():
    m = _mock_model(8)
    n = _select_and_rebase(m, 0, 4, rank=0, total=2)
    assert n == 4
    # Embedding + 4 rebased blocks (no epilog: not the last node).
    assert len(m.modules) == 1 + 4
    assert m.modules[0].key == "model.embed_tokens"
    blocks = m.modules[1:]
    assert [b.layer_idx for b in blocks] == [0, 1, 2, 3]
    # The inner Attention was rebased in lockstep.
    assert [b.attn.layer_idx for b in blocks] == [0, 1, 2, 3]
    # Weight keys are untouched (rebase must not change what loads).
    assert [b.key for b in blocks] == [f"model.layers.{i}" for i in range(4)]


def test_last_node_keeps_epilog_and_rebases():
    m = _mock_model(8)
    n = _select_and_rebase(m, 4, 8, rank=1, total=2)
    assert n == 4
    # 4 rebased blocks + RMSNorm + lm_head (no Embedding: not the first node).
    assert len(m.modules) == 4 + 2
    blocks = m.modules[:4]
    assert [b.layer_idx for b in blocks] == [0, 1, 2, 3]
    assert [b.attn.layer_idx for b in blocks] == [0, 1, 2, 3]
    # The rebase is LOCAL to the slice: the kept blocks read 0..3, not 4..7.
    assert blocks[0].layer_idx == 0
    assert m.modules[4].key == "model.norm"
    assert m.modules[5].key == "lm_head"


def test_middle_node_keeps_neither_prolog_nor_epilog():
    m = _mock_model(12)
    n = _select_and_rebase(m, 4, 8, rank=1, total=3)
    assert n == 4
    # Exactly the 4 rebased blocks -- no Embedding, no norm/lm_head.
    assert len(m.modules) == 4
    assert [b.layer_idx for b in m.modules] == [0, 1, 2, 3]
    assert all(b.key.startswith("model.layers.") for b in m.modules)


def test_single_node_keeps_everything():
    m = _mock_model(8)
    n = _select_and_rebase(m, 0, 8, rank=0, total=1)
    assert n == 8
    # Embedding + 8 blocks + norm + lm_head, blocks already 0..7 (rebase is a
    # no-op for the full model but must not disturb it).
    assert len(m.modules) == 1 + 8 + 2
    assert [b.layer_idx for b in m.modules[1:9]] == list(range(8))
    assert m.modules[0].key == "model.embed_tokens"
    assert m.modules[-1].key == "lm_head"


def test_rebase_is_idempotent_safe_on_disjoint_slices():
    """Two disjoint slices of the same model rebase independently to 0..N-1 --
    the invariant that makes per-node dense caches line up."""
    a = _mock_model(10)
    b = _mock_model(10)
    _select_and_rebase(a, 0, 5, rank=0, total=2)
    _select_and_rebase(b, 5, 10, rank=1, total=2)
    assert [m.layer_idx for m in a.modules if m.layer_idx is not None] == [0, 1, 2, 3, 4]
    assert [m.layer_idx for m in b.modules if m.layer_idx is not None] == [0, 1, 2, 3, 4]


class _IterModule:
    """exllamav3 Module shape: iterating yields itself then every descendant
    (modules/module.py:53-56)."""

    def __init__(self, key, layer_idx=None, children=(), caps=None):
        self.key, self.layer_idx = key, layer_idx
        self.modules = list(children)
        self.caps = dict(caps or {})

    def __iter__(self):
        yield self
        for m in self.modules:
            yield from m


def _hybrid_model(n=8, deepstack_after=(0, 1)):
    """Qwen3.5-style: block i holds a GatedDeltaNet (or an Attention every 4th
    layer) as a nested child carrying its own layer_idx; non-block modules
    (DeepstackEmbed-style, layer_idx None) may sit between blocks."""
    mods = [_IterModule("embed")]
    for i in range(n):
        if (i + 1) % 4 == 0:
            inner = _IterModule(f"l{i}.self_attn", i, caps={"kv_cache": True})
        else:
            inner = _IterModule(f"l{i}.linear_attn", i, caps={"recurrent_cache": True})
        mods.append(_IterModule(f"l{i}", i, children=[_IterModule(f"l{i}.norm"), inner]))
        if i in deepstack_after:
            mods.append(_IterModule(f"l{i}.deepstack"))
    mods += [_IterModule("norm"), _IterModule("lm_head", caps={"logits_output": True})]
    model = SimpleNamespace(modules=mods, last_kv_module_idx=len(mods) - 3,
                            logit_layer_idx=len(mods) - 1, first_block_idx=1,
                            modules_dict={"stale": True})
    model.__dict__["_get_cache_layers"] = ["stale"]  # a computed cached_property
    return model


def test_rebase_reaches_nested_recurrent_and_attention_children():
    m = _hybrid_model(8)
    _select_and_rebase(m, 3, 7, rank=1, total=3)
    blocks = [x for x in m.modules if x.layer_idx is not None]
    assert [b.key for b in blocks] == ["l3", "l4", "l5", "l6"]
    assert [b.layer_idx for b in blocks] == [0, 1, 2, 3]
    inner = [c for b in blocks for c in b.modules if c.layer_idx is not None]
    # GDN (recurrent cache key) and Attention (KV cache key) both rebased.
    assert [(c.key, c.layer_idx) for c in inner] == [
        ("l3.self_attn", 0), ("l4.linear_attn", 1), ("l5.linear_attn", 2), ("l6.linear_attn", 3)]


def test_between_block_modules_travel_with_their_block():
    m = _hybrid_model(8, deepstack_after=(0, 1))
    _select_and_rebase(m, 0, 1, rank=0, total=2)
    assert [x.key for x in m.modules] == ["embed", "l0", "l0.deepstack"]
    m = _hybrid_model(8, deepstack_after=(0, 1))
    _select_and_rebase(m, 1, 8, rank=1, total=2)
    assert [x.key for x in m.modules][:3] == ["l1", "l1.deepstack", "l2"]
    assert [x.key for x in m.modules][-2:] == ["norm", "lm_head"]


def test_rebase_recomputes_module_indices_and_drops_cached_layer_lists():
    m = _hybrid_model(8, deepstack_after=())
    _select_and_rebase(m, 4, 8, rank=1, total=2)
    # [l4..l7, norm, lm_head]: the last KV-affecting module is l7 (index 3),
    # the logits layer is index 5; there is no prolog.
    assert m.last_kv_module_idx == 3
    assert m.logit_layer_idx == 5
    assert m.first_block_idx == 0
    assert "_get_cache_layers" not in m.__dict__
    assert m.modules_dict is None
    m = _hybrid_model(8, deepstack_after=())
    _select_and_rebase(m, 0, 4, rank=0, total=2)
    assert m.last_kv_module_idx == 4  # [embed, l0..l3]
    assert m.logit_layer_idx is None
    assert m.first_block_idx == 1


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 3), (5, 2), (0, 9)])
def test_bad_slice_is_refused(start, end):
    with pytest.raises(ValueError):
        _select_and_rebase(_mock_model(8), start, end, rank=0, total=2)


def test_make_cache_passes_max_batch_size_only_when_supported():
    seen = {}

    class NewCache:
        def __init__(self, model, max_num_tokens, layer_type=None, max_batch_size=16,
                     max_history=0, **kwargs):
            seen["new"] = (max_num_tokens, max_batch_size)

    class OldCache:
        def __init__(self, model, max_num_tokens, layer_type=None):
            seen["old"] = max_num_tokens

    make_cache(NewCache, object(), 4096, max_batch_size=1)
    make_cache(OldCache, object(), 2048, max_batch_size=1)
    assert seen == {"new": (4096, 1), "old": 2048}


# --------------------------------------------------------------------------
# sampling settings (no torch)
# --------------------------------------------------------------------------

def test_sampling_settings_roundtrip_and_validation():
    s = SamplingSettings.from_dict({"temperature": 0.7, "top_p": 0.9, "top_k": 40,
                                    "seed": 123, "min_p": 0.05, "unknown_key": 1})
    assert s.to_dict() == {"temperature": 0.7, "top_p": 0.9, "top_k": 40,
                           "seed": 123, "min_p": 0.05}
    assert not s.greedy
    assert SamplingSettings().greedy  # default = greedy (parity runs)
    assert SamplingSettings(temperature=0.8, top_k=1).greedy
    for bad in ({"temperature": -1}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": -2},
                {"seed": "x"}, {"min_p": 2}, {"temperature": "hot"}):
        with pytest.raises(ValueError):
            SamplingSettings.from_dict(bad)


# --------------------------------------------------------------------------
# fake exllamav3 (CPU torch) -- see the module docstring for the contract
# --------------------------------------------------------------------------

H = 6        # hidden size
V = 12       # real vocab
VPAD = 16    # lm_head width: padded columns carry a large bias, so an argmax
             # that forgets to crop to config.vocab_size picks a pad id


def _torch():
    return pytest.importorskip("torch")


def _weights(seed, *shape):
    t = _torch()
    g = t.Generator().manual_seed(seed)
    return t.randn(*shape, generator=g, dtype=t.float32) * 0.6


class FakeState:
    """GDNState stand-in, same constructor (cache, slot, position, clear):
    a slot handle + position; its tensors live in the cache's per-layer pools,
    cleared when the state is created (gated_delta_net.py:124-163)."""
    made = 0

    def __init__(self, cache, slot, position, clear=True):
        self.cache, self.slot, self.position = cache, slot, position
        FakeState.made += 1
        self.serial = FakeState.made  # id() of a freed state can be reused
        self.last_history = 0
        if clear:
            for layer in cache.recurrent_layers.values():
                layer.clear(slot)


class FakeLayerState:
    """GDNLayerState stand-in: a per-slot conv ring (here: the previous input
    row, a 2-tap conv) and a per-slot recurrent row."""

    def __init__(self, n_slots):
        t = _torch()
        self.rec = t.zeros(n_slots, H)
        self.conv = t.zeros(n_slots, H)

    def clear(self, slot):
        self.rec[slot].zero_()
        self.conv[slot].zero_()


class FakePLEState:
    """PLELayerState stand-in: the carried token-id context per slot."""

    def __init__(self, n_slots):
        self.prev = _torch().zeros(n_slots, dtype=_torch().long)

    def clear(self, slot):
        self.prev[slot] = 0


class FakeCacheLayer:
    def __init__(self, max_tokens):
        self.v = _torch().zeros(max_tokens, H)


class FakeCache:
    """Pools are INFERENCE tensors, as exllamav3's loader allocates them inside
    model.load() under torch.inference_mode: an in-place op on them outside
    inference_mode raises exactly what a live node raised (exllamav3's
    gated_delta_net.py:345 conv_state[idx].zero_() via get_new_state())."""

    def __init__(self, model, max_num_tokens, layer_type=None, max_batch_size=16,
                 max_history=0, **kwargs):
        self.model, self.max_num_tokens = model, max_num_tokens
        with _torch().inference_mode():
            self.layers = {(m.layer_idx, 0): FakeCacheLayer(max_num_tokens)
                           for m in model.get_cache_layers()}
            self.recurrent_layers = {(m.layer_idx, 0): m.layer_state_cls(max_batch_size)
                                     for m in model.get_recurrent_layers()}
        self.recurrent_state_cls = FakeState
        self.num_slots = max_batch_size
        self.free_list = deque(range(max_batch_size))
        self.created = 0

    def get_new_state(self):
        assert len(self.free_list) > 0, "Cannot create new state: no available slots"
        self.created += 1
        return FakeState(self, self.free_list.popleft(), 0, clear=True)

    def release_state(self, state):
        self.free_list.appendleft(state.slot)

    def reset_states(self):
        self.free_list = deque(range(self.num_slots))

    def get_recurrent_layer(self, instance):
        return self.recurrent_layers[instance]


class FakeModule:
    caps: dict = {}
    layer_idx = None

    def __init__(self):
        self.modules = []

    def __iter__(self):
        yield self
        for m in self.modules:
            yield from m

    def prepare_for_device(self, x, params):
        return x


class FakeEmbedding(FakeModule):
    def __init__(self):
        super().__init__()
        self.E = _weights(1, V, H)

    def forward(self, ids, params):
        if "input_ids" not in params:  # embedding.py:185-186
            params["input_ids"] = ids
        return self.E[ids.long()].float()


class FakeExpand(FakeModule):
    """ExpandStreams (hyperconnections.py:43-56): (b,s,h) -> (b,s,hc,h) fp32."""

    def __init__(self, hc):
        super().__init__()
        self.hc = hc

    def forward(self, x, params):
        return x.float().unsqueeze(2).expand(-1, -1, self.hc, -1).contiguous()


class FakeCollapse(FakeModule):
    """The final stream mixer (GatedResidual, use_combine=False): (b,s,hc,h) -> (b,s,h)."""

    def __init__(self, hc):
        super().__init__()
        self.w = _torch().linspace(0.5, 1.5, hc)

    def forward(self, x, params):
        return (x * self.w.view(1, 1, -1, 1)).mean(dim=2)


class FakePLE(FakeModule):
    """PLELayer stand-in (modules/ple.py): NEGATIVE layer_idx -(i+1), sits before
    block i, reads params["input_ids"] (falls back to zeros like ple.py:503-506)
    and carries the previous id in its recurrent slot."""
    caps = {"recurrent_cache": True, "prefetch_ids": True}

    def __init__(self, block_idx):
        super().__init__()
        self.layer_idx = -(block_idx + 1)
        self.layer_state_cls = FakePLEState
        self.E2 = _weights(500 + block_idx, V, H)
        self.prefetched = []

    def prefetch(self, ids, params):
        self.prefetched.append([int(t) for t in ids.reshape(-1).tolist()])

    def forward(self, x, params):
        t = _torch()
        ids = params.get("input_ids")
        if ids is None:
            ids = t.zeros(x.shape[:2], dtype=t.long)
        rs = params["recurrent_states"]
        layer = rs[0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance", 0)))
        slot = rs[0].slot
        prev = int(layer.prev[slot])
        rows = []
        for i in ids.reshape(-1).tolist():
            rows.append(self.E2[int(i)] + 0.5 * self.E2[prev])
            prev = int(i)
        layer.prev[slot] = prev
        delta = t.stack(rows).unsqueeze(0)
        return x + (delta.unsqueeze(2) if x.dim() == 4 else delta)


class FakeGDN(FakeModule):
    """Linear-attention layer: a decaying recurrent state + a 2-tap conv.
    WITHOUT params["recurrent_states"] it runs stateless (zero state, nothing
    saved) -- exactly like GatedDeltaNet.forward (gated_delta_net.py:1193)."""
    caps = {"recurrent_cache": True}
    seen: list

    def __init__(self, idx):
        super().__init__()
        self.layer_idx = idx
        self.layer_state_cls = FakeLayerState
        self.W = _weights(100 + idx, H, H)
        self.seen = []
        self.fail_next = False

    def forward(self, x, params):
        t = _torch()
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected GDN failure")
        rs = params.get("recurrent_states")
        S = x.shape[1]
        if rs:
            assert "recurrent_slots" in params, "prepare_for_recurrence did not run"
            layer = rs[0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance", 0)))
            slot = rs[0].slot
            state, prev = layer.rec[slot].clone(), layer.conv[slot].clone()
            self.seen.append((rs[0].serial, rs[0].position, params["past_len"], S))
        else:
            state, prev = t.zeros(H), t.zeros(H)
            self.seen.append((None, None, params["past_len"], S))
        out = []
        for i in range(S):
            u = x[0, i] + 0.5 * prev
            state = 0.9 * state + u
            prev = x[0, i]
            out.append(t.tanh(state @ self.W))
        if rs:
            layer.rec[slot] = state
            layer.conv[slot] = prev
        return t.stack(out).unsqueeze(0)


class FakeAttention(FakeModule):
    """Full attention: K/V per position in cache.layers[layer_idx, inst]."""
    caps = {"kv_cache": True}

    def __init__(self, idx):
        super().__init__()
        self.layer_idx = idx
        self.W = _weights(200 + idx, H, H)

    def forward(self, x, params):
        t = _torch()
        layer = params["cache"].layers[(self.layer_idx, params.get("layer_instance", 0))]
        past, S = params["past_len"], x.shape[1]
        layer.v[past:past + S] = x[0]
        out = [layer.v[: past + i + 1].mean(dim=0) @ self.W for i in range(S)]
        return t.stack(out).unsqueeze(0)


class FakeBlock(FakeModule):
    def __init__(self, idx, kind):
        super().__init__()
        self.layer_idx = idx
        self.attn = FakeGDN(idx) if kind == "gdn" else FakeAttention(idx)
        self.M = _weights(300 + idx, H, H)
        self.c = _torch().linspace(0.8, 1.2, 4)
        self.modules = [self.attn]

    def forward(self, x, params):
        t = _torch()
        streams = x.dim() == 4  # hyper-connection residual (b, s, hc, h)
        y = self.attn.forward(x.mean(dim=2) if streams else x, params)
        if params.get("prefill"):
            return x  # transformer.py:167: prefill stops after attention
        if streams:
            x = x + y.unsqueeze(2) * self.c[: x.shape[2]].view(1, 1, -1, 1)
            return x + 0.1 * t.tanh(x.mean(dim=2) @ self.M).unsqueeze(2)
        x = x + y
        return x + 0.1 * t.tanh(x @ self.M)


class FakeNorm(FakeModule):
    def forward(self, x, params):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


class FakeHead(FakeModule):
    caps = {"logits_output": True}

    def __init__(self):
        super().__init__()
        t = _torch()
        self.W = _weights(7, H, VPAD)
        self.bias = t.zeros(VPAD)
        self.bias[V:] = 100.0  # padding columns: must be cropped away

    def forward(self, x, params):
        return x @ self.W + self.bias


def fake_prepare_for_recurrence(input_ids, params, model):
    """recurrent_util.py:18-81, batch_shape path."""
    bsz, _ = params["batch_shape"]
    past_len = params.get("past_len", 0)
    rs = params.get("recurrent_states")
    if past_len > 0:
        if rs is None:
            raise ValueError("Past length given, but no previous state for recurrence in params")
        assert all(r.position == past_len for r in rs), "recurrent states don't match input past_len"
    else:
        if rs is None:
            rs = [params["cache"].get_new_state() for _ in range(bsz)]
            params["recurrent_states"] = rs
        else:
            assert all(r.position == 0 for r in rs), "recurrent states don't match input past_len"
    params["recurrent_slots"] = tuple(r.slot for r in rs)


ADVANCES = []


def fake_advance_recurrent_states(input_ids, params, model):
    """recurrent_util.py:84-105: position += seqlen."""
    rs = params.get("recurrent_states")
    ADVANCES.append(tuple(input_ids.shape))
    if rs:
        bsz, seqlen = input_ids.shape
        assert len(rs) == bsz
        for r in rs:
            r.position += seqlen


KINDS_HYBRID = ("gdn", "gdn", "gdn", "attn", "gdn", "gdn")
KINDS_DENSE = ("attn",) * 4


class FakeModel:
    def __init__(self, kinds=KINDS_HYBRID, hc=0, ple_before=()):
        self.modules = [FakeEmbedding()] + ([FakeExpand(hc)] if hc else [])
        self.first_block_idx = len(self.modules)
        for i, k in enumerate(kinds):
            if i in ple_before:
                self.modules.append(FakePLE(i))
            self.modules.append(FakeBlock(i, k))
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += ([FakeCollapse(hc)] if hc else []) + [FakeNorm(), FakeHead()]
        self.caps = {"recurrent_states": True} if ("gdn" in kinds or ple_before) else {}
        self.config = SimpleNamespace(vocab_size=V, eos_token_id_list=[V - 1])
        self.logit_layer_idx = len(self.modules) - 1
        self.fwd_modules = []
        self.last_kv_module_idx_instance = None

    def prepare_layer_map(self):  # model.py:127-131 (no layer map)
        self.fwd_modules = [(m, 0, idx) for idx, m in enumerate(self.modules)]
        self.last_kv_module_idx_instance = (self.last_kv_module_idx, 0)

    def get_cache_layers(self):
        return [m for top in self.modules for m in top if m.caps.get("kv_cache")]

    def get_recurrent_layers(self):
        return [m for top in self.modules for m in top if m.caps.get("recurrent_cache")]

    @property
    def _get_prefetch_layers(self):
        return [m for top in self.modules for m in top if m.caps.get("prefetch_ids")]

    def prepare_inputs(self, input_ids, params):  # qwen3_5.py:498-501
        assert "past_len" in params and "cache" in params
        if self.caps.get("recurrent_states"):
            fake_prepare_for_recurrence(input_ids, params, self)
        return input_ids

    def gdn_layers(self):
        return [m for m in self.get_recurrent_layers()]

    # exllamav3's own entry points (model.py:368-441 + model_ls.py:329-361),
    # which parity_check.reference_run drives
    def prefill(self, input_ids, params):
        with _torch().inference_mode():  # model.py:367 @torch.inference_mode
            self._prefill(input_ids, params)

    def _prefill(self, input_ids, params):
        x = self.prepare_inputs(input_ids, params)
        for m in self._get_prefetch_layers:
            m.prefetch(x, params)
        for module, inst, idx in self.fwd_modules:
            params["layer_instance"] = inst
            pf = (idx, inst) == self.last_kv_module_idx_instance
            params["prefill"] = pf
            x = module.forward(module.prepare_for_device(x, params), params)
            if pf:
                break
        del params["prefill"]
        fake_advance_recurrent_states(input_ids, params, self)

    def forward(self, input_ids, params):
        with _torch().inference_mode():  # model.py:419 @torch.inference_mode
            return self._forward(input_ids, params)

    def _forward(self, input_ids, params):
        x = self.prepare_inputs(input_ids, params)
        for m in self._get_prefetch_layers:
            m.prefetch(x, params)
        for module, inst, idx in self.fwd_modules:
            params["layer_instance"] = inst
            x = module.forward(module.prepare_for_device(x, params), params)
        fake_advance_recurrent_states(input_ids, params, self)
        return x


MAXTOK = 256


def _node(kinds, start, end, rank, total, max_batch_size=1, prefill_chunk=2048,
          rebase=False, hc=0, ple_before=(), **links):
    model = FakeModel(kinds, hc=hc, ple_before=ple_before)
    _select_slice(model, start, end, rank, total, rebase=rebase)
    model.prepare_layer_map()
    cache = make_cache(FakeCache, model, MAXTOK, max_batch_size=max_batch_size)
    return SliceWorker.from_parts(
        model=model, cache=cache, rank=rank, total=total, config=model.config,
        device="cpu", max_num_tokens=MAXTOK, prefill_chunk=prefill_chunk,
        advance_fn=fake_advance_recurrent_states, **links)


def reference_generate(prompt, n, kinds=KINDS_HYBRID, hc=0, ple_before=()):
    with _torch().inference_mode():
        return _reference_generate(prompt, n, kinds, hc, ple_before)


def _reference_generate(prompt, n, kinds, hc, ple_before):
    """The monolithic model driven the way exllamav3's own Model.prefill /
    Model.forward drive it (model.py:368-441 + model_ls.py:329-361): ONE
    persistent state list, prefill of prompt[:-1] with the prefill flag on the
    last KV module, then one forward per token. Greedy over the cropped vocab."""
    t = _torch()
    model = FakeModel(kinds, hc=hc, ple_before=ple_before)
    model.prepare_layer_map()
    cache = FakeCache(model, MAXTOK, max_batch_size=1)
    states = [cache.get_new_state()] if model.caps else None

    def params(past):
        p = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, MAXTOK),
             "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        return p

    ids = t.tensor([prompt])
    S = len(prompt)
    if S > 1:
        p = params(0)
        x = model.prepare_inputs(ids[:, :-1], p)
        for m in model._get_prefetch_layers:  # model.py:385-386
            m.prefetch(x, p)
        for module, inst, idx in model.fwd_modules:
            p["layer_instance"] = inst
            pf = (idx, inst) == model.last_kv_module_idx_instance
            p["prefill"] = pf
            x = module.forward(module.prepare_for_device(x, p), p)
            if pf:
                break
        del p["prefill"]
        fake_advance_recurrent_states(ids[:, :-1], p, model)
    pos, tok, toks, rows = S - 1, prompt[-1], [], []
    for _ in range(n):
        p = params(pos)
        x = model.prepare_inputs(t.tensor([[tok]]), p)
        for m in model._get_prefetch_layers:
            m.prefetch(x, p)
        for module, inst, idx in model.fwd_modules:
            p["layer_instance"] = inst
            x = module.forward(module.prepare_for_device(x, p), p)
        fake_advance_recurrent_states(t.tensor([[tok]]), p, model)
        row = x[0, -1, :V].float()
        tok = int(t.argmax(row))
        toks.append(tok)
        rows.append(row.numpy().copy())
        pos += 1
    return toks, np.stack(rows)


def _gen(worker, prompt, n, settings=None, return_logits=True, **kw):
    toks, rows = [], []
    with closing(worker.generate(prompt, settings or SamplingSettings(), n,
                                 return_logits=return_logits, **kw)) as g:
        for tok, row in g:
            toks.append(tok)
            if row is not None:
                rows.append(np.asarray(row))
    return toks, (np.stack(rows) if rows else None)


PROMPTS = ([3], [3, 7], [1, 5, 2, 8, 0, 4, 9])


# --------------------------------------------------------------------------
# single node: the recurrent-state contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("chunk", [2048, 3])
def test_single_node_matches_the_monolithic_reference(prompt, chunk):
    w = _node(KINDS_HYBRID, 0, 6, 0, 1, prefill_chunk=chunk)
    toks, rows = _gen(w, prompt, 8)
    ref_toks, ref_rows = reference_generate(prompt, 8)
    assert toks == ref_toks
    np.testing.assert_array_equal(rows, ref_rows)
    assert all(t < V for t in toks)  # the pad columns were cropped


def test_decode_reuses_the_prefill_state_object_and_positions_line_up():
    w = _node(KINDS_HYBRID, 0, 6, 0, 1)
    prompt = [1, 5, 2, 8]
    _gen(w, prompt, 5)
    gdn = w.model.get_recurrent_layers()[0]
    ids = {sid for sid, *_ in gdn.seen}
    assert len(ids) == 1 and None not in ids, "one state object for the whole request"
    # prefill: prompt[:-1] at past_len 0 (the last prompt token is NOT fed
    # twice), then one token per decode step at past_len 3, 4, 5, ...
    assert [(pos, past, S) for _, pos, past, S in gdn.seen] == \
        [(0, 0, 3)] + [(3 + i, 3 + i, 1) for i in range(5)]


def test_decode_without_a_state_is_a_clear_error_not_a_stateless_run():
    t = _torch()
    w = _node(KINDS_HYBRID, 0, 6, 0, 1)
    with pytest.raises(BigBrainStateError, match="no recurrent state"):
        w.process(t.tensor([[3]]), past_len=5, prefill=False)


def test_position_mismatch_is_a_clear_error():
    t = _torch()
    w = _node(KINDS_HYBRID, 0, 6, 0, 1)
    w.process(t.tensor([[3, 4, 5]]), past_len=0, prefill=True)
    with pytest.raises(BigBrainStateError, match="position"):
        w.process(t.tensor([[6]]), past_len=7, prefill=False)


def test_every_request_gets_a_fresh_state_and_the_slot_is_returned():
    w = _node(KINDS_HYBRID, 0, 6, 0, 1, max_batch_size=1)
    for _ in range(5):  # a 1-slot pool would assert on a leaked slot
        toks, _ = _gen(w, [1, 5, 2], 3)
        assert w.cache.free_list == deque([0])
        assert w._states is None
    # and request N's output never depends on request N-1's state
    assert toks == reference_generate([1, 5, 2], 3)[0]


def test_abandoned_generation_releases_its_state():
    w = _node(KINDS_HYBRID, 0, 6, 0, 1)
    with closing(w.generate([1, 2, 3], SamplingSettings(), 10)) as g:
        next(g)
    assert w._states is None and w.cache.free_list == deque([0])


def test_a_leaked_slot_is_reclaimed():
    t = _torch()
    w = _node(KINDS_HYBRID, 0, 6, 0, 1, max_batch_size=1)
    w.cache.free_list.clear()  # simulate a slot leaked by a crashed request
    w.process(t.tensor([[1, 2]]), past_len=0, prefill=True)
    assert w._states is not None and w._states[0].slot == 0
    w.discard_states()
    assert w.cache.free_list == deque([0])


def test_state_create_and_release_are_safe_outside_inference_mode():
    """The pools are inference tensors (FakeCache, like exllamav3's loader):
    creating a state clears its slot in place, which raises outside
    inference_mode -- the live 2026-10-11 23:20 failure. Every state entry
    point must therefore enter inference_mode itself, not rely on its caller."""
    t = _torch()
    assert not t.is_inference_mode_enabled()
    w = _node(KINDS_HYBRID, 0, 6, 0, 1, max_batch_size=2)
    st = w._new_state(1)  # direct, from plain (non-inference) code
    assert st.slot == 1 and list(w.cache.free_list) == [0]
    w._states = [st]
    w.discard_states()
    assert sorted(w.cache.free_list) == [0, 1]
    # begin -> process -> end/RESET-style discards, all from plain code
    w.begin_request(1, SamplingSettings(), slot=1)
    w.process(t.tensor([[1, 2, 3]]), past_len=0, prefill=True)
    w.end_request(1)
    assert w._states is None and sorted(w.cache.free_list) == [0, 1]


def test_dense_model_never_touches_the_recurrent_api():
    ADVANCES.clear()
    w = _node(KINDS_DENSE, 0, 4, 0, 1)
    toks, rows = _gen(w, [1, 5, 2], 4)
    assert w._states is None and not w.recurrent and ADVANCES == []
    ref_toks, ref_rows = reference_generate([1, 5, 2], 4, kinds=KINDS_DENSE)
    assert toks == ref_toks
    np.testing.assert_array_equal(rows, ref_rows)


def test_failed_chunk_discards_the_half_advanced_state():
    t = _torch()
    w = _node(KINDS_HYBRID, 0, 6, 0, 1)
    w.model.get_recurrent_layers()[1].fail_next = True
    with pytest.raises(RuntimeError, match="injected"):
        w.process(t.tensor([[1, 2]]), past_len=0, prefill=True)
    assert w._states is None and w.cache.free_list == deque([0])


# --------------------------------------------------------------------------
# sampler (torch)
# --------------------------------------------------------------------------

def test_sampler_greedy_and_top_k_one_are_argmax():
    t = _torch()
    from tools.bigbrain.worker import Sampler
    row = t.tensor([0.1, 2.0, -1.0, 1.9])
    assert Sampler(SamplingSettings()).sample(row) == 1
    assert Sampler(SamplingSettings(temperature=1.5, top_k=1, seed=3)).sample(row) == 1


def test_sampler_seed_is_reproducible_and_filters_hold():
    t = _torch()
    from tools.bigbrain.worker import Sampler
    row = t.linspace(-2, 2, 50)
    s = SamplingSettings(temperature=1.0, top_k=5, seed=11)
    a = Sampler(s)
    b = Sampler(s)
    seq_a = [a.sample(row) for _ in range(40)]
    seq_b = [b.sample(row) for _ in range(40)]
    assert seq_a == seq_b
    assert set(seq_a) <= set(range(45, 50))  # top-5 only
    tiny_p = Sampler(SamplingSettings(temperature=1.0, top_p=0.01, seed=1))
    assert {tiny_p.sample(row) for _ in range(20)} == {49}
    minp = Sampler(SamplingSettings(temperature=1.0, min_p=0.9, seed=2))
    assert {minp.sample(row) for _ in range(20)} <= {48, 49}


# --------------------------------------------------------------------------
# the pipeline over real loopback links
# --------------------------------------------------------------------------

def _links_line(n):
    """Loopback links for an n-node line: activation hops i -> i+1 and the
    token link last -> 0. Returns per-rank kwargs for _node()."""
    kw = [dict() for _ in range(n)]
    for i in range(n - 1):
        lst = BigBrainLink()
        port = lst.listen_inbound(0, bind_host="127.0.0.1")
        out = BigBrainLink().connect("127.0.0.1", port)
        lst.accept_peer(timeout_s=5.0)
        kw[i]["outbound"] = out
        kw[i + 1]["inbound"] = lst
    tl = BigBrainLink()
    port = tl.listen_inbound(0, bind_host="127.0.0.1")
    up = BigBrainLink().connect("127.0.0.1", port)
    tl.accept_peer(timeout_s=5.0)
    kw[0]["token_link"] = tl
    kw[n - 1]["token_link"] = up
    return kw


class _Cluster:
    def __init__(self, kinds, bounds, **node_kw):
        n = len(bounds)
        links = _links_line(n)
        self.nodes = [_node(kinds, a, b, r, n, **node_kw, **links[r])
                      for r, (a, b) in enumerate(bounds)]
        self.results = {}
        self.threads = []
        for w in self.nodes[1:]:
            w.start_recv()
            th = threading.Thread(target=self._serve, args=(w,), daemon=True)
            th.start()
            self.threads.append(th)

    def _serve(self, w):
        self.results[w.rank] = w.serve(idle_dead_s=10.0)

    @property
    def head(self):
        return self.nodes[0]

    def settle(self, timeout=5.0):
        """Wait until every node has released its request state."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(w._states is None and len(w.cache.free_list) == w.cache.num_slots
                   for w in self.nodes):
                return
            time.sleep(0.01)

    def shutdown(self):
        self.head.send_terminator()
        for th in self.threads:
            th.join(timeout=10)
        for w in self.nodes:
            w.close()


@pytest.fixture
def cluster2():
    c = _Cluster(KINDS_HYBRID, [(0, 3), (3, 6)])
    yield c
    c.shutdown()


@pytest.mark.parametrize("bounds", [[(0, 3), (3, 6)], [(0, 1), (1, 4), (4, 6)]])
def test_pipeline_matches_the_monolithic_reference(bounds):
    _torch()
    c = _Cluster(KINDS_HYBRID, bounds, prefill_chunk=2)
    try:
        for prompt in PROMPTS + ([2, 2, 2, 2, 2],):  # several requests back-to-back
            toks, rows = _gen(c.head, prompt, 8)
            ref_toks, ref_rows = reference_generate(prompt, 8)
            assert toks == ref_toks
            np.testing.assert_array_equal(rows, ref_rows)
        c.settle()  # the last request's 'end' reaches downstream nodes asynchronously
        for w in c.nodes:
            assert w.cache.free_list == deque([0])
    finally:
        c.shutdown()
    assert all(r == "terminator" for r in c.results.values())


def test_pipeline_downstream_state_is_kept_per_request(cluster2):
    _gen(cluster2.head, [1, 5, 2, 8], 4)
    last = cluster2.nodes[1]
    gdn = last.model.get_recurrent_layers()[0]
    assert len({sid for sid, *_ in gdn.seen}) == 1
    assert [(pos, past, S) for _, pos, past, S in gdn.seen] == \
        [(0, 0, 3)] + [(3 + i, 3 + i, 1) for i in range(4)]
    _gen(cluster2.head, [4, 4], 2)
    assert len({sid for sid, *_ in gdn.seen}) == 2  # a NEW state for request 2


def test_pipeline_sampling_happens_on_the_last_node_with_the_request_settings(cluster2):
    s = SamplingSettings(temperature=1.3, top_k=6, top_p=0.95, seed=1234)
    pipe_toks, _ = _gen(cluster2.head, [1, 5, 2], 12, settings=s, return_logits=False)
    single = _node(KINDS_HYBRID, 0, 6, 0, 1)
    solo_toks, _ = _gen(single, [1, 5, 2], 12, settings=s, return_logits=False)
    assert pipe_toks == solo_toks
    assert pipe_toks != reference_generate([1, 5, 2], 12)[0]  # it really sampled


def test_downstream_error_fails_the_request_and_the_chain_recovers(cluster2):
    last = cluster2.nodes[1]
    last.model.get_recurrent_layers()[0].fail_next = True
    with pytest.raises(BigBrainRemoteError, match="injected GDN failure"):
        _gen(cluster2.head, [1, 5, 2], 4)
    toks, rows = _gen(cluster2.head, [1, 5, 2], 4)
    ref_toks, ref_rows = reference_generate([1, 5, 2], 4)
    assert toks == ref_toks
    np.testing.assert_array_equal(rows, ref_rows)


def test_reset_discards_downstream_states(cluster2):
    head, last = cluster2.nodes
    with closing(head.generate([1, 2, 3], SamplingSettings(), 10)) as g:
        next(g)
        # mid-request: the last node holds a live state for this request
        assert last._states is not None
        head.outbound.reset("operator abort")
        deadline = time.monotonic() + 5
        while last._states is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert last._states is None and last.cache.free_list == deque([0])


def test_stale_tokens_from_an_aborted_request_are_dropped(cluster2):
    head, last = cluster2.nodes
    rid = 777
    # A token + logits of an OLD request arrive first; the head must skip them.
    last.token_link.send_tokens([9], request_id=rid - 1, position=4)
    last.token_link.send_logits(np.zeros(V, np.float32), request_id=rid, position=4)
    last.token_link.send_tokens([5], request_id=rid, position=4)
    tok, row = head._await_token(rid, want_logits=True, timeout_s=5.0)
    assert tok == 5 and row.shape == (V,)


def test_a_dead_token_link_fails_the_wait_fast(cluster2):
    head, last = cluster2.nodes
    last.token_link.close()
    with pytest.raises(Exception):
        head._await_token(1, want_logits=False, timeout_s=5.0)


# --------------------------------------------------------------------------
# the multi-node checklist: global layer indices, one slot allocator, the
# lifecycle on every node, token ids + hyper-connection streams on the wire
# --------------------------------------------------------------------------

def test_slice_keeps_global_layer_indices_by_default():
    _torch()
    w = _node(KINDS_HYBRID, 3, 6, 1, 2)
    blocks = [m for m in w.model.modules if isinstance(m, FakeBlock)]
    assert [b.layer_idx for b in blocks] == [3, 4, 5]
    assert [b.attn.layer_idx for b in blocks] == [3, 4, 5]
    # each stage's Cache covers only the modules it loaded, keyed globally
    assert set(w.cache.layers) == {(3, 0)}
    assert set(w.cache.recurrent_layers) == {(4, 0), (5, 0)}
    assert w.model.last_kv_module_idx == 2  # a module-LIST index: [b3, b4, b5, norm, head]


def test_negative_index_modules_travel_with_the_following_block():
    """PLE layers (layer_idx -(i+1), placed before block i) are not blocks:
    slice bounds count decoder blocks only, and a PLE stays with its block."""
    _torch()
    m = FakeModel(KINDS_HYBRID, hc=2, ple_before=(3,))
    n = _select_slice(m, 3, 6, rank=1, total=2)
    assert n == 3
    assert [type(x).__name__ for x in m.modules] == [
        "FakePLE", "FakeBlock", "FakeBlock", "FakeBlock", "FakeCollapse", "FakeNorm", "FakeHead"]
    assert m.modules[0].layer_idx == -4
    m = FakeModel(KINDS_HYBRID, hc=2, ple_before=(3,))
    _select_slice(m, 0, 3, rank=0, total=2)
    assert not any(isinstance(x, FakePLE) for x in m.modules)


def test_a_layer_map_cannot_be_sliced():
    m = _mock_model(8)
    m.config = SimpleNamespace(layer_map=None, layer_map_str="0-3,2-7")
    with pytest.raises(ValueError, match="layer map"):
        _select_slice(m, 0, 4, rank=0, total=2)
    _select_slice(m, 0, 8, rank=0, total=1)  # the whole model is fine


def test_hyper_connection_sites_are_relinked_over_the_kept_modules():
    calls = []

    class Site:
        layer_idx = None

        @staticmethod
        def link_sites(modules):
            calls.append([getattr(x, "key", None) for x in modules])

    m = _mock_model(4)
    m.modules.insert(1, Site())
    _select_slice(m, 2, 4, rank=1, total=2)
    assert calls == []  # this node holds no site
    m = _mock_model(4)
    m.modules[2].attn = Site()  # block 1 carries a site
    m.modules[2].attn.layer_idx = 1
    _select_slice(m, 0, 2, rank=0, total=2)
    assert calls == [["model.embed_tokens", "model.layers.0", "model.layers.1"]]


def test_the_head_assigns_the_slot_for_every_node():
    _torch()
    c = _Cluster(KINDS_HYBRID, [(0, 3), (3, 6)], max_batch_size=3)
    try:
        head, last = c.nodes
        head.slot = 2
        with closing(head.generate([1, 2, 3], SamplingSettings(), 4)) as g:
            next(g)
            assert head._states[0].slot == 2 and last._states[0].slot == 2
        deadline = time.monotonic() + 5  # the 'end' reaches the last node asynchronously
        while sorted(last.cache.free_list) != [0, 1, 2] and time.monotonic() < deadline:
            time.sleep(0.01)
        for w in c.nodes:
            assert sorted(w.cache.free_list) == [0, 1, 2]
    finally:
        c.shutdown()


@pytest.mark.parametrize("bounds,ple", [([(0, 3), (3, 6)], (4,)), ([(0, 1), (1, 4), (4, 6)], (1, 5))])
def test_token_ids_travel_with_the_hidden_state(bounds, ple):
    """A PLE layer on a downstream node reads params['input_ids']: without the
    ids on the wire it would silently see zeros (ple.py:503-506)."""
    _torch()
    c = _Cluster(KINDS_HYBRID, bounds, ple_before=ple, prefill_chunk=3)
    try:
        for prompt in ([1, 5, 2, 8, 0, 4, 9], [7]):
            toks, rows = _gen(c.head, prompt, 6)
            ref_toks, ref_rows = reference_generate(prompt, 6, ple_before=ple)
            assert toks == ref_toks
            np.testing.assert_array_equal(rows, ref_rows)
        ples = [m for w in c.nodes[1:] for m in w.model.modules if isinstance(m, FakePLE)]
        assert ples and all(p.prefetched for p in ples)
        assert ples[-1].prefetched[0] == [1, 5, 2]  # the first prefill chunk (chunk 3)
        assert [7] in ples[-1].prefetched  # the 1-token request's decode-step input
    finally:
        c.shutdown()


def test_hyper_connection_streams_cross_the_pipe():
    _torch()
    c = _Cluster(KINDS_HYBRID, [(0, 2), (2, 4), (4, 6)], hc=4, ple_before=(2,))
    try:
        toks, rows = _gen(c.head, [3, 1, 4, 1, 5], 6)
        ref_toks, ref_rows = reference_generate([3, 1, 4, 1, 5], 6, hc=4, ple_before=(2,))
        assert toks == ref_toks
        np.testing.assert_array_equal(rows, ref_rows)
    finally:
        c.shutdown()


def test_the_rebase_fallback_still_matches():
    _torch()
    c = _Cluster(KINDS_HYBRID, [(0, 3), (3, 6)], rebase=True)
    try:
        assert [m.layer_idx for m in c.nodes[1].model.modules if isinstance(m, FakeBlock)] == [0, 1, 2]
        toks, rows = _gen(c.head, [1, 5, 2], 5)
        ref_toks, ref_rows = reference_generate([1, 5, 2], 5)
        assert toks == ref_toks
        np.testing.assert_array_equal(rows, ref_rows)
    finally:
        c.shutdown()


# --------------------------------------------------------------------------
# CLI: every flag the controller passes still parses; dumps roundtrip
# --------------------------------------------------------------------------

def _controller_commands():
    """The real ClusterPlan.build_commands() output (the registry module may be
    absent when only the node runtime is checked out; a stub stands in for the import only)."""
    if "tools.bigbrain.registry" not in sys.modules:
        try:
            import tools.bigbrain.registry  # noqa: F401
        except ImportError:
            sys.modules["tools.bigbrain.registry"] = types.ModuleType("tools.bigbrain.registry")
    from tools.bigbrain.controller import ClusterPlan, NodeSpec
    plan = ClusterPlan(model_dir="/m/Qwen3.8-27B", prompt="Hi there", max_new=16,
                       mem_fraction=0.9, token_port=16903, nodes=[
                           NodeSpec(0, "10.0.0.11", 0, 0, 22, 0),
                           NodeSpec(1, "10.0.0.11", 1, 22, 44, 16900),
                           NodeSpec(2, "10.0.0.12", 0, 44, 64, 16901)])
    return plan.build_commands()


def test_every_controller_flag_parses():
    parser = build_arg_parser()
    for cmd in _controller_commands():
        argv = shlex.split(cmd.split(" -m tools.bigbrain.worker ", 1)[1])
        args = parser.parse_args(argv)
        assert args.model_dir == "/m/Qwen3.8-27B"
        assert args.total == 3
    # The new flags have safe defaults.
    args = parser.parse_args(["--model-dir", "m", "--slice", "0", "2", "--rank", "0",
                              "--total", "1"])
    assert args.prefill_chunk == 2048 and args.max_batch_size == 1
    assert args.rebase_layers is False  # global layer indices by default
    assert args.temperature == 0.0 and args.dump is None and not args.ignore_eos


# --------------------------------------------------------------------------
# parity_check (the tool the architect runs live)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("opts", [{}, {"ple_before": (4,), "hc": 4}, {"kinds": KINDS_DENSE}])
def test_parity_reference_run_agrees_with_the_pipeline(opts):
    """reference_run drives the WHOLE model through exllamav3's own
    Model.prefill/forward API; the pipeline must match it exactly."""
    _torch()
    from tools.bigbrain.parity_check import reference_run
    kinds = opts.get("kinds", KINDS_HYBRID)
    kw = {k: v for k, v in opts.items() if k != "kinds"}
    model = FakeModel(kinds, **kw)
    model.prepare_layer_map()
    cache = FakeCache(model, MAXTOK, max_batch_size=1)
    prompt = [1, 5, 2, 8, 0, 4, 9]
    ref_toks, ref_rows = reference_run(model, cache, prompt, 8, max_num_tokens=MAXTOK,
                                       prefill_chunk=3, vocab_size=V)
    assert cache.free_list == deque([0])  # its state was released
    assert (ref_toks, ref_rows.tolist()) == (lambda t, r: (t, r.tolist()))(
        *reference_generate(prompt, 8, kinds=kinds, **kw))
    L = len(kinds)
    c = _Cluster(kinds, [(0, L // 3), (L // 3, 2 * L // 3), (2 * L // 3, L)], prefill_chunk=3, **kw)
    try:
        toks, rows = _gen(c.head, prompt, 8)
    finally:
        c.shutdown()
    assert toks == ref_toks
    np.testing.assert_array_equal(rows, ref_rows)


def _dump(tokens, rows, prompt=(1, 2, 3)):
    return {"prompt_ids": list(prompt), "tokens": list(tokens),
            "logits": np.asarray(rows, np.float32), "meta": {}}


def test_parity_compare_verdicts():
    from tools.bigbrain.parity_check import compare_dumps, format_verdict
    rows = np.random.default_rng(0).normal(size=(5, 9)).astype(np.float32)
    toks = [int(r.argmax()) for r in rows]
    v = compare_dumps(_dump(toks, rows), _dump(toks, rows))
    assert v["passed"] and v["max_abs_diff"] == 0.0 and v["logit_steps"] == 5
    assert "VERDICT: PASS" in format_verdict(v)
    # identical tokens, logits off by more than atol -> FAIL
    v = compare_dumps(_dump(toks, rows), _dump(toks, rows + 0.05))
    assert not v["passed"] and "atol" in v["reason"] and v["tokens_identical"]
    # a divergence at step 2: logits compared on steps 0..2 only, top-2 reported
    cand_rows = rows.copy()
    cand_rows[3:] += 7.0  # after the divergence the histories differ: ignored
    cand_toks = toks[:2] + [(toks[2] + 1) % 9] + toks[3:]
    v = compare_dumps(_dump(toks, rows), _dump(cand_toks, cand_rows), atol=1.0)
    assert not v["passed"] and v["first_divergence"] == 2 and v["matched"] == 2
    assert v["logit_steps"] == 3 and v["max_abs_diff"] == 0.0
    assert v["divergence_top2"]["reference"][0][0] == toks[2]
    assert "first divergence at step 2" in format_verdict(v)
    # different prompts are refused outright
    v = compare_dumps(_dump(toks, rows), _dump(toks, rows, prompt=(9,)))
    assert not v["passed"] and "different prompts" in v["reason"]
    # a dump without logits: tokens only
    v = compare_dumps(_dump(toks, rows), _dump(toks, np.zeros((0, 0))))
    assert v["passed"] and v["max_abs_diff"] is None
    # a shorter candidate fails
    v = compare_dumps(_dump(toks, rows), _dump(toks[:4], rows[:4]))
    assert not v["passed"] and "lengths differ" in v["reason"]


def test_parity_compare_cli_exit_codes(tmp_path):
    from tools.bigbrain.parity_check import main
    rows = np.eye(4, dtype=np.float32)
    a, b = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    write_dump(a, [1], [0, 1, 2, 3], rows)
    write_dump(b, [1], [0, 1, 2, 3], rows)
    assert main(["compare", a, b]) == 0
    write_dump(b, [1], [0, 1, 3, 3], rows)
    assert main(["compare", a, b, "--json"]) == 1


def test_parity_node_specs():
    from tools.bigbrain.parity_check import parse_nodes
    assert parse_nodes(["cuda:1:0:32", "cuda:2:32:64"], None, 64) == \
        [("cuda:1", 0, 32), ("cuda:2", 32, 64)]
    assert parse_nodes(None, "cuda:0,cuda:1,cuda:2", 64) == \
        [("cuda:0", 0, 22), ("cuda:1", 22, 43), ("cuda:2", 43, 64)]
    for bad in (["cuda:1:0:30", "cuda:2:31:64"], ["cuda:1:0:32"], ["nonsense"]):
        with pytest.raises(ValueError):
            parse_nodes(bad, None, 64)


def test_parity_pipeline_commands_parse_as_worker_flags():
    from tools.bigbrain.parity_check import build_pipeline_commands
    cmds = build_pipeline_commands(
        "/venv/bin/python", "/m/q", [("cuda:1", 0, 22), ("cuda:2", 22, 43), ("cuda:0", 43, 64)],
        prompt_args=["--prompt", "Hi there"], n=64, dump="/w/pipeline.npz", base_port=17900,
        max_num_tokens=4096, prefill_chunk=512, mem_fraction=0.45, extra=["--rebase-layers"])
    parser = build_arg_parser()
    parsed = [parser.parse_args(c[3:]) for c in cmds]
    assert all(c[:3] == ["/venv/bin/python", "-m", "tools.bigbrain.worker"] for c in cmds)
    a0, a1, a2 = parsed
    assert (a0.rank, a0.total, a0.slice, a0.device) == (0, 3, [0, 22], "cuda:1")
    assert a0.dump == "/w/pipeline.npz" and a1.dump is None and a2.dump is None
    assert (a0.next_host, a0.next_port, a0.token_port) == ("127.0.0.1", 17901, 17900)
    assert (a1.in_port, a1.next_port) == (17901, 17902)
    assert (a2.in_port, a2.token_host, a2.token_port) == (17902, "127.0.0.1", 17900)
    assert all(p.ignore_eos and p.temperature == 0 and p.max_new == 64 for p in parsed)
    assert all(p.prompt == "Hi there" and p.prefill_chunk == 512 and p.rebase_layers
               for p in parsed)
    assert all(p.mem_fraction == 0.45 and p.max_num_tokens == 4096 for p in parsed)


def test_connect_retry_outlasts_a_tunnel_whose_far_side_starts_late():
    """Startup order stops mattering through ssh tunnels: the first connects
    are accepted-then-closed by the 'tunnel' and retried until the far side
    listens (here: the 3rd connect is held open like a real backlog)."""
    import socket as _s
    lsock = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(8)
    port = lsock.getsockname()[1]
    held = []

    def tunnel():
        for i in range(3):
            conn, _ = lsock.accept()
            if i < 2:
                conn.close()  # far side not up yet
            else:
                held.append(conn)  # far side up: the connection stays open

    th = threading.Thread(target=tunnel, daemon=True)
    th.start()
    link = SliceWorker._connect_retry("127.0.0.1", port, watchdog_s=5.0, attempts=10,
                                      retry_sleep_s=0.05, settle_s=0.3)
    try:
        assert link.connected
        th.join(5)
        assert len(held) == 1
    finally:
        link.close()
        for c in held:
            c.close()
        lsock.close()


def test_bind_host_flag_defaults_from_env(monkeypatch):
    args = build_arg_parser().parse_args(["--model-dir", "m", "--slice", "0", "2", "--rank", "1",
                                          "--total", "2"])
    assert args.bind_host == "0.0.0.0" and args.connect_settle == 1.5
    monkeypatch.setenv("BIGBRAIN_BIND", "127.0.0.1")
    args = build_arg_parser().parse_args(["--model-dir", "m", "--slice", "0", "2", "--rank", "1",
                                          "--total", "2"])
    assert args.bind_host == "127.0.0.1"


def test_dump_roundtrip(tmp_path):
    p = tmp_path / "run.npz"
    rows = np.arange(6, dtype=np.float32).reshape(2, 3)
    write_dump(str(p), prompt_ids=[1, 2], tokens=[5, 6], logits=rows,
               meta={"model_dir": "/m", "slices": [[0, 3]]})
    d = read_dump(str(p))
    assert d["prompt_ids"] == [1, 2] and d["tokens"] == [5, 6]
    np.testing.assert_array_equal(d["logits"], rows)
    assert d["meta"]["slices"] == [[0, 3]]
