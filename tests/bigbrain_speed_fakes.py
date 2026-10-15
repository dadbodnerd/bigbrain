# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Fake per-node layers for the BigBrain multi-sequence + speculation tests.
Not a test module (no test_ prefix); needs torch (CPU), no exllamav3, no GPU.

The "model" is a deterministic function of the token history:

  embedding -> N blocks -> RMS-ish norm -> lm_head (padded vocab)

and every block's attention is one of the two kinds gpt-oss alternates:

  FULL     reads the paged KV cache: it writes its input rows at
           block_table[b][pos // 256] * 256 + pos % 256 and attends to the mean
           of every earlier row of THIS sequence -- so a wrong block table, a
           wrong cache_seqlens (a rollback that did not happen) or another
           slot's page leaking in changes the output.
  SWA      a sliding window over a RECURRENT STATE (SWAState stand-in): it
           writes at state.position and ignores cache_seqlens, like the real
           SlidingAttention -- so a state that was not rewound after a
           rejected draft writes at the wrong place and changes the output.

Every operation is done per token on 1-D vectors, so the value computed for a
token never depends on how many other rows share its batch (the real kernels
make no such promise; see ARCHITECTURE.md). That is what lets the tests assert
EXACT equality between batched/speculative runs and the plain run.

The contract follows exllamav3 1.6.x: flash_attn params with either
batch_shape+past_len (the single-sequence path) or block_table+cache_seqlens
(the paged batch path), SWAState.rewind, advance_recurrent_states, Cache slots.
"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

H = 6          # hidden size
V = 12         # real vocab
VPAD = 16      # lm_head width: the padded columns carry a large bias, so an
               # argmax that forgets to crop to vocab_size picks a pad id
PAGE = 256
WIN = 4        # the SWA window (tokens)


def _torch():
    return pytest.importorskip("torch")


def _w(seed, *shape, scale=0.6):
    t = _torch()
    g = t.Generator().manual_seed(seed)
    return t.randn(*shape, generator=g, dtype=t.float32) * scale


# --------------------------------------------------------------------------
# state + cache
# --------------------------------------------------------------------------

class FakeSWAState:
    """SWAState stand-in (modules/sliding_attn.py:45): a slot handle + a
    position; rewind moves the position back inside the stored window."""
    made = 0
    rewinds = []   # (slot, n) log
    guaranteed_rollback = PAGE   # SWAState.guaranteed_rollback: rewinds in place within its window

    def __init__(self, cache, slot, position, clear=True, stashed=None, test_state=False):
        assert test_state or position == 0, "State must be new"
        self.cache, self.slot, self.position = cache, slot, position
        self.window_beg = 0
        self.last_history = 0
        FakeSWAState.made += 1
        if clear:
            for layer in cache.recurrent_layers.values():
                layer.clear(slot)

    def rewind(self, num_tokens):
        assert 0 <= num_tokens <= self.position - self.window_beg, "rewind beyond the stored window"
        self.position -= num_tokens
        self.last_history = 0
        FakeSWAState.rewinds.append((self.slot, num_tokens))

    def post_advance(self):
        pass


class FakeSWALayerState:
    def __init__(self, n_slots, max_pos):
        self.buf = _torch().zeros(n_slots, max_pos, H)

    def clear(self, slot):
        self.buf[slot].zero_()


class FakeCacheLayer:
    def __init__(self, max_tokens):
        self.v = _torch().zeros(max_tokens, H)


class FakeCache:
    """Pools are INFERENCE tensors (allocated inside inference_mode, like the
    loader does), so an in-place write outside inference_mode raises."""

    def __init__(self, model, max_num_tokens, layer_type=None, max_batch_size=16,
                 max_history=0, **kwargs):
        t = _torch()
        self.model, self.max_num_tokens = model, max_num_tokens
        self.max_history = max_history
        with t.inference_mode():
            self.layers = {(m.layer_idx, 0): FakeCacheLayer(max_num_tokens)
                           for m in model.get_cache_layers()}
            self.recurrent_layers = {(m.layer_idx, 0): FakeSWALayerState(max_batch_size, max_num_tokens)
                                     for m in model.get_recurrent_layers()}
        self.recurrent_state_cls = model.recurrent_state_cls
        self.num_slots = max_batch_size
        self.free_list = deque(range(max_batch_size))

    def get_new_state(self):
        assert len(self.free_list) > 0, "Cannot create new state: no available slots"
        return self.recurrent_state_cls(self, self.free_list.popleft(), 0, clear=True)

    def release_state(self, state):
        self.free_list.appendleft(state.slot)

    def reset_states(self):
        self.free_list = deque(range(self.num_slots))

    def get_recurrent_layer(self, instance):
        return self.recurrent_layers[instance]


# --------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------

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
    def __init__(self, perturb=0.0, seed=1):
        super().__init__()
        self.E = _w(seed, V, H) + perturb * _w(seed + 77, V, H)

    def forward(self, ids, params):
        if "input_ids" not in params:
            params["input_ids"] = ids
        return self.E[ids.long()].float()


def _tok_full(hist_rows, W):
    return hist_rows.mean(dim=0) @ W


class FakeFullAttn(FakeModule):
    caps = {"kv_cache": True}

    def __init__(self, idx, perturb=0.0):
        super().__init__()
        self.layer_idx = idx
        self.W = _w(200 + idx, H, H) + perturb * _w(900 + idx, H, H)

    def forward(self, x, params):
        t = _torch()
        layer = params["cache"].layers[(self.layer_idx, params.get("layer_instance", 0))]
        bt, seqlens = params["block_table"], params["cache_seqlens"]
        outs = []
        for b in range(x.shape[0]):
            p0 = int(seqlens[b])

            def phys(q):
                return int(bt[b, q // PAGE]) * PAGE + q % PAGE
            for j in range(x.shape[1]):
                layer.v[phys(p0 + j)] = x[b, j]
            rows = []
            for j in range(x.shape[1]):
                idx = [phys(q) for q in range(p0 + j + 1)]
                rows.append(_tok_full(layer.v[idx], self.W))
            outs.append(t.stack(rows))
        return t.stack(outs)


class FakeSWA(FakeModule):
    caps = {"recurrent_cache": True}
    seen = []        # (forward width, recurrent_history) of every forward

    def __init__(self, idx, perturb=0.0):
        super().__init__()
        self.layer_idx = idx
        self.layer_state_cls = FakeSWALayerState
        self.W = _w(400 + idx, H, H) + perturb * _w(950 + idx, H, H)

    def forward(self, x, params):
        t = _torch()
        rs = params.get("recurrent_states")
        assert rs, "SlidingAttention.forward() called in flash_attn mode with no recurrent states"
        # Emulates what a GatedDeltaNet layer would do wrong: a history forward of ONE token (or wider
        # than the cache's history) leaves the conv ring head stale -- exllamav3's generator never does it
        if params.get("recurrent_history"):
            assert x.shape[1] > 1, "recurrent_history on a one-token forward (stale GDN conv window)"
            assert x.shape[1] <= params["cache"].max_history + 1, "history window wider than Cache(max_history)"
        FakeSWA.seen.append((int(x.shape[1]), bool(params.get("recurrent_history"))))
        layer = rs[0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance", 0)))
        assert len(rs) == x.shape[0], "one recurrent state per batch row"
        outs = []
        for b in range(x.shape[0]):
            st = rs[b]
            p0 = st.position    # the state's own position, NOT cache_seqlens
            buf = layer.buf[st.slot]
            for j in range(x.shape[1]):
                buf[p0 + j] = x[b, j]
            rows = []
            for j in range(x.shape[1]):
                lo = max(0, p0 + j - WIN + 1)
                rows.append(_tok_full(buf[lo:p0 + j + 1], self.W))
            outs.append(t.stack(rows))
        return t.stack(outs)


class FakeBlock(FakeModule):
    def __init__(self, idx, kind, perturb=0.0):
        super().__init__()
        self.layer_idx = idx
        self.attn = FakeSWA(idx, perturb) if kind == "swa" else FakeFullAttn(idx, perturb)
        self.M = _w(300 + idx, H, H) + perturb * _w(980 + idx, H, H)
        self.modules = [self.attn]

    def forward(self, x, params):
        t = _torch()
        y = self.attn.forward(x, params)
        if params.get("prefill"):
            return x   # transformer.py:167: prefill stops after the attention
        x = x + y
        out = t.empty_like(x)
        for b in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[b, j] = x[b, j] + 0.1 * t.tanh(x[b, j] @ self.M)
        return out


class FakeNorm(FakeModule):
    def forward(self, x, params):
        t = _torch()
        out = t.empty_like(x)
        for b in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[b, j] = x[b, j] / (x[b, j].norm(dim=-1, keepdim=True) + 1e-6)
        return out


class FakeHead(FakeModule):
    caps = {"logits_output": True}

    def __init__(self, perturb=0.0):
        super().__init__()
        t = _torch()
        self.W = _w(7, H, VPAD) + perturb * _w(8, H, VPAD)
        self.bias = t.zeros(VPAD)
        self.bias[V:] = 100.0   # padding columns: must be cropped away

    def forward(self, x, params):
        t = _torch()
        out = t.empty(x.shape[0], x.shape[1], VPAD)
        for b in range(x.shape[0]):
            for j in range(x.shape[1]):
                out[b, j] = x[b, j] @ self.W + self.bias
        return out


# --------------------------------------------------------------------------
# the arch hooks (qwen3_5.py / gpt_oss.py prepare_inputs)
# --------------------------------------------------------------------------

def fake_prepare_for_recurrence(input_ids, params, model):
    """cache/recurrent_util.py:18-64"""
    batch_shape = params.get("batch_shape")
    cache_seqlens = params.get("cache_seqlens")
    rs = params.get("recurrent_states")
    if batch_shape is not None:
        bsz, _ = batch_shape
        past_len = params.get("past_len", 0)
        if past_len > 0:
            if rs is None:
                raise ValueError("Past length given, but no previous state for recurrence in params")
            if not isinstance(rs, list):
                rs = [rs]
                params["recurrent_states"] = rs
            assert all(r.position == past_len for r in rs), "recurrent states don't match input past_len"
        else:
            if rs is None:
                rs = [params["cache"].get_new_state() for _ in range(bsz)]
                params["recurrent_states"] = rs
            else:
                assert all(r.position == 0 for r in rs), "recurrent states don't match input past_len"
    elif cache_seqlens is not None:
        pass   # (empty) states must be provided with cache_seqlens
    else:
        if rs is not None:
            raise ValueError("recurrent_states given without bsz and seqlens")
    if rs is not None:
        params["recurrent_slots"] = tuple(r.slot for r in rs)


def fake_prepare_for_attn(input_ids, params):
    """modules/attn.py:95-133 (flash_attn)"""
    t = _torch()
    bsz, _ = input_ids.shape
    if "batch_shape" in params:
        cache_bsz, max_seq = params["batch_shape"]
        assert cache_bsz >= bsz
        assert bsz * max_seq <= params["cache"].max_num_tokens, "Cache too small for batch shape"
        past_len = params.get("past_len")
        params["block_table"] = t.arange(bsz * (max_seq // PAGE), dtype=t.int32).view(bsz, max_seq // PAGE)
        params["cache_seqlens"] = t.tensor([past_len], dtype=t.int32).repeat(bsz)
    elif "block_table" in params:
        assert params.get("cache_seqlens") is not None
    else:
        raise ValueError("flash_attn needs batch_shape or block_table")
    return input_ids


def fake_advance_recurrent_states(input_ids, params, model):
    """cache/recurrent_util.py:66-75"""
    rs = params.get("recurrent_states")
    history = params.get("recurrent_history")
    if rs:
        bsz, seqlen = input_ids.shape
        assert len(rs) == bsz
        for r in rs:
            r.position += seqlen
            r.last_history = (seqlen - 1) if history else 0
            r.post_advance()


KINDS_MIXED = ("swa", "full", "swa", "full")     # gpt-oss: alternating
KINDS_FULL = ("full",) * 4


class FakeModel:
    def __init__(self, kinds=KINDS_MIXED, perturb=0.0):
        self.kinds = tuple(kinds)
        self.modules = [FakeEmbedding(perturb)]
        self.first_block_idx = 1
        for i, k in enumerate(kinds):
            self.modules.append(FakeBlock(i, k, perturb))
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [FakeNorm(), FakeHead(perturb)]
        has_swa = "swa" in kinds
        self.caps = {"recurrent_states": True} if has_swa else {}
        self.recurrent_state_cls = FakeSWAState if has_swa else None
        self.config = SimpleNamespace(vocab_size=V, eos_token_id_list=[V - 1])
        self.logit_layer_idx = len(self.modules) - 1
        self.fwd_modules = []
        self.last_kv_module_idx_instance = None

    def prepare_layer_map(self):
        self.fwd_modules = [(m, 0, idx) for idx, m in enumerate(self.modules)]
        self.last_kv_module_idx_instance = (self.last_kv_module_idx, 0)

    def get_cache_layers(self):
        return [m for top in self.modules for m in top if m.caps.get("kv_cache")]

    def get_recurrent_layers(self):
        return [m for top in self.modules for m in top if m.caps.get("recurrent_cache")]

    @property
    def _get_prefetch_layers(self):
        return []

    def prepare_inputs(self, input_ids, params):    # gpt_oss.py:prepare_inputs
        if self.caps.get("recurrent_states"):
            fake_prepare_for_recurrence(input_ids, params, self)
        return fake_prepare_for_attn(input_ids, params)

    # exllamav3's own entry points (model.py:368-441) -- the draft backend and
    # the single-sequence reference drive these
    def prefill(self, input_ids, params):
        with _torch().inference_mode():
            x = self.prepare_inputs(input_ids, params)
            for module, inst, idx in self.fwd_modules:
                params["layer_instance"] = inst
                pf = (idx, inst) == self.last_kv_module_idx_instance
                params["prefill"] = pf
                x = module.forward(module.prepare_for_device(x, params), params)
                if pf:
                    break
            params.pop("prefill", None)
            fake_advance_recurrent_states(input_ids, params, self)

    def forward(self, input_ids, params):
        with _torch().inference_mode():
            x = self.prepare_inputs(input_ids, params)
            for module, inst, idx in self.fwd_modules:
                params["layer_instance"] = inst
                x = module.forward(module.prepare_for_device(x, params), params)
            fake_advance_recurrent_states(input_ids, params, self)
            return x


# --------------------------------------------------------------------------
# the oracle: the model's greedy decode with no cache, no pages, no states
# --------------------------------------------------------------------------

def oracle_next(model: FakeModel, toks):
    """Greedy next token after `toks`, recomputed from scratch."""
    t = _torch()
    with t.inference_mode():
        emb = model.modules[0]
        x = [emb.E[int(i)].float() for i in toks]
        for blk in [m for m in model.modules if isinstance(m, FakeBlock)]:
            kind = "swa" if isinstance(blk.attn, FakeSWA) else "full"
            ys = []
            for p in range(len(x)):
                lo = max(0, p - WIN + 1) if kind == "swa" else 0
                ys.append(_tok_full(t.stack(x[lo:p + 1]), blk.attn.W))
            x = [x[p] + ys[p] for p in range(len(x))]
            x = [x[p] + 0.1 * t.tanh(x[p] @ blk.M) for p in range(len(x))]
        norm = model.modules[-2]
        head = model.modules[-1]
        last = x[-1]
        last = last / (last.norm(dim=-1, keepdim=True) + 1e-6)
        logits = last @ head.W + head.bias
        return int(t.argmax(logits[:V]))


def oracle_greedy(model: FakeModel, prompt, n):
    toks = list(prompt)
    out = []
    for _ in range(n):
        nxt = oracle_next(model, toks)
        out.append(nxt)
        toks.append(nxt)
    return out
