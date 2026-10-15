# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/worker.py -- one BigBrain pipeline node.

Loads a CONTIGUOUS slice of an exllamav3 model's transformer layers onto one
local GPU and participates in a cross-machine pipeline-parallel generate. Node
0 owns the tokenizer + Embedding (first slice); the last node owns RMSNorm +
lm_head and does the SAMPLING (only the token id returns to node 0, not the
~1 MB logits row). Between nodes the residual hidden state (1, seq, hidden)
hops over bigbrain/pipe.py (v2 frames: request id + position).

Runs inside a Python environment that has exllamav3 and torch (for example a TabbyAPI venv;
they are imported lazily, so the pure logic stays importable for tests), deployed to a GPU machine and launched by
the cluster controller. Node 0 may instead run bigbrain/server.py (an
OpenAI-compatible endpoint), which drives the same SliceWorker.

Slice-load mechanics (exllamav3 1.4.8 .. 1.6.1):

* Model.modules is a flat list [Embedding, TransformerBlock x N (+ any
  between-block module such as DeepstackEmbed), RMSNorm, lm_head]. The loader
  iterates self.modules, so a process loads exactly the modules in its list.
* For a slice [start,end) of DECODER BLOCKS (modules with a layer_idx >= 0):
  keep those blocks, the modules attached to them (a layer_idx-None module
  such as DeepstackEmbed belongs to the block BEFORE it; a NEGATIVE layer_idx
  module such as Qwen3.8-Flash-Next's PLELayer, -(i+1), to the block AFTER
  it), the prolog on node 0 and the epilog on the last node.
* Layer indices stay GLOBAL (the default since 2026-10-11): the Cache is a
  dict built from the LOADED modules only, keyed by (layer_idx, instance)
  (cache/cache.py:147-173, the same in 1.4.8), and every runtime lookup uses
  the module's own layer_idx -- attention cache.layers[...] (attn.py:973),
  the GDN/PLE/short-conv recurrent pools (gated_delta_net.py:1200,
  ple.py:458) -- so a slice [40,80) simply has cache keys 40..79. Nothing at
  runtime reads the global layer_types; contiguity is asserted only with a
  layer map (model.py:104-146), which cannot be sliced (refused). Keeping
  global indices also keeps every global-index reference valid (e.g. the
  export_state_layers that DFlash/MTP use, transformer.py:152).
  --rebase-layers restores the old 0..N-1 rebase (internally consistent too).
* The model's own module-LIST bookkeeping (last_kv_module_idx,
  logit_layer_idx, first_block_idx) is recomputed for the shorter list, its
  cached layer lists are dropped, and hyper-connection sites are re-linked
  over the kept modules (GatedResidual.link_sites, so a boundary site never
  hands off to a module that lives on another node).
* Cache(model, max_num_tokens, max_batch_size) is attached BEFORE model.load()
  (the loader allocates the attached caches' tensors, including the recurrent
  pools). max_batch_size = recurrent-state SLOTS; serving is batch 1, so the
  default is 1 (each extra slot of a Qwen3.8-27B costs ~6 MB per GDN layer).

The recurrent-state contract (hybrid models: Qwen3.5/3.8 GatedDeltaNet + full
attention), exllamav3 1.6.1:

* A "state" is a SLOT HANDLE: Cache.get_new_state() pops a slot off the
  cache's free_list and builds model.recurrent_state_cls (GDNState) with
  position 0, which zeroes that slot's rows in EVERY recurrent layer's pool
  -- the conv ring AND the recurrent state, both live in GDNLayerState and are
  indexed by the same slot (cache.py:326-337, gated_delta_net.py:116-163 and
  :343-346). The conv state needs no separate handling.
* prepare_for_recurrence (cache/recurrent_util.py:18-81, called by the arch's
  prepare_inputs, qwen3_5.py:498-501): with past_len > 0 it REQUIRES
  params["recurrent_states"] (else ValueError "Past length given, but no
  previous state") and asserts every state's .position == past_len; with
  past_len == 0 it creates states when none are given. It also derives the
  per-call slot tensors (recurrent_slots/_scan) from the states.
* advance_recurrent_states (recurrent_util.py:84-105) runs AFTER the forward
  and does position += seqlen (Model.prefill/forward call it: model.py:382-440).
  This worker drives its own module loop, so it calls it itself.
* A GatedDeltaNet called WITHOUT recurrent_states runs stateless -- zero
  initial state, nothing saved (gated_delta_net.py:1193-1214): silently wrong
  decode output, which is why the state must survive between steps.
* exllamav3's generator keeps ONE state per job for its whole life
  (job.py:1669-1684, passed as [job.recurrent_state.target] on every prefill
  chunk and decode step: job.py:1540, generator.py:1192) and frees it with
  state.free() -> Cache.release_state (slot back on free_list, cache.py:371).

So every node keeps, per request, the state of ITS OWN slice's recurrent
layers: created from its own cache at the request's first chunk (past_len 0),
passed as the same object on every later prefill chunk and decode step,
released when the request ends, a new request begins, or a RESET arrives.
ONE allocator: node 0 assigns the request's slot and sends it in the CONTROL
'begin'; every node builds its state at exactly that slot (the same call as
Cache.get_new_state, with the slot given instead of popped).
Every node holds a state even if its slice has no recurrent layer (a state of
an empty pool), because the arch's prepare_inputs requires one at past_len>0.
The cache's pools are INFERENCE tensors (the loader allocates them under
torch.inference_mode) and creating a state clears its slot in place
(gated_delta_net.py:152 -> :345), so every state entry point here enters
inference_mode itself -- outside it torch raises "Inplace update to inference
tensor outside InferenceMode" (live, 2026-10-11 23:20, parity reference).

Generate flow (mirrors exllamav3's generator, job.py:1381): PREFILL feeds
prompt[:-1] (positions 0..S-2, in --prefill-chunk pieces) through every node;
the last node runs it prefill_ls-style (params["prefill"] on the last KV
module, no norm/head) and discards it. Every hop carries the chunk's INPUT
TOKEN IDS too (a TOKEN frame before the activation): a downstream node passes
them to the arch's prepare_inputs, its prefetch layers and
params["input_ids"] (which the Embedding sets on node 0, embedding.py:185 --
a PLE layer reads them, ple.py:502). Hyper-connection models hop their
(1, seq, hc_mult, hidden) stream stack as-is (pipe.py stream flag). Then each DECODE step feeds one token
-- first the last prompt token at position S-1 -- and the last node samples
the next one. (The v1 flow re-fed the last prompt token after a full-prompt
prefill; harmless for KV attention, but a recurrent layer would consume that
token twice.)

Usage (one-shot; the controller's form):
  python -m bigbrain.worker --model-dir DIR --device cuda:0 \\
      --slice START END --rank R --total T [--in-port P] \\
      [--next-host H --next-port P] [--token-host H --token-port P] \\
      [--prompt "..."] [--max-new N] [--max-num-tokens M] \\
      [--temperature F --top-p F --top-k N --min-p F --seed N] [--ignore-eos] \\
      [--prefill-chunk N] [--max-batch-size N] \\
      [--tokens-out f.json] [--logits-out f.json] [--dump f.npz]
Nodes 1..T-1 serve requests until node 0's terminator / BYE, so the same
command also backs a long-running server.py on node 0.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import queue
import secrets
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .paged import (PROTO, BigBrainStateError, NodeStats, PagedSlots, StepMeta,
                    pack_rows, unpack_rows)
from .pipe import (DEFAULT_MAX_FRAME_BYTES, FLAG_PREFILL, BigBrainLink, BigBrainPeerDead,
                   BigBrainProtocolError, Message, frame_limit)
from .spec import greedy_accept

log = logging.getLogger("bigbrain.worker")


def _torch():
    import torch
    return torch


class BigBrainRemoteError(RuntimeError):
    """A downstream node reported an error for the current request."""


# --------------------------------------------------------------------------
# slice selection
# --------------------------------------------------------------------------

def _walk(module):
    """The module and every descendant: exllamav3 Modules iterate that way
    (modules/module.py:53-56); a plain stand-in yields itself + .attn."""
    if hasattr(type(module), "__iter__"):
        yield from module
    else:
        yield module
        attn = getattr(module, "attn", None)
        if attn is not None:
            yield attn


def _select_slice(model, start: int, end: int, rank: int, total: int,
                  rebase: bool = False) -> int:
    """Filter model.modules to this node's slice of decoder blocks [start,end)
    (see the module docstring for what travels with a block). Layer indices
    stay global unless rebase=True (0..N-1, the pre-2026-10-11 behaviour).
    Returns the number of blocks kept."""
    prolog, groups, post, pre = [], [], [], []
    for m in model.modules:
        idx = getattr(m, "layer_idx", None)
        if idx is None:
            if pre:  # already collecting for the NEXT block
                pre.append(m)
            elif groups:
                post.append(m)  # after a block: belongs to it (DeepstackEmbed)
            else:
                prolog.append(m)
        elif idx < 0:
            if post:
                groups[-1][2].extend(post)
                post = []
            pre.append(m)  # before a block: belongs to it (PLELayer)
        else:
            if post:
                groups[-1][2].extend(post)
                post = []
            groups.append((pre, m, []))
            pre = []
    epilog = post + pre
    if not (0 <= start < end <= len(groups)):
        raise ValueError(f"bad slice [{start},{end}) for a model with {len(groups)} blocks")
    cfg = getattr(model, "config", None)
    if (getattr(cfg, "layer_map", None) or getattr(cfg, "layer_map_str", None)) \
            and not (start == 0 and end == len(groups)):
        raise ValueError("this model has a layer map (repeated layers); a layer map "
                         "cannot be sliced across nodes")
    old_logits = None
    lli = getattr(model, "logit_layer_idx", None)
    if lli is not None and 0 <= lli < len(model.modules):
        old_logits = model.modules[lli]

    sel = groups[start:end]
    if rebase:
        for new, (_, block, _) in enumerate(sel):
            old = block.layer_idx
            for sub in list(_walk(block)):
                if getattr(sub, "layer_idx", None) == old:
                    sub.layer_idx = new
    head = prolog if rank == 0 else []
    body = [x for before, block, after in sel for x in (*before, block, *after)]
    tail = epilog if rank == total - 1 else []
    keep = head + body + tail
    model.modules = keep

    # The model computed these for the FULL list; fix them for this slice.
    if hasattr(model, "last_kv_module_idx"):
        model.last_kv_module_idx = len(head) + len(body) - 1
    if hasattr(model, "first_block_idx"):
        model.first_block_idx = len(head)
    if hasattr(model, "logit_layer_idx"):
        model.logit_layer_idx = next(
            (i for i, x in enumerate(keep) if x is old_logits), None)
    if hasattr(model, "modules_dict"):
        model.modules_dict = None
    for name in ("_get_cache_layers", "_get_recurrent_layers", "_get_prefetch_layers"):
        getattr(model, "__dict__", {}).pop(name, None)  # stale cached_property values
    # Hyper-connection sites hand a weighted copy to their successor site
    # (GatedResidual.link_sites, wired over the FULL list at construction):
    # re-wire over the kept list so a boundary site hands off to nothing.
    linked = set()
    for m in keep:
        for sub in _walk(m):
            fn = getattr(type(sub), "link_sites", None)
            if callable(fn) and type(sub) not in linked:
                linked.add(type(sub))
                fn(model.modules)
    return len(sel)


def _select_and_rebase(model, start: int, end: int, rank: int, total: int) -> int:
    """The pre-2026-10-11 selection: also rebase layer_idx to 0..N-1."""
    return _select_slice(model, start, end, rank, total, rebase=True)


def make_cache(cache_cls, model, max_num_tokens: int, max_batch_size: int = 1,
               max_history: int = 0):
    """Cache(model, max_num_tokens[, max_batch_size[, max_history]]) --
    max_batch_size (the recurrent-state slot count) and max_history (per-token
    recurrent history, what speculative decoding on a GatedDeltaNet model needs
    to roll a state back) only where the installed exllamav3 takes them."""
    try:
        params = inspect.signature(cache_cls).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {"max_batch_size": max_batch_size} if "max_batch_size" in params else {}
    if max_history and "max_history" in params:
        kw["max_history"] = int(max_history)
    return cache_cls(model, max_num_tokens=max_num_tokens, **kw)


# --------------------------------------------------------------------------
# sampling (runs on the LAST node)
# --------------------------------------------------------------------------

def _num(v, name, kind=float):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    if kind is int and int(v) != v:
        raise ValueError(f"{name} must be an integer, got {v!r}")
    return kind(v)


@dataclass
class SamplingSettings:
    """Per-request sampling, carried to the last node in the CONTROL 'begin'
    frame. temperature 0 (the default) or top_k 1 = greedy."""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    min_p: float = 0.0

    def __post_init__(self):
        self.temperature = _num(self.temperature, "temperature")
        self.top_p = _num(self.top_p, "top_p")
        self.top_k = _num(self.top_k, "top_k", int)
        self.min_p = _num(self.min_p, "min_p")
        if not (self.temperature >= 0 and math.isfinite(self.temperature)):
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0 <= self.min_p <= 1):
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if self.seed is not None:
            self.seed = _num(self.seed, "seed", int)
            if not (0 <= self.seed < 2 ** 63):
                raise ValueError(f"seed must be in [0, 2^63), got {self.seed}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0 or self.top_k == 1

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "SamplingSettings":
        d = d or {}
        kw = {k: d[k] for k in ("temperature", "top_p", "top_k", "seed", "min_p")
              if k in d and d[k] is not None}
        return cls(**kw)

    def to_dict(self) -> Dict:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "top_k": self.top_k, "seed": self.seed, "min_p": self.min_p}


class Sampler:
    """temperature -> top_k -> softmax -> min_p -> top_p -> multinomial with a
    per-request torch.Generator (seeded: the same seed reproduces the same
    tokens on the same device). Greedy = argmax (first max on a tie)."""

    def __init__(self, settings: SamplingSettings):
        self.s = settings
        self._gen = None
        self._gen_device = None

    def _generator(self, device):
        torch = _torch()
        if self._gen is None or self._gen_device != str(device):
            g = torch.Generator(device=device)
            g.manual_seed(self.s.seed if self.s.seed is not None else secrets.randbits(63))
            self._gen, self._gen_device = g, str(device)
        return self._gen

    def sample(self, row) -> int:
        torch = _torch()
        x = row.reshape(-1).float()
        if self.s.greedy:
            return int(torch.argmax(x))
        x = x / self.s.temperature
        if 0 < self.s.top_k < x.numel():
            kth = torch.topk(x, self.s.top_k).values[-1]
            x = x.masked_fill(x < kth, float("-inf"))
        probs = torch.softmax(x, dim=-1)
        if self.s.min_p > 0:
            probs = probs.masked_fill(probs < self.s.min_p * probs.max(), 0.0)
        if self.s.top_p < 1:
            sp, si = torch.sort(probs, descending=True)
            remove = (torch.cumsum(sp, dim=-1) - sp) > self.s.top_p
            sp = sp.masked_fill(remove, 0.0)
            probs = torch.zeros_like(probs).scatter(0, si, sp)
        probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=self._generator(probs.device)))


# --------------------------------------------------------------------------
# helpers shared with server.py / parity_check.py
# --------------------------------------------------------------------------

def encode_prompt(tokenizer, text: str, **kw) -> List[int]:
    """tokenizer.encode -> a flat id list (exllamav3 returns a (1, L) Tensor)."""
    ids = tokenizer.encode(text, **kw)
    if hasattr(ids, "reshape"):
        return [int(t) for t in ids.reshape(-1).tolist()]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(t) for t in ids]


def eos_token_ids(config, tokenizer=None) -> set:
    """Every stop id: config.eos_token_id_list (exllamav3 merges
    tokenizer_config + generation_config into it, tokenizer.py:150-166),
    config.eos_token_id and tokenizer.eos_token_id."""
    out = set()
    for v in (getattr(config, "eos_token_id_list", None) or []):
        out.add(int(v))
    for v in (getattr(config, "eos_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if isinstance(v, (list, tuple)):
            out.update(int(x) for x in v)
        elif v is not None:
            out.add(int(v))
    return out


def write_dump(path: str, prompt_ids: Sequence[int], tokens: Sequence[int],
               logits: Optional[np.ndarray], meta: Optional[Dict] = None) -> None:
    """A run's prompt ids, generated tokens and per-step logits (.npz) --
    what parity_check compares."""
    lg = np.zeros((0, 0), np.float32) if logits is None else np.asarray(logits, np.float32)
    with open(path, "wb") as f:  # a file object: np.savez must not append .npz
        np.savez(f, prompt_ids=np.asarray(list(prompt_ids), np.int64),
                 tokens=np.asarray(list(tokens), np.int64), logits=lg,
                 meta=np.array(json.dumps(meta or {})))


def read_dump(path: str) -> Dict:
    with np.load(path, allow_pickle=False) as z:
        return {"prompt_ids": z["prompt_ids"].tolist(), "tokens": z["tokens"].tolist(),
                "logits": z["logits"], "meta": json.loads(str(z["meta"]))}


def _to_np(t) -> np.ndarray:
    # Activations may carry requires_grad from the kernels: detach first.
    return t.detach().cpu().numpy()


def _load_advance_fn() -> Optional[Callable]:
    try:
        from exllamav3.cache.recurrent_util import advance_recurrent_states
    except ImportError:  # an exllamav3 without recurrent support
        return None
    return advance_recurrent_states


class _Tx:
    """Every send on ONE link. Direct (the single-sequence path): the caller
    blocks in the socket write. Async (batched mode): a FIFO queue drained by
    one thread, so a node starts computing its next step while this one is
    still on the wire; the order of frames is the order of calls. A send
    failure is remembered and raised by the next call."""

    def __init__(self, link: Optional[BigBrainLink], stats: Optional[NodeStats] = None,
                 async_send: bool = False):
        self.link = link
        self.stats = stats
        self.async_send = bool(async_send) and link is not None
        self.error: Optional[BaseException] = None
        self._q: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._pending = 0
        self._cv = threading.Condition()
        if self.async_send:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="bigbrain-tx")
            self._thread.start()

    def _run(self, name: str, args: tuple, kw: dict) -> None:
        t0 = time.monotonic()
        try:
            getattr(self.link, name)(*args, **kw)
        finally:
            if self.stats is not None:
                self.stats.send_s += time.monotonic() - t0

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            name, args, kw = item
            try:
                if self.error is None:
                    if name == "__pair__":
                        for n2, a2, k2 in args:
                            self._run(n2, a2, k2)
                    else:
                        self._run(name, args, kw)
            except BaseException as exc:  # noqa: BLE001 - surfaced by the next call
                self.error = exc
            finally:
                with self._cv:
                    self._pending -= 1
                    self._cv.notify_all()

    def call(self, name: str, *args, **kw) -> None:
        if self.link is None:
            raise BigBrainProtocolError("no link to send on")
        if self.error is not None:
            err, self.error = self.error, None
            raise err
        if self.async_send:
            with self._cv:
                self._pending += 1
            self._q.put((name, args, kw))
        else:
            self._run(name, args, kw)

    def call_pair(self, first: tuple, second: tuple) -> None:
        """Two sends that must go out together (a 'step' control frame and its activation):
        ONE queue item, so the sender can never write one without the other."""
        if self.link is None:
            raise BigBrainProtocolError("no link to send on")
        if self.error is not None:
            err, self.error = self.error, None
            raise err
        if self.async_send:
            with self._cv:
                self._pending += 1
            self._q.put(("__pair__", (first, second), {}))
        else:
            for name, args, kw in (first, second):
                self._run(name, args, kw)

    def send_control(self, obj: Dict, **kw) -> None:
        self.call("send_control", obj, **kw)

    def send_activation(self, arr, **kw) -> None:
        self.call("send_activation", arr, **kw)

    def send_tokens(self, ids, **kw) -> None:
        self.call("send_tokens", ids, **kw)

    def send_logits(self, row, **kw) -> None:
        self.call("send_logits", row, **kw)

    def reset(self, reason: str = "", **kw) -> None:
        self.call("reset", reason, **kw)

    def drain(self, timeout_s: float = 10.0) -> bool:
        """Wait until everything queued has been written."""
        if not self.async_send:
            return True
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while self._pending > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=2.0)
            self._thread = None


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

class SliceWorker:
    def __init__(self, model_dir: str, device: str, start: int, end: int,
                 rank: int, total: int, max_num_tokens: int = 8192,
                 in_port: Optional[int] = None,
                 next_host: Optional[str] = None, next_port: Optional[int] = None,
                 token_host: Optional[str] = None, token_port: Optional[int] = None,
                 watchdog_s: float = 5.0, mem_fraction: Optional[float] = None,
                 prefill_chunk: int = 2048, max_batch_size: int = 1,
                 rebase_layers: bool = False, bind_host: str = "0.0.0.0",
                 connect_settle_s: float = 1.5, slots: int = 1, max_draft: int = 0,
                 async_send: Optional[bool] = None, diag: bool = False, max_frame_mib: float = 0.0):
        torch = _torch()
        from exllamav3 import Config, Model, Cache, Tokenizer

        self.rank, self.total = rank, total
        self.is_first, self.is_last = rank == 0, rank == total - 1
        inbound = outbound = token_link = None
        # Every bind and connect happens BEFORE the (slow) model load: node 0
        # binds the token port, others bind their in-port; the OS backlog
        # queues the peers' connects until start_recv() accepts them post-load
        # (the startup-race fix; nodes boot in parallel).
        # bind_host 127.0.0.1 keeps a listener off the LAN when its peer comes
        # in through an ssh tunnel; connects verify a tunnel's far side
        # (pipe.connect settle_s), so through tunnels nodes may start in any order.
        if self.is_first and token_port:
            token_link = BigBrainLink(watchdog_s=watchdog_s)
            token_link.listen_inbound(token_port, bind_host=bind_host)
        if not self.is_first:
            inbound = BigBrainLink(watchdog_s=watchdog_s)
            self.inbound_port = inbound.listen_inbound(in_port or 0, bind_host=bind_host)
        if not self.is_last:
            outbound = self._connect_retry(next_host, next_port, watchdog_s,
                                           settle_s=connect_settle_s)
        if self.is_last and not self.is_first and token_host and token_port:
            token_link = self._connect_retry(token_host, token_port, watchdog_s,
                                             settle_s=connect_settle_s)

        cfg = Config.from_directory(model_dir)
        model = Model.from_config(cfg)
        _select_slice(model, start, end, rank, total, rebase=rebase_layers)
        model.prepare_layer_map()
        slots, max_draft = max(1, int(slots)), max(0, int(max_draft))
        batched = slots > 1 or max_draft > 0
        # Batched layout: slots x (context pages + 1 slack page); the legacy
        # single-sequence layout (one slot, no speculation) is unchanged.
        cache_tokens = PagedSlots.tokens_for(slots, max_num_tokens) if batched else max_num_tokens
        cache = make_cache(Cache, model, cache_tokens, max(max_batch_size, slots),
                           max_history=max_draft)
        if mem_fraction:
            # Cap THIS process's CUDA reservations (shared-GPU safety; the
            # loader's _load_single never touches the fraction).
            torch.cuda.set_per_process_memory_fraction(mem_fraction, torch.device(device))
        model.load(device=device)
        tokenizer = Tokenizer.from_config(cfg) if self.is_first else None
        self._init_runtime(model=model, cache=cache, tokenizer=tokenizer, config=cfg,
                           device=model.modules[0].device, max_num_tokens=max_num_tokens,
                           prefill_chunk=prefill_chunk, advance_fn=_load_advance_fn(),
                           inbound=inbound, outbound=outbound, token_link=token_link,
                           slots=slots, max_draft=max_draft, async_send=async_send, diag=diag,
                           max_frame_mib=max_frame_mib)

    @classmethod
    def from_parts(cls, *, model, cache, rank: int, total: int, tokenizer=None,
                   config=None, device="cpu", max_num_tokens: int = 8192,
                   prefill_chunk: int = 2048, advance_fn: Optional[Callable] = None,
                   inbound: Optional[BigBrainLink] = None,
                   outbound: Optional[BigBrainLink] = None,
                   token_link: Optional[BigBrainLink] = None, slots: int = 1,
                   max_draft: int = 0, async_send: Optional[bool] = None,
                   diag: bool = False, max_frame_mib: float = 0.0) -> "SliceWorker":
        """An already-loaded slice (tests, or an embedding process): the model
        must already be filtered/rebased with prepare_layer_map() run and the
        cache attached."""
        w = cls.__new__(cls)
        w.rank, w.total = rank, total
        w.is_first, w.is_last = rank == 0, rank == total - 1
        w._init_runtime(model=model, cache=cache, tokenizer=tokenizer,
                        config=config if config is not None else getattr(model, "config", None),
                        device=device, max_num_tokens=max_num_tokens,
                        prefill_chunk=prefill_chunk,
                        advance_fn=advance_fn if advance_fn is not None else _load_advance_fn(),
                        inbound=inbound, outbound=outbound, token_link=token_link,
                        slots=slots, max_draft=max_draft, async_send=async_send, diag=diag,
                        max_frame_mib=max_frame_mib)
        return w

    def _init_runtime(self, *, model, cache, tokenizer, config, device, max_num_tokens,
                      prefill_chunk, advance_fn, inbound, outbound, token_link,
                      slots=1, max_draft=0, async_send=None, diag=False, max_frame_mib=0.0):
        self.model, self.cache, self.tokenizer, self.config = model, cache, tokenizer, config
        self.device = device
        self.max_num_tokens = int(max_num_tokens)
        if prefill_chunk < 1:
            raise ValueError("prefill_chunk must be >= 1")
        self.prefill_chunk = int(prefill_chunk)
        self.vocab_size = getattr(config, "vocab_size", None)
        self.recurrent = bool(getattr(model, "caps", {}).get("recurrent_states"))
        self._advance = advance_fn
        if self.recurrent and advance_fn is None:
            raise RuntimeError("this model keeps recurrent state but the installed exllamav3 "
                               "has no cache.recurrent_util.advance_recurrent_states")
        self.inbound, self.outbound, self.token_link = inbound, outbound, token_link
        # Per-request state of THIS node (see the module docstring).
        self._states: Optional[list] = None
        self.request_id: Optional[int] = None
        self.slot = 0  # node 0 assigns it; the others take it from 'begin'
        self._pending_ids: Optional[Tuple[int, int, List[int]]] = None
        self.settings = SamplingSettings()
        self.sampler: Optional[Sampler] = None
        self.return_logits = False
        self.failed = False
        self.position = 0
        self.last_logits: Optional[np.ndarray] = None
        self._next_rid = 0
        self._inq: "queue.Queue[Message]" = queue.Queue()
        self._recv_thread: Optional[threading.Thread] = None
        # Batched mode (several sequences in flight, speculation): see paged.py
        # and engine.py. A node built with one slot and no speculation keeps the
        # single-sequence layout and traffic exactly.
        self.n_slots = max(1, int(slots))
        self.max_draft = max(0, int(max_draft))
        self.batched = self.n_slots > 1 or self.max_draft > 0
        self.diag = bool(diag)
        # The largest frame this node accepts on the links it READS (security review M3): from the model (hidden size, slots,
        # draft, prefill chunk), or --max-frame-mib.
        self.max_frame_bytes = self._frame_limit(config, max_frame_mib)
        for link in (inbound, token_link):
            if link is not None:
                link.max_frame_bytes = self.max_frame_bytes
        self.stats = NodeStats(self.rank)
        if async_send is None:
            async_send = self.batched
        self._tx_out = _Tx(outbound, self.stats, async_send)
        self._tx_tok = _Tx(token_link, self.stats, async_send)
        self.paged = PagedSlots(self.n_slots, self.max_num_tokens, recurrent=self.recurrent,
                                new_state=self._new_state, release_state=self._release_state_obj)
        self._samplers: Dict[int, Sampler] = {}
        self._diag_slots: set = set()
        self._pending_step: Optional[StepMeta] = None
        self._skip_step: Optional[int] = None
        self.tx_lock = threading.Lock()   # a step's control+activation frames are sent as a pair

    def _frame_limit(self, config, max_frame_mib: float = 0.0) -> int:
        """--max-frame-mib when given; else pipe.frame_limit() for the model: one activation of the widest step this pipeline
        sends (a prefill chunk, or every slot's draft) in fp32, or one logits row, with slack. A config without hidden_size keeps
        the pipe's flat default."""
        if max_frame_mib and float(max_frame_mib) > 0:
            return int(float(max_frame_mib) * 1024 * 1024)
        hidden = getattr(config, "hidden_size", None)
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden <= 0:
            return DEFAULT_MAX_FRAME_BYTES
        tokens = max(self.prefill_chunk, self.n_slots * (self.max_draft + 1))
        streams = getattr(config, "hc_mult", None)
        streams = streams if isinstance(streams, int) and not isinstance(streams, bool) and streams > 0 else 1
        vocab = self.vocab_size if isinstance(self.vocab_size, int) and not isinstance(self.vocab_size, bool) else 0
        return frame_limit(hidden, tokens, streams=streams, vocab_size=vocab)

    @staticmethod
    def _connect_retry(host: str, port: int, watchdog_s: float, attempts: int = 600,
                       retry_sleep_s: float = 1.0, settle_s: float = 1.5) -> BigBrainLink:
        """Connect with retries: the peer may still be loading its model when
        we start (nodes boot in parallel; a contended GPU can make the load
        take minutes), or an ssh tunnel's far side may not be up yet (its
        accept-then-close is retried too). 600 x 1s = 10 min budget."""
        last = None
        for _ in range(attempts):
            link = BigBrainLink(watchdog_s=watchdog_s)
            try:
                return link.connect(host, port, settle_s=settle_s)
            except Exception as exc:  # noqa: BLE001 - retry any connect failure
                last = exc
                link.close()
                time.sleep(retry_sleep_s)
        raise BigBrainPeerDead(f"could not connect to {host}:{port}: {last}")

    # ------------------------------------------------------- recurrent states

    def discard_states(self) -> None:
        """Release this node's states of the current request (slot back on the
        cache's free_list). Idempotent."""
        states, self._states = self._states, None
        if not states:
            return
        with _torch().inference_mode():  # a release may touch the pools
            for s in states:
                try:
                    if s.slot not in self.cache.free_list:
                        self.cache.release_state(s)
                except Exception:  # noqa: BLE001 - never let cleanup mask the real error
                    log.exception("rank %d: releasing a recurrent state failed", self.rank)

    def _new_state(self, slot: int):
        """Cache.get_new_state() (cache.py:326-337) with the slot GIVEN (node 0
        allocates for the whole chain) instead of popped: take the slot off the
        free list and build recurrent_state_cls(cache, slot, 0, clear=True),
        which zeroes that slot's conv ring and recurrent rows in every
        recurrent layer of this node's cache."""
        cache = self.cache
        n = getattr(cache, "num_slots", None)
        if n is not None and not 0 <= slot < n:
            raise BigBrainStateError(f"rank {self.rank}: slot {slot} outside this node's "
                                     f"{n} recurrent-state slots (--max-batch-size)")
        try:
            cache.free_list.remove(slot)
        except ValueError:
            # Only this worker uses the cache and only one request is live, so
            # a slot that is not free leaked from a crashed request: reclaim it
            # (clear=True wipes it).
            log.warning("rank %d: recurrent slot %d was not free; reclaiming it", self.rank, slot)
        with _torch().inference_mode():  # clear=True zeroes inference-tensor pools in place
            return cache.recurrent_state_cls(cache, slot, 0, clear=True)

    def _states_for(self, past_len: int) -> Optional[list]:
        if not self.recurrent:
            return None
        if past_len == 0:
            self.discard_states()
            self._states = [self._new_state(self.slot)]
        elif self._states is None:
            raise BigBrainStateError(
                f"rank {self.rank}: step at past_len={past_len} but this node holds no "
                f"recurrent state for request {self.request_id} (its prefill never "
                f"reached this node, or the state was discarded)")
        elif self._states[0].position != past_len:
            raise BigBrainStateError(
                f"rank {self.rank}: recurrent state is at position {self._states[0].position} "
                f"but the step is at past_len={past_len}")
        return self._states

    def _params(self, past_len: int) -> dict:
        p = {"attn_mode": "flash_attn", "cache": self.cache,
             "batch_shape": (1, self.max_num_tokens), "past_len": past_len}
        states = self._states_for(past_len)
        if states is not None:
            p["recurrent_states"] = states
        return p

    # ---------------------------------------------------------- module loops

    def _begin_pass(self):
        for h in getattr(getattr(self.model, "config", None), "moe_cpu_hosts", {}).values():
            h.begin_pass()

    def _run_body(self, x, params: dict):
        """Embedding? + this node's blocks, stopping after the last KV block --
        returns the residual. Never runs the epilog (norm+head)."""
        self._begin_pass()
        for module, inst, idx in self.model.fwd_modules:
            params["layer_instance"] = inst
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if (idx, inst) == self.model.last_kv_module_idx_instance:
                break
        return x

    def _run_prefill_tail(self, x, params: dict) -> None:
        """LAST node, prompt chunk: exactly prefill_ls (model_ls.py:329-345) --
        params['prefill'] on the last KV module (its block stops after the
        attention, transformer.py:167), stop there; nothing is returned."""
        self._begin_pass()
        for module, inst, idx in self.model.fwd_modules:
            params["layer_instance"] = inst
            pf = (idx, inst) == self.model.last_kv_module_idx_instance
            params["prefill"] = pf
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if pf:
                break
        params.pop("prefill", None)

    def _run_full(self, x, params: dict):
        """LAST node, decode: the ENTIRE fwd_modules (blocks -> norm -> head)
        exactly like forward_ls (model_ls.py:348-361) -- returns the logits."""
        self._begin_pass()
        for module, inst, idx in self.model.fwd_modules:
            params["layer_instance"] = inst
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x

    def process(self, x, past_len: int, prefill: bool, ids: Optional[Sequence[int]] = None):
        """Run one chunk through this node's slice at positions
        past_len..past_len+seq-1. x: token ids (node 0) or the residual; ids:
        the chunk's input token ids on a downstream node (None = unknown, the
        arch hooks then see zeros). Returns the residual (non-last), the
        logits (last, decode) or None (last, prefill). Under inference_mode:
        GatedDeltaNet state ops are in-place on inference tensors."""
        t_start = time.monotonic()
        try:
            return self._process(x, past_len, prefill, ids)
        finally:
            # the single-sequence path's share of the per-node timers (the batched path keeps its own)
            self.stats.busy_s += time.monotonic() - t_start
            self.stats.steps += 1
            if prefill:
                self.stats.prefill_steps += 1

    def _process(self, x, past_len: int, prefill: bool, ids: Optional[Sequence[int]] = None):
        torch = _torch()
        seqlen = int(x.shape[1])
        if past_len < 0 or past_len + seqlen > self.max_num_tokens:
            raise ValueError(f"positions {past_len}..{past_len + seqlen - 1} exceed "
                             f"max_num_tokens={self.max_num_tokens}")
        with torch.inference_mode():
            params = self._params(past_len)
            try:
                if self.is_first:
                    x = self.model.prepare_inputs(x, params)
                    for m in getattr(self.model, "_get_prefetch_layers", None) or ():
                        m.prefetch(x, params)
                else:
                    # The arch's own hook (prepare_for_attn [+ prepare_for_recurrence])
                    # with the chunk's real ids (CPU, as the generator passes them);
                    # params["input_ids"] is what the Embedding sets on node 0.
                    if ids is not None and len(ids) == seqlen:
                        tids = torch.tensor([list(ids)], dtype=torch.long)
                        tids = self.model.prepare_inputs(tids, params)
                        params["input_ids"] = tids
                        for m in getattr(self.model, "_get_prefetch_layers", None) or ():
                            m.prefetch(tids, params)
                    else:
                        self.model.prepare_inputs(torch.zeros((1, seqlen), dtype=torch.long), params)
                if not self.is_last:
                    out = self._run_body(x, params)
                elif prefill:
                    self._run_prefill_tail(x, params)
                    out = None
                else:
                    out = self._run_full(x, params)
                if self.recurrent:
                    self._advance(torch.zeros((1, seqlen), dtype=torch.long), params, self.model)
            except BaseException:
                self.discard_states()  # a half-advanced state is unusable
                raise
        self.position = past_len + seqlen
        return out

    def _logits_row(self, out):
        row = out[0, -1]
        if self.vocab_size:
            row = row[: self.vocab_size]  # lm_head may be padded past the vocab
        return row.float()

    # ------------------------------------------------------- request lifecycle

    def begin_request(self, request_id: int, settings: SamplingSettings,
                      return_logits: bool = False, slot: Optional[int] = None) -> None:
        self.discard_states()
        self._pending_ids = None
        if slot is not None:
            self.slot = int(slot)
        self.request_id = int(request_id)
        self.settings = settings
        self.sampler = Sampler(settings) if self.is_last else None
        self.return_logits = bool(return_logits)
        self.failed = False
        self.position = 0

    def end_request(self, request_id: int) -> None:
        if request_id == self.request_id:
            self.discard_states()
            self.request_id = None

    def _new_request_id(self) -> int:
        self._next_rid += 1
        return self._next_rid

    def generate(self, prompt_ids: Sequence[int], settings: SamplingSettings, max_new: int,
                 request_id: Optional[int] = None, return_logits: bool = False,
                 token_timeout_s: float = 600.0) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
        """NODE 0: one request. Yields (token_id, logits_row or None) per step;
        the caller decides when to stop (EOS / stop strings / cancel) and must
        close() the generator (use contextlib.closing) so the chain releases
        the request's states. return_logits fetches the last node's cropped
        fp32 logits row with every token (parity/debug; ~1 MB per token)."""
        torch = _torch()
        if not self.is_first:
            raise RuntimeError("generate() runs on node 0")
        prompt = [int(t) for t in prompt_ids]
        S = len(prompt)
        if S == 0:
            raise ValueError("empty prompt")
        if max_new < 1:
            return
        if S + max_new > self.max_num_tokens:
            raise ValueError(f"prompt ({S}) + max_new ({max_new}) exceeds "
                             f"max_num_tokens={self.max_num_tokens}")
        rid = int(request_id) if request_id is not None else self._new_request_id()
        self.begin_request(rid, settings, return_logits)
        try:
            if self.outbound is not None:
                self._tx_out.send_control(
                    {"op": "begin", "request_id": rid, "slot": self.slot,
                     "sampling": settings.to_dict(),
                     "return_logits": bool(return_logits), "prompt_tokens": S},
                    request_id=rid)
            ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            # PREFILL prompt[:-1] (exllamav3's generator stops prefill one short:
            # job.py:1381); the last prompt token is the first decode step's input.
            for start in range(0, S - 1, self.prefill_chunk):
                end = min(start + self.prefill_chunk, S - 1)
                out = self.process(ids[:, start:end], past_len=start, prefill=True)
                if not self.is_last:
                    self._tx_out.send_tokens(prompt[start:end], request_id=rid, position=start)
                    self._tx_out.send_activation(_to_np(out), request_id=rid,
                                                  position=start, flags=FLAG_PREFILL)
            pos, tok = S - 1, prompt[-1]
            for _ in range(max_new):
                out = self.process(torch.tensor([[tok]], dtype=torch.long, device=self.device),
                                   past_len=pos, prefill=False)
                if self.is_last:
                    with torch.inference_mode():
                        row = self._logits_row(out)
                        nxt = self.sampler.sample(row)
                        lg = row.cpu().numpy() if return_logits else None
                else:
                    self._tx_out.send_tokens([tok], request_id=rid, position=pos)
                    self._tx_out.send_activation(_to_np(out), request_id=rid, position=pos)
                    nxt, lg = self._await_token(rid, return_logits, token_timeout_s,
                                                expect_position=pos + 1)
                pos += 1
                yield nxt, lg
                tok = nxt
        finally:
            self.discard_states()
            self.request_id = None
            if self.outbound is not None and not self.outbound.is_dead:
                try:
                    self._tx_out.send_control({"op": "end", "request_id": rid}, request_id=rid)
                except BigBrainPeerDead:
                    pass

    def _await_token(self, request_id: int, want_logits: bool, timeout_s: float,
                     expect_position: Optional[int] = None) -> Tuple[int, Optional[np.ndarray]]:
        """NODE 0: read the token link until THIS request's token arrives
        (frames of an aborted earlier request are skipped)."""
        if self.token_link is None:
            raise BigBrainProtocolError("node 0 has no token link")
        deadline = time.monotonic() + timeout_s
        row = None
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no token for request {request_id} within {timeout_s:.0f}s")
            t_wait = time.monotonic()
            msg = self.token_link.recv_message(include_heartbeats=True)
            self.stats.wait_s += time.monotonic() - t_wait      # node 0's hop wait: the rest of the pipeline
            if msg.kind == "heartbeat":
                continue
            if msg.kind == "bye":
                raise BigBrainPeerDead("the last node closed the token link")
            if msg.kind == "control" and msg.control.get("op") == "error" \
                    and int(msg.control.get("request_id", -1)) == request_id:
                raise BigBrainRemoteError(str(msg.control.get("message", "remote error")))
            if msg.request_id != request_id:
                log.debug("dropping a stale %s of request %d", msg.kind, msg.request_id)
                continue
            if msg.kind == "logits":
                row = msg.array.reshape(-1)
                continue
            if msg.kind == "token":
                if expect_position is not None and msg.position != expect_position:
                    raise BigBrainProtocolError(
                        f"token for position {msg.position}, expected {expect_position}")
                if want_logits and row is None:
                    raise BigBrainProtocolError("token arrived without the requested logits")
                return int(msg.tokens[0]), row
            log.warning("ignoring a %s frame on the token link", msg.kind)

    # ----------------------------------------------- batched mode (paged.py)
    #
    # Several sequences in flight at once, and speculative verification. Node 0
    # runs engine.PipelineEngine, which sends STEPS: a CONTROL 'step' frame
    # (StepMeta: per row the request, slot, start position, valid length n and
    # input ids) followed by the ACTIVATION it describes, request_id = step id.
    # A step is a rectangular (B, T) batch; a row shorter than T is padded and
    # the padding is just more rejected positions. Each node runs the step in
    # exllamav3's paged batch mode, forwards the residual and, on the last
    # node, sets the greedy tokens of every row (greedy_accept decides how many
    # of a row's drafts survive) and answers node 0 with a 'result' CONTROL.

    def _release_state_obj(self, state) -> None:
        with _torch().inference_mode():
            try:
                if state.slot not in self.cache.free_list:
                    self.cache.release_state(state)
            except Exception:  # noqa: BLE001 - cleanup must not mask the real error
                log.exception("rank %d: releasing a recurrent state failed", self.rank)

    @property
    def state_needs_history(self) -> bool:
        """True when this model's recurrent state rolls back only from per-token history the
        forward recorded (GatedDeltaNet: Cache(max_history=k), params recurrent_history) --
        false for a state that rewinds in place inside a stored window (gpt-oss's
        SWAState: guaranteed_rollback = one page) and for models with no recurrent state."""
        if not self.recurrent:
            return False
        cls = getattr(self.model, "recurrent_state_cls", None)
        return int(getattr(cls, "guaranteed_rollback", 0) or 0) < 16 + 1

    def capabilities(self) -> Dict:
        """What this node can do, for node 0's handshake."""
        return {"rank": self.rank, "proto": PROTO, "slots": self.n_slots,
                "ctx": self.max_num_tokens, "recurrent": self.recurrent,
                "needs_history": self.state_needs_history,
                "max_draft": self.max_draft, "batched": self.batched,
                "slot_pages": self.paged.slot_pages}

    def seq_begin(self, rid: int, slot: int, settings: SamplingSettings, diag: bool = False) -> None:
        if self._states is not None or self.request_id is not None:
            raise BigBrainStateError(
                f"rank {self.rank}: a single-sequence request is live; the single-sequence "
                f"and batched modes cannot be mixed on one pipeline")
        self.paged.begin(int(slot), int(rid))
        if self.is_last:
            self._samplers[int(slot)] = Sampler(settings)
            (self._diag_slots.add if (diag or self.diag) else self._diag_slots.discard)(int(slot))

    def seq_end(self, rid: int, slot: int) -> None:
        slot = int(slot)
        if 0 <= slot < self.n_slots and self.paged.owner(slot) == int(rid):
            self.paged.end(slot)
            self._samplers.pop(slot, None)
            self._diag_slots.discard(slot)

    def end_all_sequences(self) -> None:
        self.paged.end_all()
        self._samplers.clear()
        self._diag_slots.clear()
        self._pending_step = None

    def _ids_tensor(self, meta: StepMeta, device):
        torch = _torch()
        t = torch.zeros((len(meta.rows), meta.T), dtype=torch.long)
        for i, r in enumerate(meta.rows):
            t[i, :r.n] = torch.tensor(r.ids, dtype=torch.long)
        return t.to(device) if device is not None else t

    def process_batch(self, meta: StepMeta, x):
        """Run one step through this node's slice. x: the padded token ids
        (B, T) on node 0, else the residual (B, T, hidden[, streams]). Returns
        the residual (non-last), the logits (B, T, vocab) (last, decode) or
        None (last, prefill). Rolls each row's slot back to its start position
        first (paged.PagedSlots.prepare)."""
        torch = _torch()
        B, T = len(meta.rows), meta.T
        rows = [(r.slot, r.rid, r.pos, T) for r in meta.rows]
        prefill = meta.kind == "prefill"
        with torch.inference_mode():
            bt, seqlens, states = self.paged.prepare(rows)
            params = {"attn_mode": "flash_attn", "cache": self.cache,
                      "block_table": bt, "cache_seqlens": seqlens}
            if states is not None:
                params["recurrent_states"] = states
                # Per-token state history only for a verify window (2..max_draft+1 tokens), as
                # exllamav3's generator does (history only with draft tokens). A ONE-token forward
                # in history mode writes the tail of GatedDeltaNet's conv ring, not its head, and
                # leaves last_history = 0, so nothing could refresh the head: the next forward
                # would read a stale conv window. A prefill chunk is wider than the ring too.
                if self.max_draft and not prefill and 1 < T <= self.max_draft + 1:
                    params["recurrent_history"] = True
            try:
                if self.is_first:
                    x = self.model.prepare_inputs(x, params)
                    for m in getattr(self.model, "_get_prefetch_layers", None) or ():
                        m.prefetch(x, params)
                else:
                    tids = self.model.prepare_inputs(self._ids_tensor(meta, None), params)
                    params["input_ids"] = tids
                    for m in getattr(self.model, "_get_prefetch_layers", None) or ():
                        m.prefetch(tids, params)
                if not self.is_last:
                    out = self._run_body(x, params)
                elif prefill:
                    self._run_prefill_tail(x, params)
                    out = None
                else:
                    out = self._run_full(x, params)
                if self.recurrent:
                    self._advance(torch.zeros((B, T), dtype=torch.long), params, self.model)
            except BaseException:
                for slot, *_ in rows:
                    self.paged.invalidate(slot)
                raise
        self.paged.commit(rows)
        return out

    def decide(self, meta: StepMeta, logits) -> Dict:
        """LAST node, decode step: each row's committed tokens. A greedy row is
        verified (greedy_accept over the argmax of every row position); a
        sampled row (one token, no drafts) is sampled with its request's
        sampler."""
        torch = _torch()
        with torch.inference_mode():
            lg = logits[..., :self.vocab_size] if self.vocab_size else logits
            am = torch.argmax(lg, dim=-1).cpu().tolist()      # one device sync for the whole step
            out_rows = []
            for i, r in enumerate(meta.rows):
                smp = self._samplers.get(r.slot)
                if smp is None or smp.s.greedy:
                    toks = greedy_accept(r.ids, am[i], r.n)
                else:
                    if r.n != 1:
                        raise BigBrainStateError(
                            f"slot {r.slot}: a sampled sequence cannot carry drafts (n={r.n})")
                    toks = [smp.sample(lg[i, 0])]
                row = {"rid": r.rid, "toks": toks}
                if r.slot in self._diag_slots:
                    # top-1 minus top-2 logit at each committed position: how close a
                    # near-tie was (a parity divergence with a tiny margin is numerics)
                    top = torch.topk(lg[i, :len(toks)].float(), 2, dim=-1).values
                    row["margins"] = [round(float(v), 6) for v in (top[:, 0] - top[:, 1]).cpu()]
                out_rows.append(row)
        self.stats.tokens += sum(len(r["toks"]) for r in out_rows)
        return {"op": "result", "step": meta.step, "kind": meta.kind, "rows": out_rows}

    def _ack(self, meta: StepMeta) -> Dict:
        return {"op": "result", "step": meta.step, "kind": meta.kind, "rows": []}

    def _send_step_on(self, meta: StepMeta, out) -> None:
        t = pack_rows(out, meta.ns) if meta.packed else out
        arr = _to_np(t)
        with self.tx_lock:
            # one queue item: the sender thread can never write the 'step' frame without its activation
            self._tx_out.call_pair(
                ("send_control", (meta.to_dict(),), {"request_id": meta.step}),
                ("send_activation", (arr,), {"request_id": meta.step,
                                             "flags": FLAG_PREFILL if meta.kind == "prefill" else 0}))

    def head_step(self, meta: StepMeta) -> Optional[Dict]:
        """NODE 0: run this node's slice of a step and put the residual on the
        wire. With one node in the pipeline the step is decided here and the
        result dict is returned; otherwise None (the result arrives on the
        token link)."""
        t0 = time.monotonic()
        out = self.process_batch(meta, self._ids_tensor(meta, self.device))
        res = None
        if self.is_last:
            res = self.decide(meta, out) if out is not None else self._ack(meta)
        else:
            self._send_step_on(meta, out)
        self.stats.busy_s += time.monotonic() - t0
        self.stats.steps += 1
        self.stats.rows += len(meta.rows)
        if meta.kind == "prefill":
            self.stats.prefill_steps += 1
        return res

    def _on_step_activation(self, meta: StepMeta, arr: np.ndarray) -> None:
        torch = _torch()
        t0 = time.monotonic()
        try:
            x = torch.from_numpy(arr).to(self.device)
            if meta.packed:
                x = unpack_rows(x, meta.ns, meta.T)
            out = self.process_batch(meta, x)
            if self.is_last:
                res = self.decide(meta, out) if out is not None else self._ack(meta)
                self.stats.busy_s += time.monotonic() - t0
                t0 = None
                self._tx_tok.send_control(res, request_id=meta.step)
            else:
                self._send_step_on(meta, out)
        except BigBrainPeerDead:
            raise
        except Exception as exc:  # noqa: BLE001 - report it upstream, keep serving
            log.exception("rank %d: step %d failed", self.rank, meta.step)
            self.stats.errors += 1
            body = {"op": "error", "step": meta.step, "rids": [r.rid for r in meta.rows],
                    "rank": self.rank, "message": f"rank {self.rank}: {exc!r}"[:2000]}
            try:
                (self._tx_tok if self.is_last else self._tx_out).send_control(
                    body, request_id=meta.step)
            except BigBrainPeerDead:
                raise
        finally:
            if t0 is not None:
                self.stats.busy_s += time.monotonic() - t0
            self.stats.steps += 1
            self.stats.rows += len(meta.rows)
            if meta.kind == "prefill":
                self.stats.prefill_steps += 1

    def node_stats(self, reset: bool = False) -> Dict:
        """This node's timers, counters and memory since the last reset."""
        extra = {"slots": self.n_slots, "ctx": self.max_num_tokens, "recurrent": self.recurrent,
                 "rollbacks": self.paged.rollbacks, "rolled_tokens": self.paged.rolled_tokens,
                 "active_slots": len(self.paged.active())}
        try:
            torch = _torch()
            dev = torch.device(self.device)
            if dev.type == "cuda":
                free, total = torch.cuda.mem_get_info(dev)
                extra["vram"] = {"allocated_mib": round(torch.cuda.memory_allocated(dev) / 2 ** 20),
                                 "reserved_mib": round(torch.cuda.memory_reserved(dev) / 2 ** 20),
                                 "peak_allocated_mib": round(torch.cuda.max_memory_allocated(dev) / 2 ** 20),
                                 "free_mib": round(free / 2 ** 20), "total_mib": round(total / 2 ** 20)}
        except Exception:  # noqa: BLE001 - a missing reading must not fail the stats call
            pass
        snap = self.stats.snapshot(extra)
        if reset:
            self.stats.reset()
            self.paged.rollbacks = self.paged.rolled_tokens = 0
            try:
                torch = _torch()
                if torch.device(self.device).type == "cuda":
                    torch.cuda.reset_peak_memory_stats(torch.device(self.device))
            except Exception:  # noqa: BLE001
                pass
        return snap

    # ----------------------------------------------------------- serve loop

    def start_recv(self) -> None:
        """Accept the inbound peer and start the receive thread (middle/last);
        node 0 accepts its (already-bound) token peer. Called after load. A
        link this node only sends on is drained (pipe.start_drain)."""
        if self.is_first and self.token_link is not None and not self.token_link.connected:
            self.token_link.accept_peer(timeout_s=300.0)
        if self.inbound is not None:
            if not self.inbound.connected:
                self.inbound.accept_peer(timeout_s=300.0)
            if self._recv_thread is None:
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
        if self.outbound is not None:
            self.outbound.start_drain()
        if self.is_last and not self.is_first and self.token_link is not None:
            self.token_link.start_drain()

    def _recv_loop(self) -> None:
        while True:
            try:
                msg = self.inbound.recv_message()
            except (BigBrainPeerDead, BigBrainProtocolError) as exc:
                self._inq.put(Message("dead", reason=str(exc)))
                return
            except Exception as exc:  # noqa: BLE001 - this thread must never die silently (security review M3)
                log.exception("rank %d: the inbound reader crashed", self.rank)
                try:
                    self.inbound.close()
                except Exception:  # noqa: BLE001
                    pass
                self._inq.put(Message("dead", reason=f"the inbound reader crashed: {type(exc).__name__}: {exc}"))
                return
            self._inq.put(msg)
            if msg.kind == "bye":
                return

    def _peer_alive(self, idle_dead_s: float) -> bool:
        """True if the inbound peer showed any liveness (frame/heartbeat) within
        idle_dead_s. A live peer heartbeats every 0.5 s."""
        return (time.monotonic() - self.inbound.last_rx) < idle_dead_s

    def _next_message(self, idle_dead_s: float) -> Optional[Message]:
        while True:
            try:
                return self._inq.get(timeout=0.2)
            except queue.Empty:
                if not self._peer_alive(idle_dead_s):
                    return None

    def serve(self, idle_dead_s: float = 30.0) -> str:
        """NODES 1..T-1: process the inbound stream until node 0's terminator
        (zero-length activation), BYE, a dead link, or idle_dead_s without any
        liveness. Returns that reason. Serves any number of requests."""
        if self.is_first:
            raise RuntimeError("serve() runs on nodes 1..T-1")
        while True:
            t_wait = time.monotonic()
            msg = self._next_message(idle_dead_s)
            self.stats.wait_s += time.monotonic() - t_wait   # the hop wait
            if msg is None:
                return "idle"
            if msg.kind == "activation":
                if msg.array.shape[1] == 0:
                    self.send_terminator()  # relay down the chain
                    return "terminator"
                self._on_activation(msg)
            elif msg.kind == "control":
                self._on_control(msg)
            elif msg.kind == "token":
                # The INPUT ids of the activation that follows (same request,
                # same position).
                self._pending_ids = (msg.request_id, msg.position, msg.tokens)
            elif msg.kind == "reset":
                self.discard_states()
                self.end_all_sequences()
                self.request_id = None
                if self.outbound is not None:
                    self._tx_out.reset(msg.reason, request_id=msg.request_id)
            elif msg.kind in ("bye", "dead"):
                if msg.kind == "dead":
                    log.warning("rank %d: inbound link died: %s", self.rank, msg.reason)
                return msg.kind
            else:
                log.debug("rank %d: ignoring a %s message", self.rank, msg.kind)

    def _relay_control(self, msg: Message) -> None:
        if self.outbound is not None:
            self._tx_out.send_control(msg.control, request_id=msg.request_id)

    def _send_on(self, c: Dict, request_id: int = 0) -> None:
        """Pass a control frame to the next node, or (last node) back to node 0."""
        if self.is_last:
            if self._tx_tok.link is not None:
                self._tx_tok.send_control(c, request_id=request_id)
        else:
            self._tx_out.send_control(c, request_id=request_id)

    def _on_batch_control(self, msg: Message) -> bool:
        """The batched-mode ops. True when `msg` was one of them."""
        c = msg.control
        op = c.get("op")
        if op == "hello":
            c = dict(c)
            c["nodes"] = list(c.get("nodes") or []) + [self.capabilities()]
            if self.is_last:
                self._tx_tok.send_control({"op": "hello_ack", "proto": PROTO, "nodes": c["nodes"]},
                                          request_id=msg.request_id)
            else:
                self._tx_out.send_control(c, request_id=msg.request_id)
        elif op == "seq_begin":
            try:
                self.seq_begin(int(c["rid"]), int(c["slot"]), SamplingSettings.from_dict(c.get("sampling")),
                               diag=bool(c.get("diag")))
            except Exception as exc:  # noqa: BLE001 - tell node 0, keep serving
                log.exception("rank %d: seq_begin failed", self.rank)
                self._send_on({"op": "error", "rids": [int(c.get("rid", -1))], "rank": self.rank,
                               "message": f"rank {self.rank}: {exc!r}"[:2000]}, msg.request_id)
                return True
            self._relay_control(msg)
        elif op == "seq_end":
            self.seq_end(int(c.get("rid", -1)), int(c.get("slot", -1)))
            self._relay_control(msg)
        elif op == "step":
            if self._pending_step is not None:
                # the previous 'step' frame never got its activation: report it, do not mix them up
                lost, self._pending_step = self._pending_step, None
                log.error("rank %d: step %d arrived without its activation", self.rank, lost.step)
                self._send_on({"op": "error", "step": lost.step, "rids": [r.rid for r in lost.rows],
                               "rank": self.rank,
                               "message": f"rank {self.rank}: step {lost.step} had no activation"},
                              lost.step)
            try:
                self._pending_step = StepMeta.from_dict(c)
            except BigBrainStateError as exc:
                self._pending_step = None
                self._skip_step = int(c.get("step", -1))     # drop the activation that follows it
                log.error("rank %d: %s", self.rank, exc)
                self._send_on({"op": "error", "step": int(c.get("step", -1)), "rids": [],
                               "rank": self.rank, "message": f"rank {self.rank}: {exc}"}, msg.request_id)
        elif op == "stats":
            c = dict(c)
            c["nodes"] = list(c.get("nodes") or []) + [self.node_stats(reset=bool(c.get("reset")))]
            if self.is_last:
                self._tx_tok.send_control({"op": "stats_result", "nodes": c["nodes"]},
                                          request_id=msg.request_id)
            else:
                self._tx_out.send_control(c, request_id=msg.request_id)
        else:
            return False
        return True

    def _on_control(self, msg: Message) -> None:
        c = msg.control
        op = c.get("op")
        if self._on_batch_control(msg):
            return
        if op == "begin":
            self.begin_request(int(c["request_id"]), SamplingSettings.from_dict(c.get("sampling")),
                               bool(c.get("return_logits")), slot=c.get("slot"))
            self._relay_control(msg)
        elif op == "end":
            self.end_request(int(c.get("request_id", -1)))
            self._relay_control(msg)
        elif op == "error":
            if "step" not in c and int(c.get("request_id", -1)) == self.request_id:
                self.failed = True
                self.discard_states()
            if self.is_last:
                self._tx_tok.send_control(c, request_id=msg.request_id)
            else:
                self._relay_control(msg)
        else:
            self._relay_control(msg)  # forward-compatible: pass unknown ops on

    def _on_activation(self, msg: Message) -> None:
        torch = _torch()
        if self._pending_step is not None:
            meta, self._pending_step = self._pending_step, None
            if msg.request_id != meta.step:
                raise BigBrainProtocolError(
                    f"rank {self.rank}: activation for step {msg.request_id} after the "
                    f"'step' frame of step {meta.step}")
            return self._on_step_activation(meta, msg.array)
        if self._skip_step is not None and msg.request_id == self._skip_step:
            self._skip_step = None      # the activation of a step frame that could not be parsed
            return
        if self.paged.active():
            # batched sequences are live: an activation with no 'step' frame is a protocol error, and the
            # single-sequence path below would write into slot 0 (and clear its recurrent state)
            log.error("rank %d: dropping an activation (request %d) that has no 'step' frame while "
                      "%d batched sequence(s) are live", self.rank, msg.request_id, len(self.paged.active()))
            self.stats.errors += 1
            return
        rid, pos, prefill = msg.request_id, msg.position, msg.is_prefill
        if rid != self.request_id:
            if pos != 0:
                log.debug("rank %d: dropping a stale activation (request %d)", self.rank, rid)
                return
            self.begin_request(rid, SamplingSettings())  # a peer that sent no 'begin'
        ids = None
        if self._pending_ids is not None:
            p_rid, p_pos, p_ids = self._pending_ids
            self._pending_ids = None
            if p_rid == rid and p_pos == pos and len(p_ids) == msg.array.shape[1]:
                ids = p_ids
        if self.failed:
            return
        try:
            out = self.process(torch.from_numpy(msg.array).to(self.device), pos, prefill, ids=ids)
            if not self.is_last:
                if ids is not None:
                    self._tx_out.send_tokens(ids, request_id=rid, position=pos)
                self._tx_out.send_activation(_to_np(out), request_id=rid, position=pos,
                                              flags=FLAG_PREFILL if prefill else 0)
            elif not prefill:
                with torch.inference_mode():
                    row = self._logits_row(out)
                    nxt = self.sampler.sample(row)
                    row_np = row.cpu().numpy()
                self.last_logits = row_np
                if self.return_logits:
                    self._tx_tok.send_logits(row_np, request_id=rid, position=pos + 1)
                self._tx_tok.send_tokens([nxt], request_id=rid, position=pos + 1)
        except BigBrainPeerDead:
            raise
        except Exception as exc:  # noqa: BLE001 - report it upstream, keep serving
            log.exception("rank %d: request %d failed", self.rank, rid)
            self.failed = True
            self.discard_states()
            body = {"op": "error", "request_id": rid, "rank": self.rank,
                    "message": f"rank {self.rank}: {exc!r}"[:2000]}
            try:
                if self.is_last:
                    self._tx_tok.send_control(body, request_id=rid)
                else:
                    self._tx_out.send_control(body, request_id=rid)
            except BigBrainPeerDead:
                raise

    def send_terminator(self) -> None:
        """End of stream: a zero-length activation travels the activation
        chain (node 0 -> ... -> last); nodes exit on it."""
        if self.outbound is not None and not self.outbound.is_dead:
            self._tx_out.send_activation(np.zeros((1, 0, 1), dtype=np.float16))

    def close(self):
        self.discard_states()
        self.end_all_sequences()
        for tx in (self._tx_out, self._tx_tok):
            tx.drain(5.0)
            tx.close()
        for link in (self.inbound, self.outbound, self.token_link):
            if link is not None:
                link.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_node_args(a: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags that place one node (shared with server.py)."""
    a.add_argument("--model-dir", required=True)
    a.add_argument("--device", default="cuda:0")
    a.add_argument("--slice", nargs=2, type=int, required=True, metavar=("START", "END"))
    a.add_argument("--rank", type=int, required=True)
    a.add_argument("--total", type=int, required=True)
    a.add_argument("--in-port", type=int, default=None)
    a.add_argument("--next-host", default=None)
    a.add_argument("--next-port", type=int, default=None)
    a.add_argument("--token-host", default=None)
    a.add_argument("--token-port", type=int, default=None)
    a.add_argument("--max-num-tokens", type=int, default=8192,
                   help="context per request (cache rows); a multiple of 256")
    a.add_argument("--mem-fraction", type=float, default=None,
                   help="cap this process's CUDA memory fraction (shared GPUs)")
    a.add_argument("--prefill-chunk", type=int, default=2048,
                   help="prompt tokens per prefill hop (bounds activation memory "
                        "and lets the next node start early)")
    a.add_argument("--max-batch-size", type=int, default=1,
                   help="how many sequences this node holds at once -- the same as --slots (the "
                        "larger of the two wins); also sizes the recurrent-state slots of the cache")
    a.add_argument("--bind-host", default=os.environ.get("BIGBRAIN_BIND") or "0.0.0.0",
                   help="address the in-port / token-port listeners bind (127.0.0.1 when "
                        "the peer arrives through an ssh tunnel); default $BIGBRAIN_BIND "
                        "or 0.0.0.0")
    a.add_argument("--connect-settle", type=float, default=1.5,
                   help="seconds to watch a fresh connect for an ssh tunnel's "
                        "accept-then-close (0 = off)")
    a.add_argument("--rebase-layers", action="store_true",
                   help="renumber this slice's layers 0..N-1 (the pre-2026-10-11 "
                        "behaviour); default keeps the GLOBAL layer indices")
    a.add_argument("--watchdog-s", type=float, default=5.0,
                   help="seconds without ANY frame (heartbeats included, they come every 0.5 s) on one of this node's "
                        "links before it declares that peer dead. Keep 5 on a quiet LAN; raise it (30) when a hop goes "
                        "through an ssh tunnel or a node can stall in one long compute step "
                        "(a dense model with a slow prefill needs it)")
    a.add_argument("--max-frame-mib", type=float, default=0.0,
                   help="the largest frame (MiB) this node accepts from a peer; a larger one closes the link. 0 = derived "
                        "from the model (hidden size x the widest step x fp32, with slack; at least 64 MiB)")
    a.add_argument("--idle-dead", type=float, default=30.0,
                   help="nodes 1..T-1: declare the upstream peer gone after this "
                        "many seconds with no liveness (frames/heartbeats)")
    a.add_argument("--slots", type=int, default=1,
                   help="sequences this node can hold at once (each gets --max-num-tokens "
                        "of context + one slack page of cache). 1 with --max-draft 0 keeps "
                        "the single-sequence layout. EVERY node of a pipeline gets the same value.")
    a.add_argument("--max-draft", type=int, default=0,
                   help="speculative decoding: the most draft tokens a verify step carries "
                        "(sizes the recurrent-state history of GatedDeltaNet models). EVERY "
                        "node gets the same value; 0 = no speculation")
    a.add_argument("--spec-draft", "--spec", dest="spec_draft", choices=("none", "ngram", "model"),
                   default="none",
                   help="speculative decoding for greedy requests: none | ngram (prompt lookup, no "
                        "model) | model (a small draft model, --spec-model). Node 0 acts on it; every "
                        "node accepts it so one flag set serves the whole pipeline. Requests may "
                        "override it (spec / spec_k in the request body).")
    a.add_argument("--spec-model", "--draft-model-dir", dest="spec_model", default=None,
                   help="the draft model folder (same tokenizer as the target), node 0 only")
    a.add_argument("--spec-k", type=int, default=4,
                   help="most draft tokens per round (1..15); sizes --max-draft when that is not given")
    a.add_argument("--draft-device", default=None,
                   help="node 0: the CUDA device the draft model loads on (default: --device)")
    a.add_argument("--ngram-min", type=int, default=3, help="shortest context suffix an n-gram draft matches")
    a.add_argument("--ngram-max", type=int, default=6, help="longest context suffix an n-gram draft matches")
    a.add_argument("--no-async-send", dest="async_send", action="store_false", default=None,
                   help="batched mode: write frames from the compute thread instead of a "
                        "per-link sender thread (default: async when --slots > 1 or --max-draft > 0)")
    a.add_argument("--diag", action="store_true",
                   help="last node: always report top-1 minus top-2 logit margins (parity diagnosis)")
    return a


def build_arg_parser() -> argparse.ArgumentParser:
    a = add_node_args(argparse.ArgumentParser(description="One BigBrain pipeline node"))
    a.add_argument("--prompt", default="Hello")
    a.add_argument("--max-new", type=int, default=32)
    a.add_argument("--temperature", type=float, default=0.0,
                   help="node 0: 0 = greedy (the parity default)")
    a.add_argument("--top-p", type=float, default=1.0)
    a.add_argument("--top-k", type=int, default=0)
    a.add_argument("--min-p", type=float, default=0.0)
    a.add_argument("--seed", type=int, default=None)
    a.add_argument("--ignore-eos", action="store_true",
                   help="node 0: generate exactly --max-new tokens (parity runs)")
    a.add_argument("--tokens-out", default=None,
                   help="node 0: write the generated token-id list (json) here")
    a.add_argument("--logits-out", default=None,
                   help="last node: write the final step's logits (json) here")
    a.add_argument("--dump", default=None,
                   help="node 0: write prompt ids + tokens + every step's logits "
                        "(.npz, fetched from the last node) for parity_check")
    return a


def effective_slots(args) -> int:
    """--slots and --max-batch-size are the same knob (the staging harness passes the latter)."""
    return max(1, int(getattr(args, "slots", 1) or 1), int(getattr(args, "max_batch_size", 1) or 1))


def effective_max_draft(args) -> int:
    """--max-draft, else --spec-k when speculation is on (every node needs it for a model whose
    recurrent state rolls back from recorded history)."""
    md = int(getattr(args, "max_draft", 0) or 0)
    if md:
        return md
    return int(getattr(args, "spec_k", 4) or 0) if getattr(args, "spec_draft", "none") != "none" else 0


#: below this the 0.5 s heartbeat has too few chances to arrive before the watchdog fires
MIN_WATCHDOG_S = 2.0


def effective_watchdog(args) -> float:
    """--watchdog-s, refused below MIN_WATCHDOG_S (two lost heartbeats would then kill a healthy link)."""
    w = float(getattr(args, "watchdog_s", 5.0))
    if not w >= MIN_WATCHDOG_S:
        raise SystemExit(f"--watchdog-s {w:g} is below {MIN_WATCHDOG_S:g}: the heartbeat (every 0.5 s) needs room to arrive")
    return w


def worker_from_args(args) -> SliceWorker:
    return SliceWorker(args.model_dir, args.device, args.slice[0], args.slice[1],
                       args.rank, args.total, args.max_num_tokens,
                       in_port=args.in_port, next_host=args.next_host,
                       next_port=args.next_port, token_host=args.token_host,
                       token_port=args.token_port, mem_fraction=args.mem_fraction,
                       prefill_chunk=args.prefill_chunk,
                       max_batch_size=args.max_batch_size,
                       rebase_layers=args.rebase_layers, bind_host=args.bind_host,
                       connect_settle_s=args.connect_settle, slots=effective_slots(args),
                       max_draft=effective_max_draft(args), async_send=args.async_send,
                       diag=getattr(args, 'diag', False),
                       watchdog_s=effective_watchdog(args),
                       max_frame_mib=float(getattr(args, "max_frame_mib", 0.0) or 0.0))


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--http-port" or a.startswith("--http-port=") for a in raw):
        # SERVING mode: the controller launches node 0 as `-m bigbrain.worker ... --http-port N
        # --api-key-file api_key`. Node 0 is the OpenAI server (server.py); the worker's own parser has no
        # HTTP flags, so hand the whole command line over.
        from .server import main as server_main
        return server_main(raw)
    args = build_arg_parser().parse_args(raw)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s bigbrain.worker[%(process)d] %(levelname)s %(message)s")
    w = worker_from_args(args)
    try:
        w.start_recv()
        if w.is_first:
            torch = _torch()
            prompt_ids = encode_prompt(w.tokenizer, args.prompt)
            settings = SamplingSettings(temperature=args.temperature, top_p=args.top_p,
                                        top_k=args.top_k, min_p=args.min_p, seed=args.seed)
            eos = set() if args.ignore_eos else eos_token_ids(w.config, w.tokenizer)
            want_logits = bool(args.dump) or bool(args.logits_out and w.is_last)
            out, rows = [], []
            with closing(w.generate(prompt_ids, settings, args.max_new,
                                    return_logits=want_logits)) as g:
                for tok, row in g:
                    out.append(tok)
                    if row is not None:
                        rows.append(row)
                    if tok in eos:
                        break
            w.send_terminator()
            if args.tokens_out:
                with open(args.tokens_out, "w") as f:
                    json.dump(out, f)
            if args.dump:
                write_dump(args.dump, prompt_ids, out, np.stack(rows) if rows else None,
                           meta={"model_dir": args.model_dir, "total": args.total,
                                 "rank0_slice": list(args.slice), "settings": settings.to_dict(),
                                 "prefill_chunk": args.prefill_chunk,
                                 "rebase_layers": args.rebase_layers, "source": "pipeline"})
            if args.logits_out and w.is_last and rows:
                with open(args.logits_out, "w") as f:
                    json.dump([float(v) for v in rows[-1].tolist()], f)
            print(w.tokenizer.decode(torch.tensor(out)), flush=True)
        else:
            reason = w.serve(idle_dead_s=args.idle_dead)
            log.info("rank %d stopped: %s", w.rank, reason)
            if w.is_last and args.logits_out and w.last_logits is not None:
                with open(args.logits_out, "w") as f:
                    json.dump([float(v) for v in w.last_logits.tolist()], f)
    finally:
        w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
