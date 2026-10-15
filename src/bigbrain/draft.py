# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/draft.py -- the draft model of draft-model speculation, and the
tokenizer-identity check that makes it legal.

The draft model is a SMALL model (gpt-oss-20b for gpt-oss-120b) loaded WHOLE
into one process -- node 0's, on node 0's own card (co-resident with its slice
of the target) or on another card of node 0's machine (--draft-device). It is
driven through spec.DraftModelDrafter; this file is its compute backend,
spec.DraftForward over a real exllamav3 Model:

* one Cache sized like a node's (PagedSlots.tokens_for(slots, ctx)): the same
  paged layout, so the draft's slot s is the target's slot s;
* a step is the same paged batch the nodes run (block_table, cache_seqlens,
  recurrent_states), so rolling the draft back to the committed text is the same
  PagedSlots.prepare() rollback (attention cache: cache_seqlens; the sliding-
  window state: SWAState.rewind);
* Model.forward returns the logits of every fed position; the greedy token after
  each row's LAST real token is the proposal (padding positions after it are
  ignored).

The target and the draft MUST tokenize identically (a proposal is a token id the
target verifies as an id): check_same_tokenizer() compares the two folders'
tokenizer files byte for byte and the vocab sizes, and load_draft_model()
refuses a mismatch.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .paged import PagedSlots
from .spec import DraftForward

log = logging.getLogger("bigbrain.draft")

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def _torch():
    import torch
    return torch


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _vocab_size(model_dir: str) -> Optional[int]:
    try:
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            return int(json.load(f).get("vocab_size"))
    except (OSError, ValueError, TypeError):
        return None


def check_same_tokenizer(target_dir: str, draft_dir: str) -> Dict:
    """Do two model folders tokenize identically? Compares the sha256 of
    tokenizer.json / tokenizer_config.json / special_tokens_map.json and the
    config.json vocab_size. {"same": bool, "files": {...}, "vocab": [a, b],
    "problems": [...]}. A file missing on BOTH sides is skipped."""
    out = {"same": True, "files": {}, "vocab": [_vocab_size(target_dir), _vocab_size(draft_dir)],
           "problems": []}
    for name in TOKENIZER_FILES:
        a, b = _sha256(os.path.join(target_dir, name)), _sha256(os.path.join(draft_dir, name))
        out["files"][name] = {"target": a, "draft": b, "same": a == b}
        if a is None and b is None:
            continue
        if a != b:
            out["same"] = False
            out["problems"].append(f"{name} differs ({'missing' if a is None or b is None else 'content'})")
    if out["vocab"][0] is not None and out["vocab"][0] != out["vocab"][1]:
        out["same"] = False
        out["problems"].append(f"vocab_size {out['vocab'][0]} vs {out['vocab'][1]}")
    if out["files"].get("tokenizer.json", {}).get("target") is None:
        out["same"] = False
        out["problems"].append("tokenizer.json not found in the target folder")
    return out


class ExllamaDraftForward(DraftForward):
    """spec.DraftForward over a whole exllamav3 Model with a paged Cache."""

    def __init__(self, model, cache, n_slots: int, ctx: int, vocab_size: int,
                 device="cuda:0", new_state=None, release_state=None):
        self.model, self.cache, self.device = model, cache, device
        self.vocab_size = int(vocab_size)
        self.recurrent = bool(getattr(model, "caps", {}).get("recurrent_states"))
        self.history = bool(getattr(cache, "max_history", 0))
        self.paged = PagedSlots(n_slots, ctx, recurrent=self.recurrent,
                                new_state=new_state or self._new_state,
                                release_state=release_state or self._release_state)

    # recurrent state handles, the same way worker.SliceWorker makes them
    def _new_state(self, slot: int):
        cache = self.cache
        try:
            cache.free_list.remove(slot)
        except ValueError:
            log.warning("draft recurrent slot %d was not free; reclaiming it", slot)
        with _torch().inference_mode():
            return cache.recurrent_state_cls(cache, slot, 0, clear=True)

    def _release_state(self, state) -> None:
        with _torch().inference_mode():
            if state.slot not in self.cache.free_list:
                self.cache.release_state(state)

    def _ensure(self, slot: int) -> None:
        if self.paged.owner(slot) is None:
            self.paged.begin(slot, slot + 1)

    def _params(self, prows, history: bool = False) -> dict:
        bt, seqlens, states = self.paged.prepare(prows)
        params = {"attn_mode": "flash_attn", "cache": self.cache,
                  "block_table": bt, "cache_seqlens": seqlens}
        if states is not None:
            params["recurrent_states"] = states
            # history only for a multi-token window that fits the cache's history (never for a
            # prefill chunk or a one-token step: see worker.process_batch)
            if history and self.history and 1 < prows[0][3] <= self.cache.max_history + 1:
                params["recurrent_history"] = True
        return params

    def _ids(self, rows: Sequence[Tuple[int, int, Sequence[int]]]):
        torch = _torch()
        W = max(len(t) for _, _, t in rows)
        ids = torch.zeros((len(rows), W), dtype=torch.long)
        for i, (_, _, toks) in enumerate(rows):
            ids[i, :len(toks)] = torch.tensor([int(t) for t in toks], dtype=torch.long)
        return ids, W

    def forward(self, rows):
        torch = _torch()
        ids, W = self._ids(rows)
        for slot, _, _ in rows:
            self._ensure(slot)
        prows = [(slot, slot + 1, pos, W) for slot, pos, _ in rows]
        with torch.inference_mode():
            params = self._params(prows, history=True)
            try:
                logits = self.model.forward(ids.to(self.device), params)
            except BaseException:
                for slot, *_ in prows:
                    self.paged.invalidate(slot)
                raise
            self.paged.commit(prows)
            lg = logits[..., :self.vocab_size]
            last = torch.tensor([len(t) - 1 for _, _, t in rows], device=lg.device)
            sel = lg[torch.arange(len(rows), device=lg.device), last]
            return [int(t) for t in torch.argmax(sel, dim=-1).cpu().tolist()]

    def prefill(self, slot, start_pos, tokens):
        torch = _torch()
        rows = [(slot, start_pos, tokens)]
        ids, W = self._ids(rows)
        self._ensure(slot)
        prows = [(slot, slot + 1, start_pos, W)]
        with torch.inference_mode():
            params = self._params(prows)
            try:
                if hasattr(self.model, "prefill"):
                    self.model.prefill(ids.to(self.device), params)
                else:
                    self.model.forward(ids.to(self.device), params)
            except BaseException:
                self.paged.invalidate(slot)
                raise
            self.paged.commit(prows)

    def close(self, slot: int) -> None:
        if 0 <= slot < self.paged.n_slots:
            self.paged.end(slot)

    def close_all(self) -> None:
        self.paged.end_all()


def load_draft_model(draft_dir: str, target_dir: str, device: str, n_slots: int, ctx: int,
                     mem_fraction: Optional[float] = None, max_draft: int = 0,
                     check_tokenizer: bool = True) -> ExllamaDraftForward:
    """Load the draft model whole on `device` with a paged cache for n_slots
    sequences of ctx tokens. Refuses a draft whose tokenizer differs from the
    target's (check_same_tokenizer). Imports exllamav3 lazily."""
    if check_tokenizer:
        same = check_same_tokenizer(target_dir, draft_dir)
        if not same["same"]:
            raise ValueError("the draft model does not share the target's tokenizer: "
                             + "; ".join(same["problems"]))
    torch = _torch()
    from exllamav3 import Cache, Config, Model
    from .worker import make_cache
    cfg = Config.from_directory(draft_dir)
    model = Model.from_config(cfg)
    model.prepare_layer_map()
    cache = make_cache(Cache, model, PagedSlots.tokens_for(n_slots, ctx), max_batch_size=n_slots,
                       max_history=max_draft)
    if mem_fraction:
        torch.cuda.set_per_process_memory_fraction(mem_fraction, torch.device(device))
    model.load(device=device)
    log.info("draft model %s loaded on %s (%d slot(s) x %d tokens)", os.path.basename(draft_dir),
             device, n_slots, ctx)
    return ExllamaDraftForward(model, cache, n_slots, ctx, int(cfg.vocab_size), device=device)
