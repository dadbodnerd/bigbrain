# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""tools/bigbrain/spec.py -- speculative decoding for the BigBrain pipeline.

Pure Python (no torch): the logic that decides WHAT to propose, how much of a
proposal the target model accepts, and how the draft model's own cache is kept
in step with the committed text. The GPU glue is in worker.py (verify) and
draft.py (the draft model's forward); everything here is checked by mock tests.

How a speculative round works (greedy)
--------------------------------------
A sequence has COMMITTED tokens c_0..c_{L-1}. Its KV cache holds positions
0..L-2 (the last committed token is the next step's first input -- the
convention the single-sequence path has always used). A normal decode step
feeds [c_{L-1}] at position L-1 and the last node argmaxes the one row.

A speculative step feeds [c_{L-1}, d_1..d_k] at positions L-1..L-1+k (the
drafts d_i come from a drafter) as ONE forward pass, like a short prefill. Row
j of the output predicts the token at position L+j. With g_j = argmax(row j):

    d_1 is accepted  iff  g_0 == d_1
    d_2 is accepted  iff  d_1 accepted and g_1 == d_2   ... and so on.

If m drafts are accepted the step commits g_0..g_m (m+1 tokens: the m accepted
drafts, which equal g_0..g_{m-1}, plus the "bonus" token g_m the target itself
produced). Every committed token is the target's greedy token for a prefix the
target actually saw, so the output equals plain greedy decoding BYTE FOR BYTE
(up to kernel numerics, see ARCHITECTURE.md "What is proven, what is measured").

The K/V of the rejected drafts was written into the cache anyway; every node
rolls the sequence back to its accepted length (PagedSlots in paged.py).

Sampled (non-greedy) sequences never speculate: exact speculative SAMPLING
needs the draft distribution, which an n-gram drafter does not have; those
sequences run k = 0 (a plain decode row) inside the same batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MAX_DRAFT = 15   # a verify row is 1 + k tokens and must stay in the decode regime (<= 16)
SPEC_MODES = ("none", "ngram", "draft")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class SpecConfig:
    """mode: none | ngram (prompt lookup) | draft (a small model). k: the most
    draft tokens per round. ngram_min/ngram_max: the context suffix lengths
    tried, longest first. adaptive: shrink/grow k from the observed acceptance
    (per sequence)."""
    mode: str = "none"
    k: int = 4
    ngram_min: int = 3
    ngram_max: int = 6
    adaptive: bool = True

    def __post_init__(self):
        if self.mode not in SPEC_MODES:
            raise ValueError(f"spec mode must be one of {SPEC_MODES}, got {self.mode!r}")
        if not 1 <= int(self.k) <= MAX_DRAFT:
            raise ValueError(f"spec k must be in 1..{MAX_DRAFT}, got {self.k}")
        self.k = int(self.k)
        if not 1 <= int(self.ngram_min) <= int(self.ngram_max):
            raise ValueError(f"need 1 <= ngram_min <= ngram_max, got {self.ngram_min}..{self.ngram_max}")
        self.ngram_min, self.ngram_max = int(self.ngram_min), int(self.ngram_max)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def to_dict(self) -> Dict:
        return {"mode": self.mode, "k": self.k, "ngram_min": self.ngram_min,
                "ngram_max": self.ngram_max, "adaptive": self.adaptive}


# --------------------------------------------------------------------------
# acceptance (runs on the LAST node)
# --------------------------------------------------------------------------

def greedy_accept(inputs: Sequence[int], argmaxes: Sequence[int], n: int) -> List[int]:
    """The committed tokens of one verify row.

    inputs[0:n] are the tokens fed (inputs[0] = the last committed token,
    inputs[1:n] = the drafts); argmaxes[j] is the target's greedy token after
    inputs[:j+1], valid for j < n. Returns [g_0, .., g_m] where m is the number
    of accepted drafts (always at least one token)."""
    if n < 1 or len(inputs) < n or len(argmaxes) < n:
        raise ValueError(f"greedy_accept: n={n}, {len(inputs)} inputs, {len(argmaxes)} argmaxes")
    out = [int(argmaxes[0])]
    for j in range(1, n):
        if int(inputs[j]) != int(argmaxes[j - 1]):
            break
        out.append(int(argmaxes[j]))
    return out


# --------------------------------------------------------------------------
# adaptive window
# --------------------------------------------------------------------------

class AdaptiveWindow:
    """Per-sequence draft length. Full acceptance grows it by one; acceptance
    under half of a window of 2+ shrinks it by one. Bounded by [kmin, kmax]."""

    def __init__(self, kmax: int, kmin: int = 1, enabled: bool = True):
        self.kmax = max(1, int(kmax))
        self.kmin = max(1, min(int(kmin), self.kmax))
        self.enabled = enabled
        self.k = self.kmax

    def update(self, proposed: int, accepted: int) -> int:
        if self.enabled and proposed > 0:
            if accepted >= proposed:
                self.k = min(self.kmax, self.k + 1)
            elif proposed >= 2 and accepted * 2 < proposed:
                self.k = max(self.kmin, self.k - 1)
        return self.k


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

@dataclass
class SpecStats:
    """What speculation achieved. A round is one decode row that carried
    drafts; rows with no draft are plain decode rows (counted separately)."""
    rounds: int = 0                 # rows that carried >= 1 draft
    proposed: int = 0               # draft tokens verified
    accepted: int = 0               # draft tokens accepted
    plain_rows: int = 0             # decode rows with no draft
    committed: int = 0              # tokens committed by decode rows (drafted or not)
    by_accepted: Dict[int, int] = field(default_factory=dict)  # histogram: m -> rounds

    def add_row(self, proposed: int, accepted: int, committed: int) -> None:
        self.committed += committed
        if proposed <= 0:
            self.plain_rows += 1
            return
        self.rounds += 1
        self.proposed += proposed
        self.accepted += accepted
        self.by_accepted[accepted] = self.by_accepted.get(accepted, 0) + 1

    def reset(self) -> None:
        self.rounds = self.proposed = self.accepted = self.plain_rows = self.committed = 0
        self.by_accepted = {}

    @property
    def acceptance_rate(self) -> Optional[float]:
        """Accepted draft tokens / proposed draft tokens."""
        return (self.accepted / self.proposed) if self.proposed else None

    @property
    def tokens_per_row(self) -> Optional[float]:
        rows = self.rounds + self.plain_rows
        return (self.committed / rows) if rows else None

    def to_dict(self) -> Dict:
        return {"rounds": self.rounds, "proposed": self.proposed, "accepted": self.accepted,
                "plain_rows": self.plain_rows, "committed": self.committed,
                "acceptance_rate": self.acceptance_rate, "tokens_per_row": self.tokens_per_row,
                "accepted_histogram": {str(k): v for k, v in sorted(self.by_accepted.items())}}


# --------------------------------------------------------------------------
# the drafter interface
# --------------------------------------------------------------------------

@dataclass
class DraftRequest:
    """Ask a drafter for up to k tokens continuing `tokens` (the sequence's
    committed tokens so far, prompt included). `slot` names the sequence; a
    drafter keeps per-slot state keyed by it."""
    slot: int
    tokens: Sequence[int]
    k: int


class Drafter:
    """Base class. open(slot, prompt) when a sequence starts, propose(requests)
    once per round for the sequences about to be verified, close(slot) when the
    sequence ends. propose returns {slot: [draft tokens]} (possibly empty)."""
    name = "none"

    def open(self, slot: int, prompt: Sequence[int]) -> None:
        pass

    def warm(self, slot: int, tokens: Sequence[int], upto: int) -> bool:
        """Optional: pre-feed tokens[:upto] (a prompt being prefilled) one
        chunk per call, so the first proposal does not stall on a long prompt.
        True once the drafter holds tokens[:upto]."""
        return True

    def propose(self, requests: Sequence[DraftRequest]) -> Dict[int, List[int]]:
        return {r.slot: [] for r in requests}

    def close(self, slot: int) -> None:
        pass

    def close_all(self) -> None:
        pass


# --------------------------------------------------------------------------
# n-gram / prompt-lookup drafting
# --------------------------------------------------------------------------

class _NgramIndex:
    """An append-only index of one sequence's tokens. For every suffix length n
    in [nmin, nmax] it maps the n-gram to the end position of its latest
    occurrence, and remembers, for the CURRENT suffix, the end position of the
    occurrence BEFORE it -- which is what prompt lookup continues from."""

    __slots__ = ("nmin", "nmax", "tokens", "last", "prev")

    def __init__(self, nmin: int, nmax: int):
        self.nmin, self.nmax = nmin, nmax
        self.tokens: List[int] = []
        self.last: Dict[int, Dict[Tuple[int, ...], int]] = {n: {} for n in range(nmin, nmax + 1)}
        self.prev: Dict[int, Optional[int]] = {n: None for n in range(nmin, nmax + 1)}

    def append(self, tok: int) -> None:
        toks = self.tokens
        toks.append(int(tok))
        end = len(toks) - 1
        for n in range(self.nmin, self.nmax + 1):
            if end + 1 < n:
                self.prev[n] = None
                continue
            key = tuple(toks[end + 1 - n:end + 1])
            table = self.last[n]
            self.prev[n] = table.get(key)
            table[key] = end

    def lookup(self, k: int) -> List[int]:
        """Up to k tokens that followed the longest earlier occurrence of the
        current suffix (most recent occurrence on a tie in length)."""
        toks = self.tokens
        for n in range(self.nmax, self.nmin - 1, -1):
            e = self.prev[n]
            if e is None:
                continue
            draft = toks[e + 1:e + 1 + k]
            if draft:
                return draft
        return []


class NgramDrafter(Drafter):
    """Prompt-lookup drafting: no model. Per sequence it indexes the committed
    text (prompt + generated) and proposes the tokens that followed the latest
    earlier occurrence of the longest matching suffix. It only ever sees
    COMMITTED tokens, so a rejected draft needs no rollback here."""
    name = "ngram"

    def __init__(self, nmin: int = 3, nmax: int = 6):
        if not 1 <= nmin <= nmax:
            raise ValueError("need 1 <= nmin <= nmax")
        self.nmin, self.nmax = nmin, nmax
        self._idx: Dict[int, _NgramIndex] = {}

    WARM_SLICE = 8192      # tokens indexed per warm() call

    def open(self, slot: int, prompt: Sequence[int]) -> None:
        # an empty index: warm() fills it while the prompt is prefilled, propose() the rest
        self._idx[slot] = _NgramIndex(self.nmin, self.nmax)

    def warm(self, slot: int, tokens: Sequence[int], upto: int) -> bool:
        idx = self._idx.get(slot)
        if idx is None:
            idx = self._idx[slot] = _NgramIndex(self.nmin, self.nmax)
        upto = min(int(upto), len(tokens))
        have = len(idx.tokens)
        for t in tokens[have:min(upto, have + self.WARM_SLICE)]:
            idx.append(t)
        return len(idx.tokens) >= upto

    def close(self, slot: int) -> None:
        self._idx.pop(slot, None)

    def close_all(self) -> None:
        self._idx.clear()

    def propose(self, requests: Sequence[DraftRequest]) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for r in requests:
            idx = self._idx.get(r.slot)
            if idx is None:
                idx = self._idx[r.slot] = _NgramIndex(self.nmin, self.nmax)
            have = len(idx.tokens)
            if have > len(r.tokens):  # the sequence shrank: rebuild (should not happen)
                idx = self._idx[r.slot] = _NgramIndex(self.nmin, self.nmax)
                have = 0
            for t in r.tokens[have:]:
                idx.append(t)
            out[r.slot] = idx.lookup(r.k) if r.k > 0 else []
        return out


# --------------------------------------------------------------------------
# draft-MODEL drafting
# --------------------------------------------------------------------------

class DraftForward:
    """The draft model's compute, one call per batched forward. A backend
    (draft.py for the real exllamav3 model, a deterministic fake in the mock
    tests) implements:

      forward(rows) -> [token per row]
          rows: [(slot, start_pos, tokens)]. Feed `tokens` into the slot's
          cache at positions start_pos.. (the backend first rolls the slot back
          to start_pos if it is ahead, and refuses a start_pos beyond what it
          holds). Returns the greedy token after the LAST fed token of each row.
      prefill(slot, start_pos, tokens)
          the same, no output wanted (a long catch-up, fed in chunks).
      close(slot) / close_all()
          free the slot's cache and state.
    """

    def forward(self, rows: Sequence[Tuple[int, int, Sequence[int]]]) -> List[int]:
        raise NotImplementedError

    def prefill(self, slot: int, start_pos: int, tokens: Sequence[int]) -> None:
        raise NotImplementedError

    def close(self, slot: int) -> None:
        pass

    def close_all(self) -> None:
        pass


class DraftModelDrafter(Drafter):
    """Draft with a small model that shares the target's tokenizer.

    The draft keeps its own cache per slot. What it has FED is remembered
    (`fed[slot]`: committed tokens followed by its own earlier drafts); at the
    next round the longest prefix of `fed` that still equals the committed text
    is kept, the rest is rolled back (the backend rewinds the slot), and the
    committed tail is fed -- normally 1 token (the bonus token after a
    rejection) or 2 (after a fully accepted window), more only on the first
    round, when the prompt is fed. Then k-1 single-token steps extend the
    proposal; the last proposed token is never fed. Every forward batches all
    the sequences of the round."""
    name = "draft"

    def __init__(self, backend: DraftForward, prefill_chunk: int = 2048, max_catchup: int = 2):
        self.backend = backend
        self.prefill_chunk = max(1, int(prefill_chunk))
        self.max_catchup = max(1, int(max_catchup))
        self.fed: Dict[int, List[int]] = {}
        self.forwards = 0          # batched forward calls (a cost metric)
        self.fed_tokens = 0        # tokens pushed through the draft model

    def open(self, slot: int, prompt: Sequence[int]) -> None:
        self.backend.close(slot)
        self.fed[slot] = []

    def close(self, slot: int) -> None:
        self.fed.pop(slot, None)
        self.backend.close(slot)

    def close_all(self) -> None:
        self.fed.clear()
        self.backend.close_all()

    def warm(self, slot: int, tokens: Sequence[int], upto: int) -> bool:
        fed = self.fed.setdefault(slot, [])
        upto = min(int(upto), len(tokens))
        valid = self._valid_prefix(fed, tokens[:upto])
        del fed[valid:]
        end = min(upto, valid + self.prefill_chunk)
        if end > valid:
            piece = list(tokens[valid:end])
            self.backend.prefill(slot, valid, piece)
            self.fed_tokens += len(piece)
            fed.extend(piece)
        return len(fed) >= upto

    def _valid_prefix(self, fed: List[int], committed: Sequence[int]) -> int:
        n = min(len(fed), len(committed))
        i = 0
        while i < n and fed[i] == committed[i]:
            i += 1
        return i

    def propose(self, requests: Sequence[DraftRequest]) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {r.slot: [] for r in requests}
        live = [r for r in requests if r.k > 0 and len(r.tokens) >= 1]
        if not live:
            return out
        # 1. align each slot's cache with the committed text and catch up
        rows = []   # (request, start_pos, tokens-to-feed)
        for r in live:
            fed = self.fed.setdefault(r.slot, [])
            valid = self._valid_prefix(fed, r.tokens)
            del fed[valid:]                          # the divergent tail is rolled back
            tail = list(r.tokens[valid:])            # >= 1 token: the last committed one is never fed yet
            if not tail:
                # the cache already holds the whole committed text (cannot happen after a
                # round, since the last committed token is fed only now): re-feed it
                valid -= 1
                del fed[valid:]
                tail = [int(r.tokens[valid])]
            # a long catch-up (the prompt) goes in as chunked prefill; the last
            # `max_catchup` tokens ride the batched forward
            if len(tail) > self.max_catchup:
                head, tail = tail[:-self.max_catchup], tail[-self.max_catchup:]
                p = valid
                for a in range(0, len(head), self.prefill_chunk):
                    piece = head[a:a + self.prefill_chunk]
                    self.backend.prefill(r.slot, p, piece)
                    self.fed_tokens += len(piece)
                    fed.extend(piece)
                    p += len(piece)
                valid = p
            rows.append((r, valid, tail))
        toks = self.backend.forward([(r.slot, p, t) for r, p, t in rows])
        self.forwards += 1
        active = []
        for (r, p, t), nxt in zip(rows, toks):
            self.fed[r.slot].extend(t)
            self.fed_tokens += len(t)
            out[r.slot].append(int(nxt))
            if len(out[r.slot]) < r.k:
                active.append(r)
        # 2. extend: feed the newest draft token, get the next
        while active:
            step_rows = []
            for r in active:
                pos = len(self.fed[r.slot])
                step_rows.append((r.slot, pos, [out[r.slot][-1]]))
            toks = self.backend.forward(step_rows)
            self.forwards += 1
            nxt_active = []
            for r, nxt in zip(active, toks):
                self.fed[r.slot].append(out[r.slot][-1])
                self.fed_tokens += 1
                out[r.slot].append(int(nxt))
                if len(out[r.slot]) < r.k:
                    nxt_active.append(r)
            active = nxt_active
        return out


def make_drafter(cfg: SpecConfig, backend: Optional[DraftForward] = None,
                 prefill_chunk: int = 2048) -> Drafter:
    """The drafter for a config: NgramDrafter, DraftModelDrafter (needs a
    backend) or the no-op base class."""
    if cfg.mode == "ngram":
        return NgramDrafter(cfg.ngram_min, cfg.ngram_max)
    if cfg.mode == "draft":
        if backend is None:
            raise ValueError("spec mode 'draft' needs a draft-model backend")
        return DraftModelDrafter(backend, prefill_chunk=prefill_chunk)
    return Drafter()
