# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/paged.py -- everything a BigBrain node needs to serve SEVERAL
sequences at once from one exllamav3 Cache.

SLOTS. A node's Cache is built for `n_slots` sequences: slot s owns the pages
[s*slot_pages, (s+1)*slot_pages) of the cache (slot_pages = context pages + one
slack page, so a speculative window that pads past the context can never write
into the next slot). A step then runs as exllamav3's PAGED batch -- params
{attn_mode: flash_attn, cache, block_table (B, pages), cache_seqlens (B),
recurrent_states [one per row]} -- exactly how its own generator drives a batch
(generator.py iterate_gen), with a block table row per sequence.

ROLLBACK. The cache is positional: whatever sits past a sequence's valid length
is garbage that the next write overwrites. `PagedSlots.prepare()` is the single
place a node learns where each sequence now stands. Every step names the
position it starts at (`pos`); if the node has written beyond it (a speculative
window the last node only partly accepted, or the padding of a ragged batch)
that is a ROLLBACK to `pos`:

  * the attention cache: nothing to undo -- cache_seqlens = pos is the truncation;
  * recurrent state (gpt-oss sliding-window layers, GatedDeltaNet): state.rewind(n)
    (SWAState.rewind moves its position back inside its stored window;
    GDNState.rewind restores the per-token history the forward recorded when
    params["recurrent_history"] was set).

Each node does this on its own, from the `pos` in the step message, so no node
has to be told the accepted length separately and a node that missed the
update cannot silently continue from a stale position: a step that starts
BEYOND what the node holds is an error, not a guess.

WIRE. A ragged batch (rows of different valid length n_i padded to T) crosses
the wire packed: (1, sum n_i, hidden) -- no padding bytes -- and is unpacked to
zeros-padded (B, T, hidden) on arrival.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

PAGE_SIZE = 256        # exllamav3 constants.PAGE_SIZE
PROTO = 3              # the batched-step protocol (CONTROL ops on pipe v2 frames)


class BigBrainStateError(RuntimeError):
    """The recurrent-state / slot contract was violated (a step for a slot this
    node does not hold, at a position it cannot reach)."""


# --------------------------------------------------------------------------
# step metadata (the CONTROL 'step' frame)
# --------------------------------------------------------------------------

@dataclass
class StepRow:
    rid: int
    slot: int
    pos: int          # sequence position of the row's first token (its past length)
    n: int            # valid tokens in the row (<= T)
    ids: List[int]    # the n input token ids (needed downstream: PLE layers, acceptance)


@dataclass
class StepMeta:
    step: int
    kind: str                 # "decode" | "prefill"
    T: int                    # padded width of the activation
    rows: List[StepRow]
    packed: bool = False      # the activation carries only the valid rows

    @property
    def ns(self) -> List[int]:
        return [r.n for r in self.rows]

    @property
    def ragged(self) -> bool:
        return any(r.n != self.T for r in self.rows)

    def to_dict(self) -> Dict:
        return {"op": "step", "step": self.step, "kind": self.kind, "T": self.T,
                "packed": self.packed,
                "rows": [{"rid": r.rid, "slot": r.slot, "pos": r.pos, "n": r.n, "ids": r.ids}
                         for r in self.rows]}

    @classmethod
    def from_dict(cls, d: Dict) -> "StepMeta":
        try:
            rows = [StepRow(int(r["rid"]), int(r["slot"]), int(r["pos"]), int(r["n"]),
                            [int(t) for t in r["ids"]]) for r in d["rows"]]
            m = cls(int(d["step"]), str(d["kind"]), int(d["T"]), rows, bool(d.get("packed")))
        except (KeyError, TypeError, ValueError) as exc:
            raise BigBrainStateError(f"malformed step frame: {exc!r}")
        if m.kind not in ("decode", "prefill") or not rows or m.T < 1:
            raise BigBrainStateError(f"malformed step frame: kind={m.kind!r} rows={len(rows)} T={m.T}")
        for r in rows:
            if not 1 <= r.n <= m.T or len(r.ids) != r.n or r.pos < 0:
                raise BigBrainStateError(
                    f"malformed step row: n={r.n} T={m.T} ids={len(r.ids)} pos={r.pos}")
        if len({r.slot for r in rows}) != len(rows):
            raise BigBrainStateError("a step names the same slot twice")
        return m


# --------------------------------------------------------------------------
# ragged packing (torch tensors)
# --------------------------------------------------------------------------

def pack_rows(x, ns: Sequence[int]):
    """(B, T, ...) -> (1, sum(ns), ...): the first n_i positions of each row."""
    import torch
    return torch.cat([x[i, :n] for i, n in enumerate(ns)], dim=0).unsqueeze(0)


def unpack_rows(x, ns: Sequence[int], T: int):
    """(1, sum(ns), ...) -> zero-padded (B, T, ...)."""
    import torch
    out = torch.zeros((len(ns), T) + tuple(x.shape[2:]), dtype=x.dtype, device=x.device)
    a = 0
    for i, n in enumerate(ns):
        out[i, :n] = x[0, a:a + n]
        a += n
    return out


# --------------------------------------------------------------------------
# slots
# --------------------------------------------------------------------------

@dataclass
class _Slot:
    rid: Optional[int] = None
    written: int = 0           # positions physically written (the cache may hold garbage past the valid length)
    state: object = None       # the recurrent state handle (recurrent models)


class PagedSlots:
    """Per-node bookkeeping for `n_slots` sequences in one Cache.

    new_state(slot) -> state     builds this node's recurrent state at that slot (None for dense models)
    release_state(state)         returns it to the cache
    recurrent                    True when the model carries recurrent state (every node then holds a
                                 state per live sequence, even a node whose slice has no recurrent layer)
    """

    def __init__(self, n_slots: int, ctx: int, *, recurrent: bool = False,
                 new_state: Optional[Callable] = None,
                 release_state: Optional[Callable] = None,
                 on_rollback: Optional[Callable[[int, int, int], None]] = None):
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if ctx < 1:
            raise ValueError("ctx must be >= 1")
        self.n_slots = int(n_slots)
        self.ctx = int(ctx)
        self.ctx_pages = -(-self.ctx // PAGE_SIZE)
        # one slack page: padding of a ragged/speculative row may run past the context
        self.slot_pages = self.ctx_pages + 1
        self.recurrent = recurrent
        self._new_state = new_state
        self._release_state = release_state
        self._on_rollback = on_rollback
        self.slots = [_Slot() for _ in range(self.n_slots)]
        self.rollbacks = 0          # rows that started behind what the node had written
        self.rolled_tokens = 0      # positions given up by those rollbacks

    # ---- sizes the cache must be built with

    @staticmethod
    def tokens_for(n_slots: int, ctx: int) -> int:
        """Cache(max_num_tokens=...) for n_slots sequences of ctx tokens."""
        return int(n_slots) * (-(-int(ctx) // PAGE_SIZE) + 1) * PAGE_SIZE

    @property
    def cache_tokens(self) -> int:
        return self.n_slots * self.slot_pages * PAGE_SIZE

    # ---- sequence lifecycle

    def begin(self, slot: int, rid: int) -> None:
        self._check_slot(slot)
        self.end(slot)
        s = self.slots[slot]
        s.rid, s.written = int(rid), 0
        if self.recurrent:
            s.state = self._new_state(slot)

    def end(self, slot: int) -> None:
        self._check_slot(slot)
        s = self.slots[slot]
        st, s.state = s.state, None
        s.rid, s.written = None, 0
        if st is not None and self._release_state is not None:
            self._release_state(st)

    def end_all(self) -> None:
        for i in range(self.n_slots):
            self.end(i)

    def owner(self, slot: int) -> Optional[int]:
        return self.slots[slot].rid

    def written(self, slot: int) -> int:
        return self.slots[slot].written

    def state(self, slot: int):
        return self.slots[slot].state

    def active(self) -> List[int]:
        return [i for i, s in enumerate(self.slots) if s.rid is not None]

    def invalidate(self, slot: int) -> None:
        """A forward failed half-way: the slot's cache/state is unusable. The
        owner stays, so the next step for it fails loudly instead of running on
        a half-advanced state."""
        s = self.slots[slot]
        s.written = -1

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.n_slots:
            raise BigBrainStateError(f"slot {slot} outside this node's {self.n_slots} slot(s) "
                                     f"(started with fewer --slots?)")

    # ---- one step

    def prepare(self, rows: Sequence[Tuple[int, int, int, int]], device=None):
        """rows: [(slot, rid, pos, T)]. Rolls each slot back to `pos` when it is
        ahead, and returns (block_table (B, W) int32, cache_seqlens (B,) int32,
        states or None). Raises BigBrainStateError for a slot another sequence
        owns, a slot whose last forward failed, or a gap (pos beyond what the
        node holds)."""
        import torch
        if not rows:
            raise ValueError("prepare() needs at least one row")
        states = [] if self.recurrent else None
        need_pages = 0
        for slot, rid, pos, T in rows:
            self._check_slot(slot)
            s = self.slots[slot]
            if s.rid != rid:
                raise BigBrainStateError(
                    f"slot {slot} is held by request {s.rid}, the step is for request {rid}")
            if s.written < 0:
                raise BigBrainStateError(f"slot {slot}: an earlier step failed; the sequence is dead")
            if pos > s.written:
                raise BigBrainStateError(
                    f"slot {slot} request {rid}: the step starts at position {pos} but this node "
                    f"has only written {s.written} (a step was lost?)")
            if pos + T > self.slot_pages * PAGE_SIZE:
                raise BigBrainStateError(
                    f"slot {slot}: positions {pos}..{pos + T - 1} overrun the slot "
                    f"({self.slot_pages * PAGE_SIZE} tokens)")
            if self.recurrent and s.state is not None and s.state.position != s.written:
                raise BigBrainStateError(
                    f"slot {slot} request {rid}: the recurrent state is at position {s.state.position} "
                    f"but this node wrote {s.written} (the state drifted from the cache)")
            if pos < s.written:
                self._rollback(slot, s, pos)
            elif self.recurrent and s.state is not None:
                self._settle_state(slot, s.state, pos)
            if self.recurrent:
                if s.state is None or s.state.position != pos:
                    raise BigBrainStateError(
                        f"slot {slot} request {rid}: the recurrent state is at position "
                        f"{getattr(s.state, 'position', None)}, the step starts at {pos}")
                states.append(s.state)
            need_pages = max(need_pages, -(-(pos + T) // PAGE_SIZE))
        # block-table width: the pages the batch can touch, rounded up to a
        # multiple of 16 like exllamav3's generator (a few distinct widths only)
        width = min(self.slot_pages, -(-need_pages // 16) * 16)
        bt = torch.tensor([[slot * self.slot_pages + j for j in range(width)]
                           for slot, _, _, _ in rows], dtype=torch.int32)
        seqlens = torch.tensor([pos for _, _, pos, _ in rows], dtype=torch.int32)
        return bt, seqlens, states

    @staticmethod
    def _settle_state(slot: int, st, pos: int) -> None:
        """Bring a recurrent state to `pos`: rewind it by however far ahead it is. A state whose
        last forward recorded per-token history (GatedDeltaNet under params recurrent_history:
        st.last_history > 0) must be rewound even when it is NOT ahead -- rewind(0) -- or its
        conv window stays in history layout (exllamav3's generator does the same after a fully
        accepted draft). A sliding-window state (SWAState) rewinds in place within its window."""
        behind = st.position - pos
        rewind = getattr(st, "rewind", None)
        if behind < 0 or (behind > 0 and rewind is None):
            raise BigBrainStateError(
                f"slot {slot}: the recurrent state is at position {st.position}, cannot reach {pos}"
                + ("" if rewind is not None else " (this state type cannot roll back)"))
        if rewind is not None and (behind > 0 or getattr(st, "last_history", 0) > 0):
            rewind(behind)

    def _rollback(self, slot: int, s: _Slot, pos: int) -> None:
        n = s.written - pos
        if self.recurrent and s.state is not None:
            self._settle_state(slot, s.state, pos)
        self.rollbacks += 1
        self.rolled_tokens += n
        s.written = pos
        if self._on_rollback is not None:
            self._on_rollback(slot, n, pos)

    def commit(self, rows: Sequence[Tuple[int, int, int, int]]) -> None:
        """The forward succeeded: each slot now holds pos + T written positions."""
        for slot, _rid, pos, T in rows:
            self.slots[slot].written = pos + T

    def stats(self) -> Dict:
        return {"slots": self.n_slots, "active": len(self.active()),
                "rollbacks": self.rollbacks, "rolled_tokens": self.rolled_tokens}


# --------------------------------------------------------------------------
# per-node timers
# --------------------------------------------------------------------------

class NodeStats:
    """Where a node's wall-clock time goes since the last reset: busy (inside a
    step's compute, incl. the device sync), wait (blocked on an empty inbound
    queue: the hop wait), send (in socket sends), plus step/token counters."""

    def __init__(self, rank: int):
        self.rank = rank
        self._t0 = time.monotonic()
        self.busy_s = self.wait_s = self.send_s = 0.0
        self.steps = self.prefill_steps = 0
        self.rows = self.tokens = 0
        self.errors = 0

    def reset(self) -> None:
        self.__init__(self.rank)

    def snapshot(self, extra: Optional[Dict] = None) -> Dict:
        el = max(time.monotonic() - self._t0, 1e-9)
        d = {"rank": self.rank, "elapsed_s": round(el, 3),
             "busy_s": round(self.busy_s, 4), "wait_s": round(self.wait_s, 4),
             "send_s": round(self.send_s, 4),
             "busy_pct": round(100.0 * self.busy_s / el, 2),
             "wait_pct": round(100.0 * self.wait_s / el, 2),
             "send_pct": round(100.0 * self.send_s / el, 2),
             "steps": self.steps, "prefill_steps": self.prefill_steps,
             "rows": self.rows, "tokens": self.tokens, "errors": self.errors}
        if extra:
            d.update(extra)
        return d
