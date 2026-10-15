# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/engine.py -- node 0's batched pipeline engine.

The single-sequence path (worker.SliceWorker.generate) keeps ONE token in the
pipeline at a time: node k computes while every other node waits, and each
token pays the full chain of hops. The engine fills the pipeline instead:

* MULTI-SEQUENCE. N sequences (slots) are in flight at once. A decode STEP
  carries one or more sequences' next tokens as a rectangular (B, T) batch;
  while node k computes step A, node k+1 computes step B. Steps in flight are
  capped at `inflight` (default: the number of nodes); each step carries up to
  `mb_size` sequences (default: enough to spread the active sequences over the
  in-flight window, so one big batch never starves the pipeline's depth).
* SPECULATION. A sequence's row in a decode step is [last committed token,
  d_1..d_k]: the drafter's k proposals verified in the same forward pass (spec.py).
  The last node returns each row's committed tokens (1..k+1 of them), and the
  next step of that sequence starts at its accepted length -- every node rolls
  the sequence's cache back to that position (paged.PagedSlots).
  Both combine: each sequence drafts and verifies on its own, rows of
  different draft length share a step (shorter rows are padded; padding is
  just more rejected positions, which the rollback already handles).

Threads: the DRIVER (admit sequences, build and run steps, apply results) and a
RECEIVER (reads the token link, hands results to the driver). Node 0 also
computes its own slice inside the driver, overlapped with the downstream nodes
by the async sender.

Greedy sequences speculate; sampled ones ride along as plain one-token rows.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .paged import PROTO, BigBrainStateError, StepMeta, StepRow
from .pipe import BigBrainPeerDead, BigBrainProtocolError
from .spec import (MAX_DRAFT, AdaptiveWindow, DraftRequest, Drafter, NgramDrafter,
                   SpecConfig, SpecStats)

log = logging.getLogger("bigbrain.engine")


class BigBrainRemoteError(RuntimeError):
    """A node reported an error for a request."""


class EngineDead(RuntimeError):
    """The pipeline is gone (a link died, the handshake failed)."""


@dataclass
class EngineConfig:
    spec: SpecConfig = field(default_factory=SpecConfig)
    inflight: int = 0            # decode steps in flight (0 = the number of nodes)
    mb_size: int = 0             # max sequences per decode step (0 = auto)
    prefill_chunk: int = 2048    # prompt tokens per prefill step
    prefill_window: int = 2      # prefill steps in flight
    hello_timeout_s: Optional[float] = None   # None = wait for as long as the nodes take to load
    result_timeout_s: float = 900.0           # a step in flight this long = a dead pipeline


# --------------------------------------------------------------------------
# a sequence
# --------------------------------------------------------------------------

@dataclass
class _Seq:
    rid: int
    prompt: List[int]
    settings: object
    max_new: int
    stop_ids: frozenset
    spec_mode: str
    k_max: int
    diag: bool
    info: Dict
    out: "queue.Queue" = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    t_submit: float = field(default_factory=time.monotonic)
    # runtime (driver thread only)
    slot: int = -1
    phase: str = "queued"          # queued | prefill | decode | done
    tokens: List[int] = field(default_factory=list)   # committed: prompt + generated
    prefill_sent: int = 0
    prefill_inflight: int = 0
    inflight: bool = False
    ready_since: float = 0.0
    window: Optional[AdaptiveWindow] = None
    produced: int = 0
    t_admit: float = 0.0
    t_first: Optional[float] = None
    proposed: int = 0
    accepted: int = 0
    rounds: int = 0
    margins: List[float] = field(default_factory=list)
    greedy: bool = True


@dataclass
class _Step:
    meta: StepMeta
    t_sent: float
    rids: List[int]


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

class PipelineEngine:
    """Node 0's batched engine. `worker` is the node-0 SliceWorker (built with
    slots >= this engine's capacity); `drafters` maps a spec mode to its
    Drafter ('ngram' is always available, 'draft' when a draft model is loaded)."""

    def __init__(self, worker, cfg: Optional[EngineConfig] = None,
                 drafters: Optional[Dict[str, Drafter]] = None):
        if not worker.is_first:
            raise ValueError("the engine runs on node 0")
        self.w = worker
        self.cfg = cfg or EngineConfig()
        self.slots = worker.n_slots
        self.ctx = worker.max_num_tokens
        self.n_nodes = worker.total
        self.inflight_cap = self.cfg.inflight or max(1, self.n_nodes)
        self.drafters: Dict[str, Drafter] = {"ngram": NgramDrafter(self.cfg.spec.ngram_min,
                                                                   self.cfg.spec.ngram_max)}
        self.drafters.update(drafters or {})
        self.eos_ids: Set[int] = set()
        self.max_spec_k = MAX_DRAFT
        # queues and maps
        self._admit_q: Deque[_Seq] = collections.deque()
        self._lock = threading.Lock()            # guards _admit_q and the stats snapshot
        self._events: "queue.Queue[Dict]" = queue.Queue()   # receiver -> driver (+ wake-ups)
        self._boxes: Dict[str, "queue.Queue"] = {"hello_ack": queue.Queue(), "stats_result": queue.Queue()}
        self._seqs: Dict[int, _Seq] = {}         # rid -> live sequence (driver thread)
        self._free: List[int] = list(range(self.slots))
        self._steps: Dict[int, _Step] = {}
        self._step_id = 0
        self._rid = 0
        self._turn_decode = True
        self._stop = threading.Event()
        self.dead: Optional[str] = None
        self.ready = threading.Event()
        self._driver: Optional[threading.Thread] = None
        self._receiver: Optional[threading.Thread] = None
        self.node_caps: List[Dict] = []
        self.reset_stats()

    # ------------------------------------------------------------ lifecycle

    def start(self, hello_timeout_s: Optional[float] = None) -> List[Dict]:
        """Start the receiver, run the handshake (every node must speak the
        batched protocol, hold enough slots/context, and -- for a recurrent
        model with speculation -- enough state history), then start the driver.
        Raises EngineDead when a node does not answer or cannot serve."""
        w = self.w
        if w.total > 1 and w.token_link is None:
            raise EngineDead("node 0 has no token link")
        self._receiver = threading.Thread(target=self._recv_loop, name="bigbrain-recv", daemon=True)
        self._receiver.start()
        caps0 = w.capabilities()
        nodes = [caps0]
        if w.total > 1:
            timeout = self.cfg.hello_timeout_s if hello_timeout_s is None else hello_timeout_s
            with w.tx_lock:
                w._tx_out.send_control({"op": "hello", "proto": PROTO, "nodes": [caps0],
                                        "slots": self.slots, "ctx": self.ctx}, request_id=0)
            ack = self._wait_box("hello_ack", timeout, "the nodes' handshake (a node on an older "
                                 "build never answers: stage the same bigbrain/ package on every node)")
            nodes = ack.get("nodes") or []
        self._validate_nodes(nodes)
        self.node_caps = nodes
        self._driver = threading.Thread(target=self._drive, name="bigbrain-driver", daemon=True)
        self._driver.start()
        self.ready.set()
        log.info("engine ready: %d node(s), %d slot(s), ctx %d, in flight %d, spec %s",
                 self.n_nodes, self.slots, self.ctx, self.inflight_cap, self.cfg.spec.mode)
        return nodes

    def _validate_nodes(self, nodes: List[Dict]) -> None:
        if len(nodes) != self.n_nodes:
            raise EngineDead(f"the handshake covered {len(nodes)} of {self.n_nodes} nodes "
                             f"(a node on an older build relays but does not answer)")
        spec_k = max(self.cfg.spec.k, 1) if self.cfg.spec.enabled else 0
        for n in nodes:
            r = n.get("rank")
            if int(n.get("proto", 0)) < PROTO:
                raise EngineDead(f"node {r} speaks protocol {n.get('proto')}, need {PROTO}")
            if not n.get("batched"):
                raise EngineDead(f"node {r} was started without --slots/--max-draft (single-sequence cache layout)")
            if int(n.get("slots", 0)) < self.slots:
                raise EngineDead(f"node {r} has {n.get('slots')} slot(s), the engine needs {self.slots}")
            if int(n.get("ctx", 0)) < self.ctx:
                raise EngineDead(f"node {r} has context {n.get('ctx')}, the engine needs {self.ctx}")
            if n.get("needs_history"):
                self.max_spec_k = min(self.max_spec_k, int(n.get("max_draft", 0)))
        if spec_k and self.max_spec_k < spec_k:
            raise EngineDead(f"speculation k={spec_k} needs --max-draft {spec_k} on every node of a "
                             f"model whose recurrent state rolls back from recorded history "
                             f"(GatedDeltaNet); the smallest on the pipeline is {self.max_spec_k}")

    def _wait_box(self, name: str, timeout: Optional[float], what: str) -> Dict:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                raise EngineDead(f"timed out waiting for {what}")
            try:
                got = self._boxes[name].get(timeout=1.0 if left is None else min(1.0, left))
            except queue.Empty:
                if self.dead:
                    raise EngineDead(self.dead)
                continue
            if got.get("op") == "_dead":
                raise EngineDead(got.get("message", "the pipeline died"))
            return got

    def close(self) -> None:
        self._stop.set()
        self._events.put({"op": "_wake"})
        if self._driver is not None:
            self._driver.join(timeout=5.0)
        if self._receiver is not None:
            # the receiver exits when the token link closes (worker.close())
            pass

    # --------------------------------------------------------------- receiver

    def _recv_loop(self) -> None:
        link = self.w.token_link
        if link is None:   # single node: results are produced locally
            return
        while not self._stop.is_set():
            try:
                msg = link.recv_message(include_heartbeats=True)
            except (BigBrainPeerDead, BigBrainProtocolError) as exc:
                self._die(f"the token link died: {exc}")
                return
            except Exception as exc:  # noqa: BLE001 - never die silently: /health must say so (security review M3)
                log.exception("the token-link reader crashed")
                self._die(f"the token-link reader crashed: {type(exc).__name__}: {exc}")
                try:
                    link.close()
                except Exception:  # noqa: BLE001
                    pass
                return
            if msg.kind == "heartbeat":
                continue
            if msg.kind == "bye":
                self._die("the last node closed the token link")
                return
            if msg.kind != "control":
                log.debug("ignoring a %s frame on the token link", msg.kind)
                continue
            op = msg.control.get("op")
            if op in self._boxes:
                self._boxes[op].put(msg.control)
            else:
                self._events.put(msg.control)

    def _die(self, why: str) -> None:
        self.dead = self.dead or why
        for box in self._boxes.values():
            box.put({"op": "_dead", "message": why})
        self._events.put({"op": "_dead", "message": why})

    # ------------------------------------------------------------- statistics

    def reset_stats(self) -> None:
        self._t_reset = time.monotonic()
        self.spec_stats = SpecStats()
        self._decode_tokens = 0
        self._prefill_tokens = 0
        self._steps_done = {"decode": 0, "prefill": 0}
        self._step_lat: Deque[float] = collections.deque(maxlen=2048)
        self._inflight_sum = 0
        self._inflight_n = 0
        self._inflight_max = 0
        self._first_lat: List[float] = []
        self._prefill_rates: List[Tuple[int, float]] = []
        self._seq_rates: List[float] = []
        self._active_s = 0.0
        # a reset in the middle of a run keeps counting active time from now
        self._t_active: Optional[float] = time.monotonic() if getattr(self, "_seqs", None) else None
        self._idle_s = 0.0
        self._requests = 0
        self._failed = 0
        self._rows_per_step_sum = 0
        self._t_per_step_sum = 0

    def stats(self) -> Dict:
        now = time.monotonic()
        active = self._active_s + ((now - self._t_active) if self._t_active else 0.0)
        lat = sorted(self._first_lat)
        sl = sorted(self._step_lat)

        def pct(a, p):
            return a[min(len(a) - 1, int(len(a) * p))] if a else None

        dec_steps = self._steps_done["decode"]
        pre_tok = sum(n for n, _ in self._prefill_rates)
        pre_s = sum(t for _, t in self._prefill_rates)
        return {
            "slots": self.slots, "active": len(self._seqs), "queued": len(self._admit_q),
            "nodes": self.n_nodes, "inflight_cap": self.inflight_cap,
            "window_s": round(now - self._t_reset, 3), "active_s": round(active, 3),
            "requests": self._requests, "failed": self._failed,
            "decode_tokens": self._decode_tokens,
            "decode_tok_s": round(self._decode_tokens / active, 3) if active > 0 else None,
            "prefill_tokens": self._prefill_tokens,
            "prefill_tok_s": round(pre_tok / pre_s, 2) if pre_s > 0 else None,
            "per_sequence_tok_s": (round(sum(self._seq_rates) / len(self._seq_rates), 3)
                                   if self._seq_rates else None),
            "first_token_s": {"n": len(lat), "mean": round(sum(lat) / len(lat), 4) if lat else None,
                              "p50": pct(lat, 0.5), "max": lat[-1] if lat else None},
            "steps": dict(self._steps_done),
            "step_latency_s": {"mean": round(sum(sl) / len(sl), 5) if sl else None,
                               "p50": pct(sl, 0.5), "p95": pct(sl, 0.95)},
            "inflight": {"mean": round(self._inflight_sum / self._inflight_n, 3) if self._inflight_n else None,
                         "max": self._inflight_max},
            "rows_per_decode_step": (round(self._rows_per_step_sum / dec_steps, 3) if dec_steps else None),
            "driver_idle_pct": round(100.0 * self._idle_s / max(now - self._t_reset, 1e-9), 2),
            "spec": {"mode": self.cfg.spec.mode, "k": self.cfg.spec.k, **self.spec_stats.to_dict(),
                     "drafters": {n: getattr(d, "forwards", None) for n, d in self.drafters.items()
                                  if getattr(d, "forwards", None) is not None}},
            "dead": self.dead,
        }

    def collect_node_stats(self, reset: bool = False, timeout: float = 15.0) -> List[Dict]:
        """Every node's timers/counters/VRAM (stats() has node 0's engine view)."""
        w = self.w
        mine = w.node_stats(reset=reset)
        if w.total == 1:
            return [mine]
        box = self._boxes["stats_result"]
        while True:                      # drop answers to earlier, timed-out requests
            try:
                box.get_nowait()
            except queue.Empty:
                break
        with w.tx_lock:
            w._tx_out.send_control({"op": "stats", "reset": bool(reset), "nodes": [mine]}, request_id=0)
        return self._wait_box("stats_result", timeout, "the nodes' stats").get("nodes") or []

    def report(self, reset: bool = False) -> Dict:
        """The engine's view + every node's: what a benchmark records per run."""
        eng = self.stats()
        try:
            nodes = self.collect_node_stats(reset=reset)
        except EngineDead as exc:
            nodes = [{"error": str(exc)}]
        if reset:
            self.reset_stats()
        return {"engine": eng, "nodes": nodes}

    # -------------------------------------------------------------- client API

    def submit(self, prompt_ids: Sequence[int], settings, max_new: int,
               request_id: Optional[int] = None, stop_ids: Sequence[int] = (),
               spec: Optional[str] = None, spec_k: Optional[int] = None,
               diag: bool = False, info: Optional[Dict] = None) -> _Seq:
        if self.dead:
            raise EngineDead(self.dead)
        prompt = [int(t) for t in prompt_ids]
        S = len(prompt)
        if S == 0:
            raise ValueError("empty prompt")
        if max_new < 1:
            raise ValueError("max_new must be >= 1")
        if S + max_new > self.ctx:
            raise ValueError(f"prompt ({S}) + max_new ({max_new}) exceeds max_num_tokens={self.ctx}")
        mode = spec or self.cfg.spec.mode
        if mode not in ("none", "ngram", "draft"):
            raise ValueError(f"spec must be none, ngram or draft, got {mode!r}")
        if mode == "draft" and "draft" not in self.drafters:
            raise ValueError("spec 'draft' needs a draft model (--draft-model-dir)")
        k = self.cfg.spec.k if spec_k is None else int(spec_k)
        if not 1 <= k <= MAX_DRAFT:
            raise ValueError(f"spec_k must be in 1..{MAX_DRAFT}")
        if mode != "none" and k > self.max_spec_k:
            raise ValueError(f"spec_k {k} exceeds what the nodes were started for ({self.max_spec_k})")
        with self._lock:
            self._rid += 1
            rid = int(request_id) if request_id is not None else self._rid
        seq = _Seq(rid=rid, prompt=prompt, settings=settings, max_new=int(max_new),
                   stop_ids=frozenset(int(t) for t in stop_ids), spec_mode=mode, k_max=k,
                   diag=bool(diag), info=info if info is not None else {},
                   greedy=bool(getattr(settings, "greedy", True)))
        with self._lock:
            self._admit_q.append(seq)
        self._events.put({"op": "_wake"})
        return seq

    def generate(self, prompt_ids: Sequence[int], settings, max_new: int,
                 request_id: Optional[int] = None, return_logits: bool = False,
                 token_timeout_s: float = 600.0, stop_ids: Sequence[int] = (),
                 spec: Optional[str] = None, spec_k: Optional[int] = None,
                 diag: bool = False, info: Optional[Dict] = None) -> Iterator[Tuple[int, None]]:
        """The server's call (same shape as SliceWorker.generate): yields
        (token_id, None) per committed token. Close the generator (contextlib.
        closing) to cancel; the sequence's slot is freed once its step in flight
        lands. Concurrent generate() calls share the pipeline (up to `slots`
        run at once, the rest wait for a slot)."""
        if return_logits:
            raise ValueError("logits are not available in batched mode; start the pipeline with "
                             "--slots 1 --spec none (the single-sequence engine)")
        seq = self.submit(prompt_ids, settings, max_new, request_id, stop_ids, spec, spec_k, diag, info)
        try:
            while True:
                try:
                    ev = seq.out.get(timeout=token_timeout_s)
                except queue.Empty:
                    raise TimeoutError(f"no token for request {seq.rid} within {token_timeout_s:.0f}s")
                if ev[0] == "tok":
                    for t in ev[1]:
                        yield int(t), None
                elif ev[0] == "done":
                    return
                else:
                    raise BigBrainRemoteError(str(ev[1]))
        finally:
            seq.cancel.set()
            self._events.put({"op": "_wake"})

    # ------------------------------------------------------------------ driver

    def _drive(self) -> None:
        try:
            while not self._stop.is_set():
                worked = self._pump_events(block=False)
                worked |= self._admit()
                worked |= self._reap_cancelled()
                worked |= self._send_next()
                self._check_timeouts()
                if not worked:
                    t0 = time.monotonic()
                    self._pump_events(block=True)
                    idle = time.monotonic() - t0
                    self._idle_s += idle
                    self.w.stats.wait_s += idle     # node 0's hop wait: nothing to send, waiting for the pipeline
        except Exception as exc:  # noqa: BLE001 - the driver must never die silently
            log.exception("the engine driver crashed")
            self._die(f"engine driver crashed: {type(exc).__name__}: {exc}")
        finally:
            self._fail_all(self.dead or "the engine stopped")

    def _pump_events(self, block: bool) -> bool:
        worked = False
        while True:
            try:
                ev = self._events.get(timeout=0.05) if (block and not worked) else self._events.get_nowait()
            except queue.Empty:
                return worked
            op = ev.get("op")
            if op == "_wake":
                worked = True
                continue
            worked = True
            if op == "_dead":
                self._fail_all(ev.get("message", "the pipeline died"))
                self.dead = self.dead or ev.get("message")
            elif op == "result":
                self._on_result(ev)
            elif op == "error":
                self._on_error(ev)
            else:
                log.debug("ignoring control %r from the token link", op)

    # ---- admission / cancellation

    def _admit(self) -> bool:
        did = False
        while self._free:
            with self._lock:
                if not self._admit_q:
                    break
                seq = self._admit_q.popleft()
            if seq.cancel.is_set():
                seq.out.put(("done", "cancelled"))
                continue
            did = True
            slot = self._free.pop(0)
            seq.slot, seq.t_admit = slot, time.monotonic()
            seq.tokens = list(seq.prompt)
            seq.window = AdaptiveWindow(seq.k_max, 1, self.cfg.spec.adaptive)
            self._seqs[seq.rid] = seq
            self._requests += 1
            if self._t_active is None:
                self._t_active = seq.t_admit
            try:
                self.w.seq_begin(seq.rid, slot, seq.settings, diag=seq.diag)
                with self.w.tx_lock:
                    if self.w.total > 1:
                        self.w._tx_out.send_control(
                            {"op": "seq_begin", "rid": seq.rid, "slot": slot, "diag": seq.diag,
                             "sampling": seq.settings.to_dict(), "prompt_tokens": len(seq.prompt)},
                            request_id=seq.rid)
                if seq.spec_mode != "none":
                    self.drafters[seq.spec_mode].open(slot, seq.prompt)
            except BigBrainPeerDead as exc:
                self._die(f"send failed: {exc}")
                return True
            except Exception as exc:  # noqa: BLE001
                self._end(seq, error=f"could not start the sequence: {exc!r}")
                continue
            seq.phase = "prefill" if len(seq.prompt) > 1 else "decode"
            seq.ready_since = time.monotonic()
        return did

    def _reap_cancelled(self) -> bool:
        did = False
        for seq in list(self._seqs.values()):
            if seq.cancel.is_set() and seq.phase != "done" and not seq.inflight \
                    and seq.prefill_inflight == 0:
                self._end(seq, finish="cancelled")
                did = True
        return did

    def _check_timeouts(self) -> None:
        if not self._steps:
            return
        now = time.monotonic()
        oldest = min(s.t_sent for s in self._steps.values())
        if now - oldest > self.cfg.result_timeout_s:
            self._die(f"no result for a step in flight for {now - oldest:.0f}s")

    # ---- building and dispatching steps

    def _decode_inflight(self) -> int:
        return sum(1 for s in self._steps.values() if s.meta.kind == "decode")

    def _prefill_inflight(self) -> int:
        return sum(1 for s in self._steps.values() if s.meta.kind == "prefill")

    def _send_next(self) -> bool:
        """Dispatch at most one step: alternate decode / prefill so neither starves."""
        order = ("decode", "prefill") if self._turn_decode else ("prefill", "decode")
        for kind in order:
            meta = self._build_decode() if kind == "decode" else self._build_prefill()
            if meta is not None:
                self._turn_decode = (kind != "decode")
                self._dispatch(meta)
                return True
        return False

    def _new_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    def _build_decode(self) -> Optional[StepMeta]:
        n_dec = self._decode_inflight()
        if n_dec >= self.inflight_cap:
            return None
        ready = [s for s in self._seqs.values()
                 if s.phase == "decode" and not s.inflight and not s.cancel.is_set()]
        if not ready:
            return None
        ready.sort(key=lambda s: s.ready_since)
        n_active = sum(1 for s in self._seqs.values() if s.phase == "decode")
        cap = self.cfg.mb_size or max(1, -(-n_active // self.inflight_cap))
        chosen = sorted(ready[:cap], key=lambda s: s.slot)
        # (row order within a step is arbitrary; ordering by slot keeps exllamav3's per-slot-tuple
        # block-table caches to subsets instead of every permutation)
        # drafts
        want: Dict[str, List[DraftRequest]] = {}
        for s in chosen:
            remaining = s.max_new - s.produced
            k = max(0, min(s.window.k, remaining - 1)) if s.window else 0
            if s.spec_mode != "none" and s.greedy and k > 0:
                want.setdefault(s.spec_mode, []).append(DraftRequest(s.slot, s.tokens, k))
        drafts: Dict[int, List[int]] = {}
        for mode, reqs in want.items():
            try:
                got = self.drafters[mode].propose(reqs)
            except Exception:  # noqa: BLE001 - a failing drafter degrades to plain decode
                log.exception("the %s drafter failed; these sequences run without drafts from now on", mode)
                got = {}
                failed = {r.slot for r in reqs}
                for s in chosen:
                    if s.slot in failed:
                        s.spec_mode = "none"      # its draft state may be half-advanced: do not trust it again
            for r in reqs:
                drafts[r.slot] = [int(t) for t in (got.get(r.slot) or [])][:r.k]
        rows = []
        for s in chosen:
            ids = [s.tokens[-1]] + drafts.get(s.slot, [])
            rows.append(StepRow(s.rid, s.slot, len(s.tokens) - 1, len(ids), ids))
        T = max(r.n for r in rows)
        meta = StepMeta(self._new_step_id(), "decode", T, rows)
        meta.packed = meta.ragged
        for s in chosen:
            s.inflight = True
        return meta

    def _build_prefill(self) -> Optional[StepMeta]:
        if self._prefill_inflight() >= self.cfg.prefill_window:
            return None
        for s in sorted(self._seqs.values(), key=lambda q: q.t_admit):
            if s.phase != "prefill" or s.cancel.is_set():
                continue
            target = len(s.prompt) - 1
            start = s.prefill_sent
            if start >= target:
                continue      # every chunk is out; the sequence decodes once the last ack is in
            end = min(start + self.cfg.prefill_chunk, target)
            row = StepRow(s.rid, s.slot, start, end - start, s.prompt[start:end])
            meta = StepMeta(self._new_step_id(), "prefill", end - start, [row])
            s.prefill_sent = end
            s.prefill_inflight += 1
            return meta
        return None

    def _dispatch(self, meta: StepMeta) -> None:
        now = time.monotonic()
        rids = [r.rid for r in meta.rows]
        self._steps[meta.step] = _Step(meta, now, rids)
        n_dec = self._decode_inflight()
        self._inflight_sum += n_dec
        self._inflight_n += 1
        self._inflight_max = max(self._inflight_max, n_dec)
        try:
            res = self.w.head_step(meta)
        except BigBrainPeerDead as exc:
            self._die(f"send failed: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - this step's rows fail, the pipeline lives
            log.exception("node 0 failed step %d", meta.step)
            self._steps.pop(meta.step, None)
            for rid in rids:
                seq = self._seqs.get(rid)
                if seq is not None:
                    seq.inflight = False
                    seq.prefill_inflight = max(0, seq.prefill_inflight - (1 if meta.kind == "prefill" else 0))
                    self._end(seq, error=f"node 0: {exc!r}")
            return
        if meta.kind == "prefill":
            self._prefill_tokens += meta.rows[0].n
            seq = self._seqs.get(meta.rows[0].rid)
            if seq is not None and seq.spec_mode != "none":
                try:
                    # the drafter pre-feeds the prompt a chunk at a time while the pipeline works on it
                    self.drafters[seq.spec_mode].warm(seq.slot, seq.prompt, seq.prefill_sent)
                except Exception:  # noqa: BLE001
                    log.exception("the %s drafter could not warm up; it will catch up on its first "
                                  "proposal", seq.spec_mode)
        if res is not None:   # a one-node pipeline decides locally
            self._events.put(res)

    # ---- results

    def _on_result(self, res: Dict) -> None:
        step = self._steps.pop(int(res.get("step", -1)), None)
        if step is None:
            return   # a step whose sequences already failed
        now = time.monotonic()
        self._step_lat.append(now - step.t_sent)
        meta = step.meta
        self._steps_done[meta.kind] += 1
        if meta.kind == "prefill":
            for rid in step.rids:
                seq = self._seqs.get(rid)
                if seq is not None:
                    seq.prefill_inflight = max(0, seq.prefill_inflight - 1)
                    if seq.phase == "prefill" and seq.prefill_inflight == 0 \
                            and seq.prefill_sent >= len(seq.prompt) - 1:
                        seq.phase = "decode"        # the whole prompt is in every node's cache
                        seq.ready_since = now
            return
        self._rows_per_step_sum += len(meta.rows)
        self._t_per_step_sum += meta.T
        by_rid = {int(r["rid"]): r for r in res.get("rows") or []}
        for row in meta.rows:
            seq = self._seqs.get(row.rid)
            if seq is None:
                continue
            seq.inflight = False
            got = by_rid.get(row.rid)
            if got is None or not got.get("toks"):
                self._end(seq, error="the last node returned no tokens for the request")
                continue
            self._commit(seq, row, got, now)

    def _commit(self, seq: _Seq, row: StepRow, got: Dict, now: float) -> None:
        toks = [int(t) for t in got["toks"]]
        proposed, accepted = row.n - 1, len(toks) - 1
        if accepted > proposed:
            self._end(seq, error=f"the last node accepted {accepted} of {proposed} drafts")
            return
        self.spec_stats.add_row(proposed, accepted, len(toks))
        seq.proposed += proposed
        seq.accepted += accepted
        seq.rounds += 1 if proposed else 0
        if seq.window is not None:
            seq.window.update(proposed, accepted)
        room = seq.max_new - seq.produced
        out: List[int] = []
        finish = None
        for i, t in enumerate(toks):
            out.append(t)
            if seq.diag and got.get("margins") and i < len(got["margins"]):
                seq.margins.append(got["margins"][i])
            if t in seq.stop_ids:
                finish = "stop"
                break
            if len(out) >= room:
                finish = "length"
                break
        seq.tokens.extend(out)
        seq.produced += len(out)
        self._decode_tokens += len(out)
        if seq.t_first is None:
            seq.t_first = now
            self._first_lat.append(now - seq.t_submit)
            self._prefill_rates.append((len(seq.prompt), max(now - seq.t_admit, 1e-9)))
        seq.out.put(("tok", out))
        if finish is not None:
            self._end(seq, finish=finish)
        else:
            seq.ready_since = now

    def _on_error(self, ev: Dict) -> None:
        msg = str(ev.get("message", "remote error"))
        step = self._steps.pop(int(ev.get("step", -1)), None) if "step" in ev else None
        rids = list(ev.get("rids") or [])
        if step is not None:
            rids = list(dict.fromkeys(rids + step.rids))
            for rid in step.rids:
                seq = self._seqs.get(rid)
                if seq is not None:
                    seq.inflight = False
                    if step.meta.kind == "prefill":
                        seq.prefill_inflight = max(0, seq.prefill_inflight - 1)
        for rid in rids:
            seq = self._seqs.get(int(rid))
            if seq is not None:
                self._end(seq, error=msg)

    # ---- ending sequences

    def _end(self, seq: _Seq, finish: Optional[str] = None, error: Optional[str] = None) -> None:
        if seq.phase == "done":
            return
        seq.phase = "done"
        now = time.monotonic()
        seq.info.update({"proposed": seq.proposed, "accepted": seq.accepted, "rounds": seq.rounds,
                         "tokens": seq.produced, "slot": seq.slot,
                         "prompt_tokens": len(seq.prompt),
                         "first_token_s": (seq.t_first - seq.t_submit) if seq.t_first else None,
                         "total_s": now - seq.t_submit, "spec_mode": seq.spec_mode})
        if seq.diag:
            seq.info["margins"] = list(seq.margins)
        if seq.t_first is not None and seq.produced > 1:
            seq.info["decode_tok_s"] = (seq.produced - 1) / max(now - seq.t_first, 1e-9)
            self._seq_rates.append(seq.info["decode_tok_s"])
        if error is not None:
            self._failed += 1
            seq.out.put(("error", error))
        else:
            seq.out.put(("done", finish or "length"))
        self._seqs.pop(seq.rid, None)
        if seq.slot >= 0:
            for d in self.drafters.values():
                try:
                    d.close(seq.slot)
                except Exception:  # noqa: BLE001
                    log.exception("closing a drafter slot failed")
            try:
                self.w.seq_end(seq.rid, seq.slot)
                if self.w.total > 1 and not self.dead:
                    with self.w.tx_lock:
                        self.w._tx_out.send_control({"op": "seq_end", "rid": seq.rid, "slot": seq.slot},
                                                    request_id=seq.rid)
            except BigBrainPeerDead as exc:
                self._die(f"send failed: {exc}")
            self._free.append(seq.slot)
            self._free.sort()
        if not self._seqs and self._t_active is not None:
            self._active_s += now - self._t_active
            self._t_active = None

    def _fail_all(self, why: str) -> None:
        for seq in list(self._seqs.values()):
            if seq.phase != "done":
                seq.phase = "done"
                seq.out.put(("error", why))
        self._seqs.clear()
        with self._lock:
            queued, self._admit_q = list(self._admit_q), collections.deque()
        for seq in queued:
            seq.out.put(("error", why))
        self._steps.clear()

    # ------------------------------------------------------------------ health

    def health(self) -> Dict:
        return {"ok": self.ready.is_set() and not self.dead, "slots": self.slots,
                "active": len(self._seqs), "queued": len(self._admit_q),
                "dead": self.dead, "spec": self.cfg.spec.mode, "inflight_cap": self.inflight_cap}
