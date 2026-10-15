# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Regression tests for the defects an independent review found in the first version of the batched
engine (2026-10-13), each reproduced against the real SliceWorker / PipelineEngine on the fake model:

  1. history-mode recurrent state only for multi-token verify windows
  2. a failed prefill chunk must not take healthy sequences down with it
  4. an activation without a 'step' frame must not run the single-sequence path
  5. the control+activation pair is one queued item; a lone 'step' frame is reported
  6. decode rows are ordered by slot
  7. only the sequence's own drafter is opened; n-gram indexes during prefill
  8. a stale stats answer, shutdown order
No GPU, no exllamav3. Needs torch (CPU).
"""
from __future__ import annotations

import threading

import pytest

from bigbrain_speed_fakes import H, KINDS_MIXED, FakeSWA
from bigbrain_speed_harness import PROMPTS, Pipe, oracle, pipes  # noqa: F401
from bigbrain.engine import BigBrainRemoteError
from bigbrain.paged import StepMeta, StepRow
from bigbrain.pipe import Message
from bigbrain.spec import DraftRequest, NgramDrafter
from bigbrain.worker import SamplingSettings, _Tx


def _torch():
    return pytest.importorskip("torch")


# ---- 1. history only for multi-token windows -------------------------------------------------

def test_state_history_is_only_recorded_for_multi_token_verify_windows(pipes):
    """GatedDeltaNet's conv ring is refreshed from history only by rewind(); a ONE-token forward in
    history mode leaves it stale (conv1d.py writes the ring's tail, last_history stays 0). The fake
    sliding-window layer raises if that ever happens, on the target nodes and the draft model alike."""
    _torch()
    FakeSWA.seen.clear()
    p = pipes([(0, 2), (2, 4)], slots=3, max_draft=4, spec="draft", k=4, draft=True, perturb=0.3)
    outs = p.run(PROMPTS[:3], 20)
    for prompt, toks in zip(PROMPTS[:3], outs):
        assert toks == oracle(KINDS_MIXED, prompt, 20)
    assert (1, True) not in set(FakeSWA.seen)
    assert any(h for _, h in FakeSWA.seen), "no verify window ever recorded history"
    assert all(1 < w <= 5 for w, h in FakeSWA.seen if h)
    # a mix with plain one-token rows (sampled, no match, last token) is just as clean
    p2 = pipes([(0, 4)], slots=2, max_draft=4, spec="ngram", k=4)
    p2.run([PROMPTS[1], PROMPTS[3]], 12, settings=[SamplingSettings(temperature=0.9, seed=3), None])


# ---- 2. a failed prefill kills only its own sequence ------------------------------------------

def test_a_failed_prefill_chunk_kills_only_its_own_sequence(pipes):
    """A sequence decodes only after its prefill was ACKNOWLEDGED, so a prefill that fails on a downstream
    node can never take healthy sequences down (they would have shared its first decode step)."""
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=4, max_draft=0, spec="none", inflight=4, mb=4, prefill_chunk=3)
    armed = {"on": True}
    victim = PROMPTS[3]
    node1 = p.nodes[1]
    orig = node1.process_batch

    def flaky(meta, x):
        row = meta.rows[0]
        if meta.kind == "prefill" and armed["on"] and row.pos == 0 and list(row.ids) == victim[:3]:
            armed["on"] = False
            raise RuntimeError("injected prefill failure (e.g. CUDA out of memory)")
        return orig(meta, x)
    node1.process_batch = flaky
    results = {}

    def one(i):
        try:
            results[i] = [t for t, _ in p.engine.generate(PROMPTS[i], SamplingSettings(), 14)]
        except BigBrainRemoteError as exc:
            results[i] = exc
    ths = [threading.Thread(target=one, args=(i,)) for i in (0, 2, 3, 4)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
    assert isinstance(results[3], BigBrainRemoteError) and "injected prefill failure" in str(results[3])
    for i in (0, 2, 4):
        assert results[i] == oracle(KINDS_MIXED, PROMPTS[i], 14), i
    assert p.settle()


# ---- 4. an activation without a step frame --------------------------------------------------------

def test_an_activation_without_a_step_frame_is_dropped_while_batched_sequences_are_live(pipes):
    """It used to fall into the single-sequence path: slot 0's pages and recurrent state overwritten,
    and every later seq_begin refused."""
    t = _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=0, start=False)
    w = p.nodes[1]
    w.seq_begin(5, 0, SamplingSettings())
    before = w.paged.written(0)
    rogue = Message("activation", {"request_id": 7, "position": 0, "flags": 0}, array=t.zeros((1, 3, H)).numpy())
    w._on_activation(rogue)
    assert w.request_id is None and w.paged.written(0) == before and w.stats.errors == 1
    w.seq_begin(6, 1, SamplingSettings())          # later sequences are still accepted
    # a step frame that cannot be parsed makes the activation that follows it be skipped, not executed
    sent = []
    w._send_on = lambda c, rid=0: sent.append(c)
    bad = Message("control", {"request_id": 9},
                  control={"op": "step", "step": 9, "kind": "decode", "T": 2, "rows": [], "packed": False})
    w._on_control(bad)
    assert sent and sent[0]["op"] == "error" and sent[0]["step"] == 9
    w._on_activation(Message("activation", {"request_id": 9, "position": 0, "flags": 0},
                             array=t.zeros((1, 2, H)).numpy()))
    assert w.request_id is None and w.paged.written(0) == before


# ---- 5. the step pair ----------------------------------------------------------------------------

def test_a_second_step_frame_before_the_first_activation_reports_the_lost_step(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=0, start=False)
    w = p.nodes[1]
    w.seq_begin(5, 0, SamplingSettings())
    sent = []
    w._send_on = lambda c, rid=0: sent.append(c)
    good = {"op": "step", "step": 20, "kind": "decode", "T": 1, "packed": False,
            "rows": [{"rid": 5, "slot": 0, "pos": 0, "n": 1, "ids": [1]}]}
    w._on_control(Message("control", {"request_id": 20}, control=dict(good)))
    w._on_control(Message("control", {"request_id": 21}, control=dict(good, step=21)))
    assert sent and sent[0]["step"] == 20 and sent[0]["rids"] == [5] and "no activation" in sent[0]["message"]
    assert w._pending_step.step == 21


def test_tx_sends_a_pair_as_one_item_and_surfaces_a_failure():
    class Link:
        def __init__(self):
            self.sent, self.fail_on = [], None

        def send_control(self, obj, **kw):
            if obj == self.fail_on:
                raise OSError("boom")
            self.sent.append(("c", obj))

        def send_activation(self, arr, **kw):
            self.sent.append(("a", arr))

    link = Link()
    tx = _Tx(link, async_send=True)
    tx.send_control("A")
    tx.call_pair(("send_control", ("B",), {}), ("send_activation", ("B-act",), {}))
    tx.send_control("C")
    assert tx.drain(2.0)
    assert link.sent == [("c", "A"), ("c", "B"), ("a", "B-act"), ("c", "C")]
    link.fail_on = "D"
    tx.send_control("D")
    tx.drain(2.0)
    with pytest.raises(OSError):
        tx.send_control("E")                  # the failure is raised by the next call, never swallowed
    tx.close()
    link2 = Link()                            # direct mode sends the pair inline, in order
    _Tx(link2, async_send=False).call_pair(("send_control", ("X",), {}), ("send_activation", ("X-act",), {}))
    assert link2.sent == [("c", "X"), ("a", "X-act")]


# ---- 6. row order ------------------------------------------------------------------------------------

def test_decode_rows_are_ordered_by_slot(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=5, max_draft=0, spec="none", inflight=3, mb=3)
    metas = []
    orig = p.head._send_step_on
    p.head._send_step_on = lambda meta, out: (metas.append(meta), orig(meta, out))[1]
    p.run(PROMPTS[:5], 10)
    dec = [m for m in metas if m.kind == "decode"]
    assert dec and all([r.slot for r in m.rows] == sorted(r.slot for r in m.rows) for m in dec)


# ---- 7. drafters --------------------------------------------------------------------------------------

def test_only_the_sequences_own_drafter_is_opened_and_ngram_indexes_during_prefill(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=3, max_draft=4, spec="none", k=4, draft=True, prefill_chunk=4)
    ng = p.engine.drafters["ngram"]
    seen = {"open": [], "warm": []}
    o_open, o_warm = ng.open, ng.warm
    ng.open = lambda slot, prompt: (seen["open"].append(slot), o_open(slot, prompt))[1]
    ng.warm = lambda slot, tokens, upto: (seen["warm"].append(upto), o_warm(slot, tokens, upto))[1]
    p.run([PROMPTS[3]], 8, kw=[{"spec": "none"}])
    assert seen["open"] == [] and seen["warm"] == []                 # spec none: the n-gram drafter is untouched
    p.run([PROMPTS[3]], 8, kw=[{"spec": "ngram", "spec_k": 3}])
    assert len(seen["open"]) == 1 and seen["warm"], "n-gram must index the prompt while it is prefilled"
    assert max(seen["warm"]) == len(PROMPTS[3]) - 1


def test_ngram_warm_indexes_a_bounded_slice_per_call():
    d = NgramDrafter(nmin=2, nmax=3)
    toks = [(i * 7) % 11 for i in range(20000)]
    d.open(0, toks)
    assert len(d._idx[0].tokens) == 0                 # open() does not index the prompt any more
    done = [d.warm(0, toks, upto=19999) for _ in range(3)]
    assert done == [False, False, True]
    assert len(d._idx[0].tokens) == 19999
    # propose() indexes whatever remains and agrees with an index built in one go
    d2 = NgramDrafter(nmin=2, nmax=3)
    d2.open(0, [])
    assert d.propose([DraftRequest(0, toks, 4)])[0] == d2.propose([DraftRequest(0, toks, 4)])[0]


# ---- 8. stats ------------------------------------------------------------------------------------------

def test_a_late_stats_answer_does_not_answer_the_next_request(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=0)
    p.engine._boxes["stats_result"].put({"op": "stats_result", "nodes": [{"stale": True}]})
    nodes = p.engine.collect_node_stats()
    assert [n.get("rank") for n in nodes] == [0, 1] and not any(n.get("stale") for n in nodes)
