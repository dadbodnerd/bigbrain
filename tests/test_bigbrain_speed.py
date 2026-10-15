# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""BigBrain multi-sequence pipelining + speculative decoding -- no GPU, no exllamav3.

Layers:
  * pure logic (spec.py, paged.py, the wire step frame): no torch for most;
  * the whole pipeline over real loopback links, driving the REAL SliceWorker /
    PipelineEngine / DraftModelDrafter code on fake per-node layers whose "model"
    is a deterministic function (bigbrain_speed_fakes.py). The central claim is
    checked here: every sequence's GREEDY output is identical to the plain
    single-sequence run, for N sequences x {no speculation, n-gram, draft
    model}, on 1..3 nodes, with the per-node cache rollback exercised.
"""
from __future__ import annotations

import threading
import time
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from bigbrain_speed_fakes import (H, KINDS_FULL, KINDS_MIXED, V, FakeCache, FakeModel,
                                  FakeSWAState, fake_advance_recurrent_states, oracle_greedy)
from tools.bigbrain.draft import ExllamaDraftForward, check_same_tokenizer
from tools.bigbrain.engine import (BigBrainRemoteError, EngineConfig, EngineDead,
                                   PipelineEngine)
from tools.bigbrain.paged import (PAGE_SIZE, BigBrainStateError, NodeStats, PagedSlots,
                                  StepMeta, StepRow, pack_rows, unpack_rows)
from tools.bigbrain.pipe import BigBrainLink
from tools.bigbrain.spec import (AdaptiveWindow, DraftForward, DraftModelDrafter, DraftRequest,
                                 NgramDrafter, SpecConfig, SpecStats, greedy_accept, make_drafter)
from tools.bigbrain.worker import (SamplingSettings, SliceWorker, _select_slice, make_cache)


def _torch():
    return pytest.importorskip("torch")


# ==========================================================================
# acceptance
# ==========================================================================

def test_greedy_accept_commits_the_matching_prefix_plus_the_bonus_token():
    # inputs: last committed token 7, drafts 1 2 3; argmax after each input
    assert greedy_accept([7, 1, 2, 3], [1, 2, 3, 9], 4) == [1, 2, 3, 9]      # all accepted + bonus
    assert greedy_accept([7, 1, 2, 3], [1, 2, 8, 9], 4) == [1, 2, 8]         # d3 rejected: g2 is the bonus
    assert greedy_accept([7, 1, 2, 3], [5, 2, 3, 9], 4) == [5]               # d1 rejected: just g0
    assert greedy_accept([7], [4], 1) == [4]                                  # a plain decode row
    # only the first n rows count: padded rows are ignored
    assert greedy_accept([7, 1, 0, 0], [1, 6, 0, 0], 2) == [1, 6]
    with pytest.raises(ValueError):
        greedy_accept([], [], 0)


def test_adaptive_window_grows_on_full_acceptance_and_shrinks_on_poor():
    w = AdaptiveWindow(kmax=6, kmin=1)
    assert w.k == 6
    assert w.update(proposed=6, accepted=0) == 5
    assert w.update(proposed=5, accepted=1) == 4
    assert w.update(proposed=4, accepted=4) == 5     # full acceptance grows
    assert w.update(proposed=5, accepted=3) == 5     # decent: hold
    assert w.update(proposed=0, accepted=0) == 5     # a plain row changes nothing
    for _ in range(10):
        w.update(proposed=w.k, accepted=0)
    assert w.k == 1                                   # never below kmin
    assert AdaptiveWindow(4, enabled=False).update(4, 0) == 4


def test_spec_config_validates():
    with pytest.raises(ValueError):
        SpecConfig(mode="medusa")
    with pytest.raises(ValueError):
        SpecConfig(mode="ngram", k=0)
    with pytest.raises(ValueError):
        SpecConfig(mode="ngram", k=99)
    with pytest.raises(ValueError):
        SpecConfig(ngram_min=5, ngram_max=3)
    assert not SpecConfig().enabled and SpecConfig(mode="ngram").enabled


def test_spec_stats_acceptance_rate_and_histogram():
    s = SpecStats()
    s.add_row(proposed=4, accepted=3, committed=4)
    s.add_row(proposed=4, accepted=0, committed=1)
    s.add_row(proposed=0, accepted=0, committed=1)
    d = s.to_dict()
    assert (d["rounds"], d["proposed"], d["accepted"], d["plain_rows"]) == (2, 8, 3, 1)
    assert d["acceptance_rate"] == pytest.approx(3 / 8)
    assert d["tokens_per_row"] == pytest.approx(6 / 3)
    assert d["accepted_histogram"] == {"0": 1, "3": 1}
    s.reset()
    assert s.acceptance_rate is None


# ==========================================================================
# n-gram / prompt lookup
# ==========================================================================

def test_ngram_proposes_the_continuation_of_the_latest_earlier_match():
    d = NgramDrafter(nmin=2, nmax=4)
    ctx = [5, 6, 7, 8, 9, 1, 2, 5, 6, 7]        # the suffix 5 6 7 occurred at the start
    d.open(0, ctx[:-1])
    got = d.propose([DraftRequest(0, ctx, 4)])
    assert got[0] == [8, 9, 1, 2]
    # k clamps the proposal
    assert d.propose([DraftRequest(0, ctx, 2)])[0] == [8, 9]
    assert d.propose([DraftRequest(0, ctx, 0)])[0] == []


def test_ngram_prefers_the_longest_suffix_then_the_most_recent_occurrence():
    d = NgramDrafter(nmin=2, nmax=3)
    #            0  1  2  3  4  5  6  7  8  9 10 11
    ctx = [1, 2, 3, 9, 9, 2, 3, 4, 4, 1, 2, 3]
    # suffix (1,2,3) occurred at 0..2 -> continuation 9,9 ; suffix (2,3) latest earlier at 5..6 -> 4,4
    d.open(0, [])
    assert d.propose([DraftRequest(0, ctx, 2)])[0] == [9, 9]          # the 3-gram wins
    d2 = NgramDrafter(nmin=2, nmax=2)
    d2.open(0, [])
    assert d2.propose([DraftRequest(0, ctx, 2)])[0] == [4, 4]         # only 2-grams: the latest is at 5..6


def test_ngram_no_match_no_draft_and_index_grows_incrementally():
    d = NgramDrafter(nmin=2, nmax=3)
    d.open(3, [1, 2, 3, 4])
    assert d.propose([DraftRequest(3, [1, 2, 3, 4], 4)])[3] == []
    ctx = [1, 2, 3, 4, 1, 2, 3]
    assert d.propose([DraftRequest(3, ctx, 4)])[3] == [4, 1, 2, 3]
    ctx2 = ctx + [4]
    assert d.propose([DraftRequest(3, ctx2, 3)])[3] == [1, 2, 3]
    d.close(3)
    assert 3 not in d._idx


def test_ngram_matches_across_the_prompt_generation_boundary_only_on_committed_text():
    """Drafts are never indexed: a rejected draft cannot poison later proposals."""
    d = NgramDrafter(nmin=2, nmax=3)
    committed = [1, 2, 3, 1, 2]
    d.open(0, committed)
    first = d.propose([DraftRequest(0, committed, 3)])[0]
    assert first == [3, 1, 2]
    # the target rejected it and committed 9 instead
    committed2 = committed + [9]
    assert d.propose([DraftRequest(0, committed2, 3)])[0] == []


# ==========================================================================
# the draft-model drafter (logic, on a pure-python backend)
# ==========================================================================

class ListBackend(DraftForward):
    """A draft 'model' that is a pure function of the fed tokens, with a real
    per-slot cache: rows feeding at a position behind the cache roll it back;
    a position ahead of it is an error -- the contract draft.py implements."""

    def __init__(self, fn):
        self.fn = fn
        self.cache = {}
        self.calls = []        # (kind, slot, start, ntokens)
        self.batches = []      # rows per forward() call
        self.rollbacks = 0

    def _feed(self, slot, pos, toks):
        c = self.cache.setdefault(slot, [])
        assert pos <= len(c), f"gap: slot {slot} holds {len(c)}, fed at {pos}"
        if pos < len(c):
            self.rollbacks += 1
            del c[pos:]
        c.extend(toks)
        return self.fn(c)

    def forward(self, rows):
        out = []
        self.batches.append(len(rows))
        for slot, pos, toks in rows:
            self.calls.append(("fwd", slot, pos, len(toks)))
            out.append(self._feed(slot, pos, list(toks)))
        return out

    def prefill(self, slot, start_pos, tokens):
        self.calls.append(("pre", slot, start_pos, len(tokens)))
        self._feed(slot, start_pos, list(tokens))

    def close(self, slot):
        self.cache.pop(slot, None)

    def close_all(self):
        self.cache.clear()


def _f(tokens):               # next token = f(history): depends on the last two tokens
    return (3 * tokens[-1] + (tokens[-2] if len(tokens) > 1 else 1)) % 10


def _continue(tokens, k):
    t, out = list(tokens), []
    for _ in range(k):
        n = _f(t)
        out.append(n)
        t.append(n)
    return out


def test_draft_drafter_proposes_the_models_own_greedy_continuation():
    be = ListBackend(_f)
    dr = DraftModelDrafter(be, prefill_chunk=3)
    prompt = [4, 5, 6, 7, 8]
    dr.open(0, prompt)
    got = dr.propose([DraftRequest(0, prompt, 4)])[0]
    assert got == _continue(prompt, 4)
    # the first round fed the prompt (chunked prefill) and left <= 2 tokens for the batched call
    assert [c for c in be.calls if c[0] == "pre"] == [("pre", 0, 0, 3)]
    assert be.calls[1] == ("fwd", 0, 3, 2)
    # k-1 single-token extension steps followed
    assert [c[3] for c in be.calls if c[0] == "fwd"] == [2, 1, 1, 1]


def test_draft_drafter_rolls_back_to_the_committed_text_every_round():
    be = ListBackend(_f)
    dr = DraftModelDrafter(be, prefill_chunk=64)
    committed = [4, 5, 6]
    dr.open(0, committed)
    for rnd in range(15):
        props = dr.propose([DraftRequest(0, committed, 4)])[0]
        assert props == _continue(committed, 4), f"round {rnd}"
        m = rnd % 5                                    # the target accepts 0..4 of the 4 drafts
        correct = _continue(committed + props[:m], 1)[0]
        bonus = correct if m == 4 else (props[m] + 1) % 10   # on a rejection the target's token differs
        committed = committed + props[:m] + [bonus]
        # what the draft model holds is committed text followed by at most its own newest drafts
        held = be.cache[0]
        agree = 0
        while agree < min(len(held), len(committed)) and held[agree] == committed[agree]:
            agree += 1
        assert agree >= len(committed) - 1 - (1 if m == 4 else 0) or agree >= len(committed) - 2
    assert be.rollbacks > 0       # rejections really rolled the slot back
    # after the first round the catch-up is never more than 2 tokens
    widths = [c[3] for c in be.calls if c[0] == "fwd" and c[2] > 0 and c[3] > 1]
    assert widths and max(widths) <= 2


def test_draft_drafter_batches_every_sequence_of_a_round():
    be = ListBackend(_f)
    dr = DraftModelDrafter(be)
    a, b = [1, 2, 3], [9, 8]
    dr.open(0, a)
    dr.open(1, b)
    got = dr.propose([DraftRequest(0, a, 3), DraftRequest(1, b, 3)])
    assert got[0] == _continue(a, 3) and got[1] == _continue(b, 3)
    assert be.batches == [2, 2, 2]               # k=3: one catch-up forward + 2 extensions, both rows each
    assert dr.forwards == 3


def test_draft_drafter_warm_prefills_a_prompt_one_chunk_at_a_time():
    be = ListBackend(_f)
    dr = DraftModelDrafter(be, prefill_chunk=4)
    prompt = list(range(10)) + [1]
    dr.open(0, prompt)
    done = [dr.warm(0, prompt, upto=10)]
    done.append(dr.warm(0, prompt, upto=10))
    done.append(dr.warm(0, prompt, upto=10))
    assert done == [False, False, True]
    assert [c[3] for c in be.calls] == [4, 4, 2]
    # the first proposal then only has the last prompt token to feed
    got = dr.propose([DraftRequest(0, prompt, 2)])[0]
    assert got == _continue(prompt, 2)
    assert be.calls[3] == ("fwd", 0, 10, 1)


def test_make_drafter():
    assert isinstance(make_drafter(SpecConfig(mode="ngram")), NgramDrafter)
    with pytest.raises(ValueError):
        make_drafter(SpecConfig(mode="draft"))
    assert make_drafter(SpecConfig(mode="draft"), ListBackend(_f)).name == "draft"
    assert make_drafter(SpecConfig()).propose([DraftRequest(0, [1], 3)]) == {0: []}


# ==========================================================================
# paged slots: layout, rollback, errors
# ==========================================================================

def test_paged_slots_layout_and_cache_size():
    _torch()                                      # PagedSlots builds torch tensors
    p = PagedSlots(n_slots=3, ctx=512)
    assert p.slot_pages == 3                      # 2 context pages + 1 slack
    assert p.cache_tokens == 3 * 3 * 256 == PagedSlots.tokens_for(3, 512)
    p.begin(1, rid=7)
    bt, seqlens, states = p.prepare([(1, 7, 0, 5)])
    assert states is None
    assert bt.tolist() == [[3, 4, 5]]             # slot 1 owns pages 3..5
    assert seqlens.tolist() == [0]
    p.commit([(1, 7, 0, 5)])
    assert p.written(1) == 5


def test_paged_slots_rolls_back_to_the_step_position():
    _torch()                                      # PagedSlots builds torch tensors
    p = PagedSlots(2, 256)
    p.begin(0, 1)
    p.prepare([(0, 1, 0, 6)])
    p.commit([(0, 1, 0, 6)])                      # wrote 6 positions (a verify of 1+5)
    bt, seqlens, _ = p.prepare([(0, 1, 3, 1)])    # accepted 3 of them: the step starts at 3
    assert seqlens.tolist() == [3]
    assert (p.rollbacks, p.rolled_tokens) == (1, 3)
    p.commit([(0, 1, 3, 1)])
    assert p.written(0) == 4
    p.prepare([(0, 1, 4, 1)])                     # a normal append: no rollback
    assert p.rollbacks == 1


def test_paged_slots_refuses_gaps_foreign_slots_overruns_and_dead_slots():
    _torch()                                      # PagedSlots builds torch tensors
    p = PagedSlots(2, 256)
    p.begin(0, 1)
    with pytest.raises(BigBrainStateError, match="only written 0"):
        p.prepare([(0, 1, 5, 1)])                                 # a step was lost
    with pytest.raises(BigBrainStateError, match="held by request 1"):
        p.prepare([(0, 2, 0, 1)])                                 # another request's step
    with pytest.raises(BigBrainStateError, match="overrun"):
        p.prepare([(0, 1, 0, 513)])                               # past the slack page
    with pytest.raises(BigBrainStateError, match="outside"):
        p.begin(2, 1)
    p.invalidate(0)
    with pytest.raises(BigBrainStateError, match="dead"):
        p.prepare([(0, 1, 0, 1)])
    p.end(0)
    with pytest.raises(BigBrainStateError, match="held by request None"):
        p.prepare([(0, 1, 0, 1)])


def test_paged_slots_the_slack_page_keeps_padding_inside_the_slot():
    _torch()                                      # PagedSlots builds torch tensors
    p = PagedSlots(2, 256)           # 1 context page + 1 slack page
    p.begin(0, 1)
    p.begin(1, 2)
    # a row at the very end of the context padded by 7 more positions stays in slot 0's pages
    p.slots[0].written = 255
    bt, seqlens, _ = p.prepare([(0, 1, 255, 8)])
    assert set(bt.flatten().tolist()) <= {0, 1}                    # never slot 1's pages 2, 3


def test_paged_slots_recurrent_states_are_rewound_and_checked():
    _torch()
    made = []

    class St:
        def __init__(self, slot):
            self.slot, self.position, self.rewound = slot, 0, []

        def rewind(self, n):
            self.position -= n
            self.rewound.append(n)

    def new(slot):
        s = St(slot)
        made.append(s)
        return s

    p = PagedSlots(2, 256, recurrent=True, new_state=new, release_state=lambda s: made.remove(s))
    p.begin(0, 1)
    st = made[0]
    _, _, states = p.prepare([(0, 1, 0, 4)])
    assert states == [st]
    st.position += 4                              # the forward advanced the state by 4
    p.commit([(0, 1, 0, 4)])
    p.prepare([(0, 1, 2, 1)])                     # accepted 2 of 4
    assert st.rewound == [2] and st.position == 2
    p.commit([(0, 1, 2, 1)])                      # written 3
    st.position = 9                               # a state that drifted from the cache
    with pytest.raises(BigBrainStateError, match="recurrent state"):
        p.prepare([(0, 1, 3, 1)])                 # an append at 3, but the state says 9

    p.end(0)
    assert made == []                             # released

    class Stateless:
        def __init__(self, slot):
            self.slot, self.position = slot, 0
    q = PagedSlots(1, 256, recurrent=True, new_state=Stateless)
    q.begin(0, 1)
    q.slots[0].state.position = 4
    q.slots[0].written = 4
    with pytest.raises(BigBrainStateError, match="cannot roll back"):
        q.prepare([(0, 1, 2, 1)])


def test_pack_and_unpack_rows_roundtrip():
    t = _torch()
    x = t.arange(3 * 4 * 2, dtype=t.float32).view(3, 4, 2)
    ns = [4, 1, 2]
    packed = pack_rows(x, ns)
    assert packed.shape == (1, 7, 2)
    back = unpack_rows(packed, ns, 4)
    assert back.shape == x.shape
    for i, n in enumerate(ns):
        assert t.equal(back[i, :n], x[i, :n])
        assert float(back[i, n:].abs().sum()) == 0.0          # padding is zeros


def test_step_meta_roundtrip_and_validation():
    m = StepMeta(5, "decode", 3, [StepRow(1, 0, 10, 3, [1, 2, 3]), StepRow(2, 1, 4, 1, [9])], packed=True)
    back = StepMeta.from_dict(m.to_dict())
    assert back.to_dict() == m.to_dict() and back.ragged and back.ns == [3, 1]
    for bad in (dict(m.to_dict(), kind="verify"),
                dict(m.to_dict(), rows=[]),
                dict(m.to_dict(), T=0),
                dict(m.to_dict(), rows=[{"rid": 1, "slot": 0, "pos": 0, "n": 4, "ids": [1, 2, 3, 4]}]),
                dict(m.to_dict(), rows=[{"rid": 1, "slot": 0, "pos": 0, "n": 2, "ids": [1]}]),
                dict(m.to_dict(), rows=[m.to_dict()["rows"][0], m.to_dict()["rows"][0]]),
                {"op": "step"}):
        with pytest.raises(BigBrainStateError):
            StepMeta.from_dict(bad)


def test_node_stats_snapshot_and_reset():
    s = NodeStats(2)
    s.busy_s, s.wait_s, s.send_s = 0.5, 0.25, 0.05
    s.steps, s.rows = 3, 5
    time.sleep(0.01)
    snap = s.snapshot({"x": 1})
    assert snap["rank"] == 2 and snap["steps"] == 3 and snap["x"] == 1
    assert 0 < snap["busy_pct"] and snap["busy_s"] == 0.5
    s.reset()
    assert s.snapshot()["steps"] == 0 and s.rank == 2


def test_check_same_tokenizer(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
        (d / "config.json").write_text('{"vocab_size": 201088}')
        (d / "tokenizer_config.json").write_text('{"x": 1}')
    for d in (a, b):
        (d / "tokenizer.json").write_text('{"model": {"vocab": {"a": 0}}}')
    (c / "tokenizer.json").write_text('{"model": {"vocab": {"b": 0}}}')
    assert check_same_tokenizer(str(a), str(b))["same"]
    r = check_same_tokenizer(str(a), str(c))
    assert not r["same"] and any("tokenizer.json" in p for p in r["problems"])
    (b / "config.json").write_text('{"vocab_size": 5}')
    assert any("vocab_size" in p for p in check_same_tokenizer(str(a), str(b))["problems"])
    assert not check_same_tokenizer(str(tmp_path / "none"), str(a))["same"]


def test_a_history_mode_recurrent_state_is_settled_even_when_nothing_is_rolled_back():
    """GatedDeltaNet after a forward that recorded per-token history (last_history > 0) must get
    rewind(0) when the whole window was accepted, or its conv window stays in history layout --
    exllamav3's generator does the same (batch_states[j].rewind(0)). A state with no history needs
    no call; an in-place sliding-window state (last_history 0) is only rewound when ahead."""
    _torch()
    calls = []

    class HistState:
        def __init__(self, slot):
            self.slot, self.position, self.last_history = slot, 0, 0

        def rewind(self, n):
            calls.append(n)
            self.position -= n
            self.last_history = 0

    p = PagedSlots(1, 256, recurrent=True, new_state=HistState)
    p.begin(0, 1)
    st = p.state(0)
    p.prepare([(0, 1, 0, 5)])
    st.position, st.last_history = 5, 4           # a verify forward of 5 tokens recorded history
    p.commit([(0, 1, 0, 5)])
    p.prepare([(0, 1, 5, 1)])                     # ALL accepted: the next step starts at 5 = written
    assert calls == [0] and st.last_history == 0
    st.position, st.last_history = 6, 0           # a plain 1-token step: no history
    p.commit([(0, 1, 5, 1)])
    p.prepare([(0, 1, 6, 1)])
    assert calls == [0]                           # nothing to settle
    st.position, st.last_history = 10, 3
    p.slots[0].written = 10
    p.prepare([(0, 1, 8, 1)])                     # 2 rejected
    assert calls == [0, 2] and st.position == 8
