# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""The central claims of the BigBrain speed work, over the whole pipeline (real
loopback links, the real SliceWorker / PipelineEngine / drafters) on a fake model
whose "model" is a deterministic function:

  * with greedy sampling every sequence's output is IDENTICAL to the plain
    single-sequence run -- for N sequences x {no speculation, n-gram, draft
    model}, on 1..3 nodes, with the per-node cache rollback exercised;
  * several sequences really are in flight at once;
  * the batched protocol is negotiated, and the old single-sequence traffic
    still works against the new workers.
No GPU, no exllamav3. Needs torch (CPU).
"""
from __future__ import annotations

import threading
import time
from contextlib import closing

import pytest

from bigbrain_speed_fakes import KINDS_FULL, KINDS_MIXED, FakeBlock, FakeModel, FakeSWAState, V
from bigbrain_speed_harness import (BOUNDS, CTX, PROMPTS, Pipe, legacy_generate, make_node, oracle,
                                    pipes)  # noqa: F401  (pipes is a fixture)
from bigbrain.engine import BigBrainRemoteError, EngineConfig, EngineDead, PipelineEngine
from bigbrain.paged import StepMeta, StepRow
from bigbrain.spec import SpecConfig
from bigbrain.worker import SamplingSettings


def _torch():
    return pytest.importorskip("torch")


# ---- sanity: the fakes and the legacy path agree with the cache-free oracle

@pytest.mark.parametrize("kinds", [KINDS_MIXED, KINDS_FULL])
def test_the_plain_run_equals_the_cache_free_oracle(kinds):
    _torch()
    for prompt in PROMPTS:
        assert legacy_generate(kinds, prompt, 14) == oracle(kinds, prompt, 14), prompt


def test_the_fake_model_is_sensitive_to_a_missing_rollback():
    """If a node did NOT roll the cache and the sliding-window state back, outputs would
    change -- otherwise none of the parity tests below would prove anything."""
    t = _torch()
    w = make_node(KINDS_MIXED, 0, 4, 0, 1, slots=2, max_draft=4)
    prompt = [3, 1, 4, 1, 5, 9, 2]
    w.seq_begin(1, 0, SamplingSettings())

    def meta(step, kind, pos, ids):
        return StepMeta(step, kind, len(ids), [StepRow(1, 0, pos, len(ids), list(ids))])

    w.process_batch(meta(1, "prefill", 0, prompt[:6]), t.tensor([prompt[:6]]))
    good = w.process_batch(meta(2, "decode", 6, [prompt[6]]), t.tensor([[prompt[6]]]))
    # a verify window [last, 8, 8, 8] of wrong drafts; the next step re-feeds at 6 and the
    # node must roll the K/V cache AND the sliding-window state back to position 6
    w.process_batch(meta(3, "decode", 6, [prompt[6], 8, 8, 8]), t.tensor([[prompt[6], 8, 8, 8]]))
    again = w.process_batch(meta(4, "decode", 6, [prompt[6]]), t.tensor([[prompt[6]]]))
    assert t.equal(good, again)
    assert w.paged.rollbacks >= 1 and w.paged.rolled_tokens >= 4
    # a state that was NOT rewound (position left at 10) gives something else
    st = w.paged.state(0)
    st.position += 3
    w.paged.slots[0].written += 3
    bad = w.process_batch(meta(5, "decode", 9, [prompt[6]]), t.tensor([[prompt[6]]]))
    assert not t.equal(good, bad)


# ---- the central claim: N sequences x {none, ngram, draft} == plain, on 1..3 nodes

@pytest.mark.parametrize("bounds", BOUNDS)
@pytest.mark.parametrize("spec", ["none", "ngram", "draft"])
def test_n_sequences_greedy_output_is_identical_to_the_plain_run(pipes, bounds, spec):
    _torch()
    p = pipes(bounds, slots=4, max_draft=4, spec=spec, k=4, draft=(spec == "draft"))
    prompts = PROMPTS[:4]
    got = p.run(prompts, 24)
    for prompt, toks in zip(prompts, got):
        assert toks == oracle(KINDS_MIXED, prompt, 24), (spec, bounds, prompt)
        assert toks == legacy_generate(KINDS_MIXED, prompt, 24)      # the plain single-sequence run
    assert p.settle(), "slots/states leaked"
    rep = p.engine.report()
    assert rep["engine"]["decode_tokens"] == 4 * 24
    if spec != "none":
        s = rep["engine"]["spec"]
        assert s["proposed"] > 0, "speculation never proposed anything"
        assert 0 <= s["accepted"] <= s["proposed"]


@pytest.mark.parametrize("kinds", [KINDS_FULL])
def test_dense_attention_only_model_also_matches(pipes, kinds):
    _torch()
    p = pipes([(0, 2), (2, 4)], kinds=kinds, slots=3, max_draft=0, spec="ngram", k=4)
    got = p.run(PROMPTS[:3], 20)
    for prompt, toks in zip(PROMPTS[:3], got):
        assert toks == oracle(kinds, prompt, 20)


def test_speculation_rolls_back_every_node_and_still_matches(pipes):
    """A deliberately weak draft model: most drafts are rejected, so every node must roll
    its sequences back constantly -- and the output must not change."""
    _torch()
    FakeSWAState.rewinds.clear()
    p = pipes([(0, 1), (1, 3), (3, 4)], slots=3, max_draft=4, spec="draft", k=4, draft=True, perturb=0.9)
    prompts = PROMPTS[:3]
    got = p.run(prompts, 30)
    for prompt, toks in zip(prompts, got):
        assert toks == oracle(KINDS_MIXED, prompt, 30)
    rep = p.engine.report()
    s = rep["engine"]["spec"]
    assert s["proposed"] > 0 and s["accepted"] < s["proposed"], "the draft should be wrong sometimes"
    assert FakeSWAState.rewinds, "the sliding-window states were never rewound"
    for node in rep["nodes"]:
        assert node["rollbacks"] > 0 and node["rolled_tokens"] > 0, node["rank"]


def test_a_perfect_draft_is_fully_accepted_and_commits_k_plus_one_per_round(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=1, max_draft=4, spec="draft", k=4, draft=True, perturb=0.0)
    got = p.run([PROMPTS[0]], 25)[0]
    assert got == oracle(KINDS_MIXED, PROMPTS[0], 25)
    s = p.engine.report()["engine"]["spec"]
    assert s["acceptance_rate"] == pytest.approx(1.0)
    assert s["rounds"] + s["plain_rows"] <= 6          # 25 tokens in rounds of 5, not 25 rows
    assert s["tokens_per_row"] >= 4.0


def test_ngram_speculation_commits_more_than_one_token_per_row_on_repetitive_text(pipes):
    _torch()
    prompt = PROMPTS[3]
    p = pipes([(0, 4)], slots=1, max_draft=4, spec="ngram", k=4)
    got = p.run([prompt], 40)[0]
    assert got == oracle(KINDS_MIXED, prompt, 40)
    s = p.engine.report()["engine"]["spec"]
    assert s["accepted"] > 0 and s["tokens_per_row"] > 1.0


def test_each_request_picks_its_own_speculation_and_they_share_steps(pipes):
    """none / n-gram (k=2) / draft (k=4) / n-gram (k=6) concurrently: rows of different
    draft length share verify steps (shorter ones padded), packed on the wire."""
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=4, max_draft=6, spec="none", k=4, draft=True, perturb=0.4, inflight=2, mb=4)
    sent = []
    orig = p.head._send_step_on
    p.head._send_step_on = lambda meta, out: (sent.append((meta.kind, meta.T, meta.ns, meta.packed)), orig(meta, out))[1]
    modes = [{"spec": "none"}, {"spec": "ngram", "spec_k": 2}, {"spec": "draft", "spec_k": 4},
             {"spec": "ngram", "spec_k": 6}]
    prompts = [PROMPTS[3], PROMPTS[3], PROMPTS[0], PROMPTS[4]]
    outs = p.run(prompts, 28, kw=modes)
    for prompt, toks in zip(prompts, outs):
        assert toks == oracle(KINDS_MIXED, prompt, 28)
    assert any(kind == "decode" and packed and len(set(ns)) > 1 for kind, T, ns, packed in sent), \
        "no ragged step was ever sent"
    assert p.settle()


def test_a_sampled_sequence_rides_along_with_greedy_ones(pipes):
    """A sampled request never carries drafts, and gets exactly the tokens it would get alone."""
    _torch()
    s = SamplingSettings(temperature=1.2, top_k=6, seed=99)
    alone = legacy_generate(KINDS_MIXED, PROMPTS[0], 20, settings=s)
    p = pipes([(0, 2), (2, 4)], slots=3, max_draft=4, spec="ngram", k=4)
    outs = p.run([PROMPTS[0], PROMPTS[3], PROMPTS[2]], 20, settings=[s, None, None])
    assert outs[0] == alone
    assert outs[1] == oracle(KINDS_MIXED, PROMPTS[3], 20)
    assert outs[2] == oracle(KINDS_MIXED, PROMPTS[2], 20)


def test_more_requests_than_slots_queue_and_all_complete(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=4, spec="ngram", k=3)
    prompts = list(PROMPTS)
    outs = p.run(prompts, 12)
    for prompt, toks in zip(prompts, outs):
        assert toks == oracle(KINDS_MIXED, prompt, 12)
    assert p.settle()
    assert p.engine.stats()["requests"] == len(prompts)


def test_several_sequences_are_really_in_flight_at_once(pipes):
    _torch()
    p = pipes([(0, 1), (1, 3), (3, 4)], slots=4, max_draft=0, spec="none")
    p.run(PROMPTS[:4], 16)
    st = p.engine.stats()
    assert st["inflight"]["max"] >= 2, st["inflight"]
    assert st["rows_per_decode_step"] is not None


def test_pipelining_beats_one_sequence_at_a_time_when_nodes_are_slow(pipes, monkeypatch):
    """Each block call costs 6 ms: a one-token-at-a-time pipeline pays every node in turn per
    token; with 4 sequences in flight the nodes work on different sequences at once."""
    _torch()
    orig = FakeBlock.forward

    def slow(self, x, params):
        time.sleep(0.006)
        return orig(self, x, params)
    monkeypatch.setattr(FakeBlock, "forward", slow)
    prompts, n = PROMPTS[:4], 10
    t0 = time.monotonic()
    serial = [legacy_generate(KINDS_MIXED, pr, n) for pr in prompts]
    t_serial = time.monotonic() - t0
    p = pipes([(0, 1), (1, 3), (3, 4)], slots=4, max_draft=0, spec="none")
    t0 = time.monotonic()
    got = p.run(prompts, n)
    t_batched = time.monotonic() - t0
    assert got == serial
    assert t_batched < 0.7 * t_serial, (t_batched, t_serial)


def _slow_target_blocks(pipe, seconds):
    """Make every block of the TARGET nodes cost `seconds` (the draft model, a separate
    object, stays fast: a small draft is cheap next to the big model)."""
    for w in pipe.nodes:
        for m in w.model.modules:
            if isinstance(m, FakeBlock):
                orig = m.forward
                m.forward = lambda x, params, orig=orig: (time.sleep(seconds), orig(x, params))[1]


def test_speculation_cuts_pipeline_trips_when_the_text_is_predictable(pipes):
    _torch()
    p0 = pipes([(0, 2), (2, 4)], slots=1, max_draft=4, spec="none")
    _slow_target_blocks(p0, 0.004)
    t0 = time.monotonic()
    plain = p0.run([PROMPTS[0]], 30)[0]
    t_plain = time.monotonic() - t0
    p1 = pipes([(0, 2), (2, 4)], slots=1, max_draft=4, spec="draft", k=4, draft=True, perturb=0.0)
    _slow_target_blocks(p1, 0.004)
    t0 = time.monotonic()
    spec = p1.run([PROMPTS[0]], 30)[0]
    t_spec = time.monotonic() - t0
    assert spec == plain
    trips0 = p0.engine.stats()["steps"]["decode"]
    trips1 = p1.engine.stats()["steps"]["decode"]
    assert trips0 == 30 and trips1 <= 8, (trips0, trips1)      # pipeline trips: 30 -> ~6
    assert t_spec < 0.6 * t_plain, (t_spec, t_plain)


# ---- lifecycle / failure handling

def test_closing_a_generator_frees_its_slot_on_every_node(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=4, spec="ngram")
    with closing(p.engine.generate(PROMPTS[0], SamplingSettings(), 100)) as g:
        for _, _ in zip(range(3), g):
            pass
    assert p.settle(), "the cancelled sequence's slot leaked"
    # and the pipeline is fine afterwards
    assert p.run([PROMPTS[2]], 10)[0] == oracle(KINDS_MIXED, PROMPTS[2], 10)


def test_stop_ids_end_a_sequence_at_the_first_stop_token(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=4, spec="ngram")
    full = oracle(KINDS_MIXED, PROMPTS[0], 30)
    stop = full[5]
    first = full.index(stop)
    got = [t for t, _ in p.engine.generate(PROMPTS[0], SamplingSettings(), 30, stop_ids=[stop])]
    assert got == full[:first + 1]
    assert p.settle()


def test_prompt_plus_max_new_over_the_context_is_refused(pipes):
    _torch()
    p = pipes([(0, 4)], slots=2, max_draft=0)
    with pytest.raises(ValueError, match="exceeds"):
        list(p.engine.generate([1] * 250, SamplingSettings(), 20))
    with pytest.raises(ValueError, match="logits"):
        list(p.engine.generate([1, 2], SamplingSettings(), 2, return_logits=True))
    with pytest.raises(ValueError, match="draft"):
        list(p.engine.generate([1, 2], SamplingSettings(), 2, spec="draft"))
    with pytest.raises(ValueError, match="spec"):
        list(p.engine.generate([1, 2], SamplingSettings(), 2, spec="medusa"))


def test_a_sequence_that_runs_to_the_end_of_the_context_never_spills_into_the_next_slot(pipes):
    """Padding of a speculative row past the context stays in the slot's slack page: the
    neighbour sequence's output is untouched."""
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=4, spec="ngram", k=4)
    long_prompt = ([1, 5, 2, 8, 0, 4, 9, 6, 3, 3, 7] * 21)[:230]
    n = CTX - len(long_prompt)
    outs = p.run([long_prompt, PROMPTS[3]], n)
    assert outs[0] == oracle(KINDS_MIXED, long_prompt, n)
    assert outs[1] == oracle(KINDS_MIXED, PROMPTS[3], n)


def test_a_failing_node_fails_only_the_requests_in_that_step_and_the_pipeline_recovers(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=0, spec="none")
    fail = {"on": True}
    layer = p.nodes[1].model.modules[0].attn      # the first block's attention on node 1
    orig = layer.forward

    def flaky(x, params):
        if fail["on"]:
            fail["on"] = False
            raise RuntimeError("injected node failure")
        return orig(x, params)
    layer.forward = flaky
    with pytest.raises(BigBrainRemoteError, match="injected node failure"):
        p.run([PROMPTS[0]], 10)
    assert p.settle()
    assert p.run([PROMPTS[0]], 10)[0] == oracle(KINDS_MIXED, PROMPTS[0], 10)


# ---- the protocol: handshake, mixed builds, the old single-sequence traffic

def test_handshake_reports_every_node(pipes):
    _torch()
    p = pipes([(0, 1), (1, 3), (3, 4)], slots=3, max_draft=4)
    caps = p.engine.node_caps
    assert [c["rank"] for c in caps] == [0, 1, 2]
    assert all(c["proto"] == 3 and c["slots"] == 3 and c["batched"] for c in caps)


def test_a_node_with_too_few_slots_is_refused(pipes):
    _torch()
    with pytest.raises(EngineDead, match="slot"):
        pipes([(0, 2), (2, 4)], slots=4, max_draft=0, node_slots=[4, 2])


def test_a_recurrent_model_needs_enough_state_history_for_speculation(pipes):
    """A GatedDeltaNet-style state rewinds from recorded history; the nodes must have been
    started with --max-draft >= k.  (The sliding-window state of gpt-oss does not need it.)"""
    _torch()
    # the fake sliding-window state rewinds in place (guaranteed_rollback): no history needed
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=0, spec="ngram", k=4)
    assert p.run([PROMPTS[3]], 12)[0] == oracle(KINDS_MIXED, PROMPTS[3], 12)

    class NeedsHistory(FakeSWAState):
        guaranteed_rollback = 0
    import bigbrain_speed_fakes as fk
    orig = fk.FakeModel.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        if self.recurrent_state_cls is not None:
            self.recurrent_state_cls = NeedsHistory
    fk.FakeModel.__init__ = init
    try:
        with pytest.raises(EngineDead, match="max-draft"):
            pipes([(0, 2), (2, 4)], slots=2, max_draft=0, spec="ngram", k=4)
    finally:
        fk.FakeModel.__init__ = orig


def test_a_node_on_an_older_build_never_answers_and_the_handshake_says_so(pipes):
    """An old worker relays an unknown control op and never acks it: node 0 must not start
    sending steps into it."""
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=0, start=False, hello_timeout=0.6)
    p.nodes[1]._on_batch_control = lambda msg: False        # the old build knows no batched ops
    with pytest.raises(EngineDead, match="handshake|timed out"):
        p.engine.start()


def test_an_old_style_node_zero_still_drives_new_workers(pipes):
    """Interop the other way: the single-sequence traffic (begin / tokens / activations) works
    against workers built with several slots and speculation headroom."""
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=4, max_draft=4, start=False)
    p.engine.close()
    toks = []
    with closing(p.head.generate(PROMPTS[0], SamplingSettings(), 12)) as g:
        toks = [t for t, _ in g]
    assert toks == oracle(KINDS_MIXED, PROMPTS[0], 12)
    # and a batched sequence is refused while the pipeline is in single-sequence mode
    assert p.settle()


def test_single_sequence_and_batched_modes_cannot_be_mixed(pipes):
    _torch()
    p = pipes([(0, 4)], slots=2, max_draft=0, start=False)
    gen = p.head.generate(PROMPTS[0], SamplingSettings(), 5)
    next(gen)
    from bigbrain.paged import BigBrainStateError
    with pytest.raises(BigBrainStateError, match="cannot be mixed"):
        p.head.seq_begin(9, 1, SamplingSettings())
    gen.close()


def test_node_stats_report_busy_wait_rollbacks_and_slots(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=4, spec="ngram", k=4)
    p.run(PROMPTS[:2], 16)
    rep = p.engine.report(reset=False)
    assert len(rep["nodes"]) == 2
    for n in rep["nodes"]:
        assert n["steps"] > 0 and 0.0 <= n["busy_pct"] <= 100.0 and n["slots"] == 2
        assert n["wait_s"] >= 0 and n["send_s"] >= 0
    e = rep["engine"]
    assert e["decode_tokens"] == 32 and e["decode_tok_s"] and e["first_token_s"]["n"] == 2
    assert e["per_sequence_tok_s"] and e["spec"]["mode"] == "ngram"
    # reset zeroes the window
    p.engine.report(reset=True)
    again = p.engine.report()
    assert again["engine"]["decode_tokens"] == 0 and again["nodes"][0]["steps"] == 0


# ---- randomised: the engine under many shapes of load, everything checked against the oracle

def _fuzz_once(pipes, seed):
    import random
    rnd = random.Random(seed)
    bounds = rnd.choice(BOUNDS)
    slots = rnd.choice([1, 2, 3, 5])
    spec = rnd.choice(["none", "ngram", "draft"])
    k = rnd.randint(1, 6)
    p = pipes(bounds, slots=slots, max_draft=6, spec=spec, k=k, draft=(spec == "draft"),
              perturb=rnd.choice([0.0, 0.3, 1.0]), inflight=rnd.choice([0, 1, 2, 5]),
              mb=rnd.choice([0, 1, 2, 4]), prefill_chunk=rnd.choice([1, 2, 3, 7, 50]),
              window=rnd.choice([1, 2, 4]))
    n_req = rnd.randint(1, 7)
    prompts = [[rnd.randint(0, 10) for _ in range(rnd.randint(1, 30))] for _ in range(n_req)]
    lens = [rnd.randint(1, 25) for _ in range(n_req)]
    kws = [{"spec": rnd.choice(["none", "ngram"] + (["draft"] if spec == "draft" else [])),
            "spec_k": rnd.randint(1, 6)} for _ in range(n_req)]
    outs = [None] * n_req
    errs = []

    stops = [rnd.choice([None, None, None, rnd.randint(1, 8)]) for _ in range(n_req)]   # some clients leave early

    def one(i):
        try:
            got = []
            with closing(p.engine.generate(prompts[i], SamplingSettings(), lens[i], **kws[i])) as g:
                for t, _ in g:
                    got.append(t)
                    if stops[i] is not None and len(got) >= stops[i]:
                        break
            outs[i] = got
        except BaseException as exc:  # noqa: BLE001
            errs.append((i, exc))
    ths = [threading.Thread(target=one, args=(i,)) for i in range(n_req)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(60)
        assert not th.is_alive(), f"seed {seed}: hung"
    assert not errs, (seed, errs)
    for i in range(n_req):
        want = oracle(KINDS_MIXED, prompts[i], lens[i])
        if stops[i] is not None and stops[i] < lens[i]:
            assert outs[i] == want[:stops[i]], (seed, i, "cancelled")
        else:
            assert outs[i] == want, (seed, i, bounds, slots, spec, kws[i])
    assert p.settle(), f"seed {seed}: leaked"


@pytest.mark.parametrize("seed", range(12))
def test_randomised_loads_always_match_the_oracle_and_never_leak(pipes, seed):
    _torch()
    _fuzz_once(pipes, seed)


def test_a_draft_model_that_fails_degrades_to_plain_decode_without_changing_the_output(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=4, spec="draft", k=4, draft=True)
    calls = {"n": 0}
    real = p.draft_backend.forward

    def broken(rows):
        calls["n"] += 1
        raise RuntimeError("draft CUDA out of memory (injected)")
    p.draft_backend.forward = broken
    outs = p.run(PROMPTS[:2], 16)
    for prompt, toks in zip(PROMPTS[:2], outs):
        assert toks == oracle(KINDS_MIXED, prompt, 16)
    assert calls["n"] <= 2, "a failed drafter must not be retried every step"
    st = p.engine.stats()["spec"]
    assert st["proposed"] == 0 and st["plain_rows"] > 0
    p.draft_backend.forward = real
