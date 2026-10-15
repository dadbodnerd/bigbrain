# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""A BigBrain pipeline in one process for tests: a line of SliceWorker nodes over
real loopback sockets running the fake model of bigbrain_speed_fakes.py, node 0
driven by the real PipelineEngine. Not a test module."""
from __future__ import annotations

import threading
import time
from contextlib import closing

import pytest

from bigbrain_speed_fakes import (KINDS_MIXED, V, FakeCache, FakeModel,
                                  fake_advance_recurrent_states, oracle_greedy)
from bigbrain.draft import ExllamaDraftForward
from bigbrain.engine import EngineConfig, PipelineEngine
from bigbrain.paged import PagedSlots
from bigbrain.pipe import BigBrainLink
from bigbrain.spec import DraftModelDrafter, SpecConfig
from bigbrain.worker import SamplingSettings, SliceWorker, _select_slice, make_cache

CTX = 256
PROMPTS = (
    [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
    [7],                                   # one token: no prefill at all
    [2, 2, 2, 2, 2, 2],
    [1, 5, 2, 8, 0, 4, 9, 6, 3, 3, 7, 1, 5, 2, 8, 0, 4],
    [8, 8, 1, 0, 8, 8, 1, 0, 8, 8, 1],
    [10, 9, 8, 7, 6],
)
BOUNDS = ([(0, 4)], [(0, 2), (2, 4)], [(0, 1), (1, 3), (3, 4)])
_ORACLE = {}


def oracle(kinds, prompt, n):
    key = (tuple(kinds), tuple(prompt), n)
    if key not in _ORACLE:
        _ORACLE[key] = oracle_greedy(FakeModel(kinds), prompt, n)
    return _ORACLE[key]


def links_line(n):
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


def make_node(kinds, start, end, rank, total, *, slots, max_draft, ctx=CTX, prefill_chunk=3, **links):
    model = FakeModel(kinds)
    _select_slice(model, start, end, rank, total)
    model.prepare_layer_map()
    batched = slots > 1 or max_draft > 0
    tokens = PagedSlots.tokens_for(slots, ctx) if batched else ctx
    cache = make_cache(FakeCache, model, tokens, max_batch_size=slots, max_history=max_draft)
    return SliceWorker.from_parts(
        model=model, cache=cache, rank=rank, total=total, config=model.config, device="cpu",
        max_num_tokens=ctx, prefill_chunk=prefill_chunk,
        advance_fn=fake_advance_recurrent_states, slots=slots, max_draft=max_draft, **links)


def draft_backend(kinds, slots, max_draft, ctx=CTX, perturb=0.25):
    dm = FakeModel(kinds, perturb=perturb)
    dm.prepare_layer_map()
    cache = make_cache(FakeCache, dm, PagedSlots.tokens_for(slots, ctx), max_batch_size=slots,
                       max_history=max_draft)
    return ExllamaDraftForward(dm, cache, slots, ctx, vocab_size=V, device="cpu")


class Pipe:
    """A line of nodes, node 0 driven by a PipelineEngine."""

    def __init__(self, bounds, kinds=KINDS_MIXED, *, slots=4, max_draft=4, spec="none", k=4,
                 inflight=0, mb=0, draft=False, perturb=0.25, hello_timeout=10.0,
                 prefill_chunk=3, ctx=CTX, window=2, start=True, node_slots=None, node_draft=None):
        self.kinds, self.slots, self.ctx = kinds, slots, ctx
        n = len(bounds)
        links = links_line(n)
        node_slots = node_slots or [slots] * n
        node_draft = node_draft or [max_draft] * n
        self.nodes = [make_node(kinds, a, b, r, n, slots=node_slots[r], max_draft=node_draft[r],
                                ctx=ctx, prefill_chunk=prefill_chunk, **links[r])
                      for r, (a, b) in enumerate(bounds)]
        self.results = {}
        self.threads = []
        for w in self.nodes[1:]:
            w.start_recv()
            th = threading.Thread(
                target=lambda w=w: self.results.__setitem__(w.rank, w.serve(idle_dead_s=20.0)),
                daemon=True)
            th.start()
            self.threads.append(th)
        self.head = self.nodes[0]
        self.head.start_recv()
        drafters = {}
        if draft:
            be = draft_backend(kinds, slots, max_draft, ctx, perturb)
            drafters["draft"] = DraftModelDrafter(be, prefill_chunk=prefill_chunk)
            self.draft_backend = be
        self.engine = PipelineEngine(
            self.head, EngineConfig(spec=SpecConfig(mode=spec, k=k), inflight=inflight, mb_size=mb,
                                    prefill_chunk=prefill_chunk, prefill_window=window,
                                    hello_timeout_s=hello_timeout), drafters)
        if start:
            self.engine.start()

    def run(self, prompts, n, settings=None, kw=None, **kw2):
        """Generate every prompt CONCURRENTLY; returns the token lists in order.
        kw may be a list of per-prompt kwargs."""
        outs = [None] * len(prompts)
        errs = [None] * len(prompts)

        def one(i):
            try:
                k = kw[i] if isinstance(kw, list) else (kw or kw2)
                s = (settings[i] if isinstance(settings, list) else settings) or SamplingSettings()
                outs[i] = [t for t, _ in self.engine.generate(prompts[i], s, n, **k)]
            except BaseException as exc:  # noqa: BLE001
                errs[i] = exc
        ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(prompts))]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=60)
            assert not th.is_alive(), "a generation hung"
        for e in errs:
            if e is not None:
                raise e
        return outs

    def settle(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(not w.paged.active() and len(w.cache.free_list) == w.cache.num_slots
                   for w in self.nodes):
                return True
            time.sleep(0.01)
        return False

    def shutdown(self):
        self.engine.close()
        self.head.send_terminator()
        for th in self.threads:
            th.join(timeout=10)
        for w in self.nodes:
            w.close()


@pytest.fixture
def pipes():
    made = []

    def make(*a, **kw):
        p = Pipe(*a, **kw)
        made.append(p)
        return p
    yield make
    for p in made:
        p.shutdown()


def legacy_generate(kinds, prompt, n, settings=None, prefill_chunk=3):
    """THE PLAIN SINGLE-SEQUENCE RUN: one node, the original generate() path."""
    w = make_node(kinds, 0, len(kinds), 0, 1, slots=1, max_draft=0, prefill_chunk=prefill_chunk)
    with closing(w.generate(prompt, settings or SamplingSettings(), n)) as g:
        return [t for t, _ in g]


# ---- the OpenAI-compatible server over a Pipe (server + bench tests)

KEY = "speed-test-key"


def _server_pieces():
    from bigbrain.server import BigBrainApp, ChatTemplate, WorkerEngine, serve_in_thread
    return BigBrainApp, ChatTemplate, WorkerEngine, serve_in_thread


class FakeWorkerEngine:
    """server.WorkerEngine over the fake pipeline: token ids are the text (a..k), no tokenizer files."""

    def __new__(cls, pipe):
        _, ChatTemplate, WorkerEngine, _ = _server_pieces()

        class _E(WorkerEngine):
            def __init__(self, pipe):
                self.w = pipe.head
                self.batch = pipe.engine
                self.model_name = "fake-gptoss"
                self.max_context = pipe.head.max_num_tokens
                self.template = ChatTemplate(None)
                self.eos_ids = {V - 1}
                self.add_bos = False
                self.token_timeout_s = 30.0

            def encode(self, text, special=True):
                return [ord(c) - 97 for c in text]

            def decode(self, ids):
                return "".join(chr(97 + int(i)) for i in ids)
        return _E(pipe)


@pytest.fixture
def served(pipes):
    BigBrainApp, _, _, serve_in_thread = _server_pieces()
    started = []

    def make(*a, **kw):
        p = pipes(*a, **kw)
        app = BigBrainApp(FakeWorkerEngine(p), api_key=KEY, queue_size=8, retry_after=1)
        httpd, port, th = serve_in_thread(app)
        started.append((app, httpd))
        return p, app, f"http://127.0.0.1:{port}"
    yield make
    for app, httpd in started:
        httpd.shutdown()
        httpd.server_close()
        app.close()
