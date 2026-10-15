# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""The OpenAI-compatible node-0 server in BATCHED mode (several requests at once,
speculation, metrics), and the CLI surface the cluster controller and the staging
harness drive. The pipeline behind it is the in-process fake one."""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from bigbrain_speed_fakes import KINDS_MIXED, V, FakeBlock
from bigbrain_speed_harness import (CTX, KEY, PROMPTS, FakeWorkerEngine, Pipe, oracle,  # noqa: F401
                                    pipes, served)
from bigbrain import server as srv
from bigbrain import worker as wk
from bigbrain.server import BigBrainApp, serve_in_thread

def _torch():
    return pytest.importorskip("torch")


def call(base, path, body=None, method=None, key=KEY):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 method=method or ("GET" if body is None else "POST"),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def gen(base, prompt, n, **extra):
    return call(base, "/v1/bigbrain/generate",
                {"prompt_ids": prompt, "max_tokens": n, "ignore_eos": True, **extra})


def test_concurrent_requests_all_match_the_plain_run_through_http(served):
    _torch()
    p, app, base = served([(0, 2), (2, 4)], slots=4, max_draft=4, spec="ngram", k=4)
    prompts = PROMPTS[:4]
    out = [None] * 4

    def one(i):
        out[i] = gen(base, prompts[i], 20)
    ths = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(60)
    for prompt, (code, body) in zip(prompts, out):
        assert code == 200, body
        assert body["tokens"] == oracle(KINDS_MIXED, prompt, 20)
        assert body["finish_reason"] == "length" and body["usage"]["completion_tokens"] == 20
        assert body["info"]["tokens"] == 20 and "proposed" in body["info"]


def test_the_scheduler_runs_several_requests_at_the_same_time(served, monkeypatch):
    _torch()
    orig = FakeBlock.forward
    monkeypatch.setattr(FakeBlock, "forward", lambda self, x, params: (time.sleep(0.003), orig(self, x, params))[1])
    p, app, base = served([(0, 2), (2, 4)], slots=3, max_draft=0)
    seen = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            seen.append(app.scheduler.running)
            time.sleep(0.002)
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    res = [None] * 3
    ts = [threading.Thread(target=lambda i=i: res.__setitem__(i, gen(base, PROMPTS[i], 12))) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    stop.set()
    assert max(seen) >= 2, "requests were served one at a time"
    assert all(code == 200 for code, _ in res)
    assert app.scheduler.workers == 3


def test_per_request_speculation_override_and_diag_margins(served):
    _torch()
    p, app, base = served([(0, 4)], slots=2, max_draft=4, spec="none", draft=True, perturb=0.2)
    for spec in ("none", "ngram", "draft"):
        code, body = gen(base, PROMPTS[3], 18, spec=spec, spec_k=3, diag=True)
        assert code == 200, body
        assert body["tokens"] == oracle(KINDS_MIXED, PROMPTS[3], 18), spec
        assert body["info"]["spec_mode"] == spec
        assert len(body["info"]["margins"]) == 18 and all(m >= 0 for m in body["info"]["margins"])
    code, body = gen(base, PROMPTS[3], 18, spec="ngram", spec_k=4)
    assert body["info"]["proposed"] > 0 and 0 <= body["info"]["accepted"] <= body["info"]["proposed"]


def test_bad_speculation_requests_are_refused(served):
    _torch()
    p, app, base = served([(0, 4)], slots=2, max_draft=4, spec="none")
    assert gen(base, PROMPTS[0], 4, spec="medusa")[0] == 400
    assert gen(base, PROMPTS[0], 4, spec="ngram", spec_k=99)[0] == 400
    code, body = gen(base, PROMPTS[0], 4, spec="draft")
    assert code == 400 and "draft model" in body["error"]["message"]
    code, body = gen(base, PROMPTS[0], 4, return_logits=True)
    assert code == 400 and "batched" in body["error"]["message"]
    assert call(base, "/v1/bigbrain/generate", {"prompt_ids": [1], "max_tokens": 2}, key="wrong")[0] == 401


def test_metrics_route_reports_engine_and_every_node_and_can_reset(served):
    _torch()
    p, app, base = served([(0, 2), (2, 4)], slots=2, max_draft=4, spec="ngram")
    gen(base, PROMPTS[3], 16)
    code, m = call(base, "/v1/bigbrain/metrics")
    assert code == 200 and m["mode"] == "batched"
    e = m["engine"]
    assert e["decode_tokens"] == 16 and e["decode_tok_s"] > 0 and e["spec"]["mode"] == "ngram"
    assert e["first_token_s"]["n"] == 1 and e["steps"]["decode"] > 0
    assert [n["rank"] for n in m["nodes"]] == [0, 1]
    assert all("busy_pct" in n and "wait_pct" in n and "send_pct" in n for n in m["nodes"])
    assert m["scheduler"]["served"] == 1
    code, m2 = call(base, "/v1/bigbrain/metrics", {"reset": True})
    assert code == 200 and m2["engine"]["decode_tokens"] == 16
    code, m3 = call(base, "/v1/bigbrain/metrics")
    assert m3["engine"]["decode_tokens"] == 0 and m3["nodes"][0]["steps"] == 0


def test_health_reports_slots_and_stays_503_until_the_pipeline_is_ready(pipes):
    _torch()
    p = pipes([(0, 2), (2, 4)], slots=2, max_draft=0, start=False)
    app = BigBrainApp(FakeWorkerEngine(p), api_key=KEY, queue_size=2, retry_after=1)
    httpd, port, _ = serve_in_thread(app)
    base = f"http://127.0.0.1:{port}"
    try:
        code, body = call(base, "/health", key="")
        assert code == 503 and body["status"] == "degraded" and body["engine"]["ready"] is False
        code, body = gen(base, PROMPTS[0], 3)
        assert code == 503 and "starting" in body["error"]["message"]
        p.engine.start()
        code, body = call(base, "/health", key="")
        assert code == 200 and body["slots"] == 2 and body["engine"]["batched"] is True
        assert body["engine"]["slots"] == 2
        gen(base, PROMPTS[0], 4)
        code, body = call(base, "/health", key="")
        assert body["requests_served"] == 1 and body["served"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_a_dead_pipeline_is_reported_and_refuses_requests(served):
    _torch()
    p, app, base = served([(0, 2), (2, 4)], slots=2, max_draft=0)
    p.engine._die("the token link died: test")
    time.sleep(0.2)
    code, body = gen(base, PROMPTS[0], 3)
    assert code == 503 and "down" in body["error"]["message"]
    code, h = call(base, "/health", key="")
    assert code == 503 and h["engine"]["dead"]


# ---- the command-line surface

def _parse_node(argv):
    return wk.build_arg_parser().parse_args(argv)


BASE = ["--model-dir", "/m", "--slice", "0", "9", "--rank", "0", "--total", "2"]


def test_slots_max_batch_size_and_spec_flags_agree_across_worker_and_server():
    a = _parse_node(BASE + ["--max-batch-size", "4"])
    assert wk.effective_slots(a) == 4 and wk.effective_max_draft(a) == 0
    a = _parse_node(BASE + ["--slots", "6", "--max-batch-size", "2"])
    assert wk.effective_slots(a) == 6
    a = _parse_node(BASE + ["--spec-draft", "ngram", "--spec-k", "5"])
    assert wk.effective_max_draft(a) == 5 and a.spec_draft == "ngram"
    a = _parse_node(BASE + ["--spec-draft", "model", "--spec-model", "/m20", "--draft-device", "cuda:1",
                            "--max-draft", "8"])
    assert (a.spec_draft, a.spec_model, a.draft_device, wk.effective_max_draft(a)) == ("model", "/m20", "cuda:1", 8)
    # the server takes the same node flags plus its own
    s = srv.build_arg_parser().parse_args(BASE + ["--http-port", "9", "--api-token", "T", "--slots", "3",
                                                  "--spec-draft", "ngram", "--inflight", "4",
                                                  "--mb-size", "2", "--hello-timeout", "30"])
    assert (s.slots, s.api_token, s.inflight, s.mb_size, s.hello_timeout) == (3, "T", 4, 2, 30.0)


def test_the_launch_script_speed_flags_parse():
    """A launch script appends these to EVERY node's command (rank 0 is the server):
    --max-batch-size N plus the draft setting."""
    sweep = {"slots": "--max-batch-size {n}", "draft": {"ngram": "--spec-draft ngram",
                                                         "20b": "--spec-draft model --spec-model {path}"}}
    import shlex
    extra = (sweep["slots"].format(n=4) + " " + sweep["draft"]["20b"].format(path="/m/gpt-oss-20b")).strip()
    node = BASE + ["--prefill-chunk", "2048", "--max-batch-size", "4"] + shlex.split(extra)
    a = _parse_node(node)
    assert wk.effective_slots(a) == 4 and a.spec_draft == "model" and a.spec_model == "/m/gpt-oss-20b"
    srv_args = srv.build_arg_parser().parse_args(
        node + ["--host", "0.0.0.0", "--http-port", "16980", "--model-name", "x", "--api-key-file", "api_key"])
    assert srv_args.http_port == 16980 and wk.effective_slots(srv_args) == 4


def test_watchdog_flag_reaches_the_slice_worker(monkeypatch):
    """A token link once died 'no frame for 5.0s' and the 5 s was a constructor default no flag could change.
    --watchdog-s now reaches every node (worker AND server parser) and defaults to the old 5.0."""
    default = _parse_node(BASE)
    assert default.watchdog_s == 5.0 and wk.effective_watchdog(default) == 5.0
    longer = _parse_node(BASE + ["--watchdog-s", "30"])
    srv_args = srv.build_arg_parser().parse_args(
        BASE + ["--watchdog-s", "30", "--host", "0.0.0.0", "--http-port", "16980", "--model-name", "x", "--api-key-file", "k"])
    assert longer.watchdog_s == 30.0 and srv_args.watchdog_s == 30.0
    seen = {}

    class Recorder:
        def __init__(self, *a, **kw):
            seen.update(kw)
    monkeypatch.setattr(wk, "SliceWorker", Recorder)
    wk.worker_from_args(longer)
    assert seen["watchdog_s"] == 30.0
    wk.worker_from_args(default)
    assert seen["watchdog_s"] == 5.0
    with pytest.raises(SystemExit):                       # two lost heartbeats must not be able to kill a healthy link
        wk.effective_watchdog(_parse_node(BASE + ["--watchdog-s", "1"]))


def test_serving_flags_on_the_worker_command_line_hand_over_to_the_server(monkeypatch):
    """REGRESSION: the controller launches node 0 as
    `-m bigbrain.worker ... --http-port N --api-token T` and worker.py rejected it."""
    got = []
    monkeypatch.setattr(srv, "main", lambda argv=None: (got.append(list(argv)), 0)[1])
    argv = BASE + ["--http-port", "16980", "--api-token", "SECRET"]
    assert wk.main(argv) == 0
    assert got == [argv]
    got.clear()
    assert wk.main(BASE + ["--http-port=16980"]) == 0 and got      # the = form too


def test_api_token_flag_beats_file_and_env(tmp_path, monkeypatch):
    f = tmp_path / "k"
    # keys of at least MIN_API_KEY_LENGTH (16) characters since v0.2.1
    f.write_text("from-file-0123456789")
    monkeypatch.setenv("BIGBRAIN_API_KEY", "from-env-0123456789")
    assert srv.resolve_api_key(str(f), "BIGBRAIN_API_KEY", False, token="from-flag-0123456789") == "from-flag-0123456789"
    assert srv.resolve_api_key(str(f), "BIGBRAIN_API_KEY", False) == "from-file-0123456789"
    assert srv.resolve_api_key(None, "BIGBRAIN_API_KEY", False) == "from-env-0123456789"


def test_the_shipped_speed_flags_file_works_with_a_launch_scripts_formatting():
    """A launch script builds `extra` exactly like this and appends it to EVERY node's
    command (after its own `--max-batch-size <slots>`)."""
    import shlex
    from pathlib import Path
    import importlib.resources
    flags = json.loads(importlib.resources.files("bigbrain").joinpath("speed_flags.json").read_text(encoding="utf-8"))
    for slots in (1, 4, 8):
        for draft in ("none", "ngram", "20b"):
            extra = (flags.get("slots", "").format(n=slots) + " " +
                     (flags.get("draft", {}).get(draft, "") if draft != "none" else "")).strip()
            node = BASE + ["--prefill-chunk", "2048", "--max-batch-size", str(slots)] + shlex.split(extra)
            for parser, more in ((wk.build_arg_parser(), []),
                                 (srv.build_arg_parser(), ["--http-port", "16980"])):
                a = parser.parse_args(node + more)
                assert wk.effective_slots(a) == slots
                assert a.spec_draft == {"none": "none", "ngram": "ngram", "20b": "model"}[draft]
                assert (wk.effective_max_draft(a) > 0) == (draft != "none")
                if draft == "20b":
                    assert a.spec_model.endswith("gpt-oss-20b-4.0bpw-exl3")


def test_make_batch_engine_follows_the_flags():
    _torch()
    from bigbrain_speed_harness import make_node
    from bigbrain_speed_fakes import KINDS_MIXED
    parse = srv.build_arg_parser().parse_args

    plain = make_node(KINDS_MIXED, 0, 4, 0, 1, slots=1, max_draft=0)
    assert srv.make_batch_engine(plain, parse(BASE + ["--http-port", "9"])) is None      # single-sequence server

    w = make_node(KINDS_MIXED, 0, 4, 0, 1, slots=3, max_draft=4)
    e = srv.make_batch_engine(w, parse(BASE + ["--http-port", "9", "--slots", "3", "--spec-draft", "ngram",
                                               "--spec-k", "4", "--inflight", "5", "--mb-size", "2",
                                               "--prefill-chunk", "512", "--prefill-window", "3"]))
    assert e.cfg.spec.mode == "ngram" and e.cfg.spec.k == 4 and e.slots == 3
    assert (e.cfg.inflight, e.cfg.mb_size, e.cfg.prefill_chunk, e.cfg.prefill_window) == (5, 2, 512, 3)
    assert e.cfg.hello_timeout_s is None and "draft" not in e.drafters
    with pytest.raises(SystemExit, match="spec-model"):
        srv.make_batch_engine(w, parse(BASE + ["--http-port", "9", "--slots", "3", "--spec-draft", "model"]))
