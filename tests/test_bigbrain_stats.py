# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Tests for the BigBrain server's per-request stats record (tools/bigbrain/server.py).

StatsRecorder is exercised on a fake clock (exact numbers); the HTTP route and the /health
`stats` block are exercised against a byte-level fake engine over loopback. No GPU, no
network beyond 127.0.0.1.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools.bigbrain.server import (
    BigBrainApp,
    ChatTemplate,
    JOB_LABEL_MAX,
    StatsRecorder,
    parse_job_tag,
    serve_in_thread,
)

KEY = "stats-key-0123456789abcdef"
EOS = 256


# --------------------------------------------------------------------------
# StatsRecorder on a fake clock
# --------------------------------------------------------------------------

class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _req(prompt=100, created=0.0, start=None, first=None, route="chat", job=None, spec=None):
    return SimpleNamespace(prompt_ids=[1] * prompt, t_created=created, t_start=start, t_first=first,
                           route=route, job=job, info={}, spec=spec)


def _recorder(avg_n=20):
    mono, wall = Clock(0.0), Clock(1_800_000_000.0)
    return StatsRecorder(avg_n=avg_n, wall=wall, mono=mono), mono, wall


def test_one_request_decode_prefill_and_first_token_numbers():
    rec, mono, _ = _recorder()
    mono.t = 13.0
    # arrived 0, a slot took it at 1, first token at 3, last token at 13: 101 tokens
    rec.record(_req(prompt=100, created=0.0, start=1.0, first=3.0), "stop", 101)
    s = rec.snapshot(served=1)
    assert s["decode_tok_s"]["last"] == 10.0            # 100 tokens after the first, in 10 s
    assert s["decode_tok_s"]["avg"] == 10.0 and s["decode_tok_s"]["avg_n"] == 1
    assert s["decode_tok_s"]["per_sequence"] == 10.0
    assert s["prefill_tok_s"]["last"] == 50.0           # 100 prompt tokens in the 2 s before the first token
    assert s["first_token_s"]["last"] == 3.0            # arrival -> first token, the 1 s queue wait included
    assert s["tokens"] == {"prompt_total": 100, "completion_total": 101}
    assert s["last_request"]["finish"] == "stop" and s["last_request"]["total_s"] == 13.0
    assert s["requests"]["served"] == 1


def test_avg_is_token_weighted_but_per_sequence_is_the_plain_mean():
    rec, mono, _ = _recorder()
    mono.t = 10.0
    rec.record(_req(created=0.0, start=0.0, first=0.0), "length", 101)    # 100 tokens / 10 s = 10 tok/s
    mono.t = 21.0
    rec.record(_req(created=10.0, start=10.0, first=20.0), "length", 31)  # 30 tokens / 1 s = 30 tok/s
    d = rec.snapshot()["decode_tok_s"]
    assert d["avg"] == pytest.approx(130 / 11, abs=0.01)                  # 11.82: the long request weighs more
    assert d["per_sequence"] == 20.0
    assert d["last"] == 30.0


def test_avg_covers_only_the_last_n_requests():
    rec, mono, _ = _recorder(avg_n=3)
    t = 0.0
    for i, rate in enumerate([100, 100, 100, 5, 5, 5]):
        start = t
        t += 10.0
        mono.t = t
        rec.record(_req(created=start, start=start, first=start), "stop", 10 * rate + 1)
    d = rec.snapshot()["decode_tok_s"]
    assert d["avg"] == 5.0 and d["avg_n"] == 3


def test_a_single_token_answer_has_no_decode_rate_and_is_left_out():
    rec, mono, _ = _recorder()
    mono.t = 4.0
    rec.record(_req(created=0.0, start=0.0, first=2.0), "stop", 1)
    d = rec.snapshot()["decode_tok_s"]
    assert d["last"] is None and d["avg"] is None and d["avg_n"] is None and d["aggregate"] is None
    # but its first-token time and its tokens still count
    assert rec.snapshot()["first_token_s"]["last"] == 2.0
    assert rec.snapshot()["tokens"]["completion_total"] == 1


def test_aggregate_counts_overlapping_slots_over_the_time_any_of_them_decoded():
    rec, mono, _ = _recorder()
    mono.t = 10.0
    rec.record(_req(created=0.0, start=0.0, first=0.0), "stop", 101)      # decodes 0..10: 100 tokens
    mono.t = 15.0
    rec.record(_req(created=5.0, start=5.0, first=5.0), "stop", 101)      # decodes 5..15: 100 tokens
    # 200 tokens over the 15 s at least one of them was decoding
    assert rec.snapshot()["decode_tok_s"]["aggregate"] == pytest.approx(200 / 15, abs=0.01)


def test_aggregate_keeps_showing_the_last_busy_minute_when_idle():
    rec, mono, _ = _recorder()
    mono.t = 10.0
    rec.record(_req(created=0.0, start=0.0, first=0.0), "stop", 101)
    mono.t = 100_000.0                                                    # long idle
    assert rec.snapshot()["decode_tok_s"]["aggregate"] == 10.0


def test_aggregate_window_is_the_last_minute_of_activity():
    rec, mono, _ = _recorder()
    mono.t = 100.0
    rec.record(_req(created=0.0, start=0.0, first=0.0), "stop", 1001)     # 10 tok/s, 0..100
    # the newest request ends at 100: only its last 60 s count -> 600 tokens / 60 s
    assert rec.snapshot()["decode_tok_s"]["aggregate"] == 10.0


def test_cancelled_error_and_unserved_prompts():
    rec, mono, _ = _recorder()
    mono.t = 1.0
    rec.record(_req(prompt=500, created=0.0, start=0.5), "cancelled", 0)      # left while queued: no first token
    rec.record(_req(prompt=200, created=0.0, start=0.5, first=0.8), "error", 5)
    s = rec.snapshot(served=0, failed=1)
    assert s["requests"]["cancelled"] == 1 and s["requests"]["failed"] == 1
    assert s["tokens"]["prompt_total"] == 200          # the 500-token prompt never produced a token
    assert s["tokens"]["completion_total"] == 5


def test_recent_is_capped_listed_newest_last_and_has_no_private_keys():
    rec, mono, _ = _recorder()
    for i in range(5):
        mono.t = float(i + 1)
        rec.record(_req(created=float(i), start=float(i), first=float(i)), "stop", 11)
    assert "recent" not in rec.snapshot()
    got = rec.snapshot(recent=3)["recent"]
    assert [r["id"] for r in got] == [3, 4, 5]
    assert not any(k.startswith("_") for r in got for k in r)
    small = StatsRecorder(recent=2, wall=Clock(1_800_000_000.0), mono=Clock(0.0))
    for _ in range(4):
        small.record(_req(created=0.0, start=0.0, first=0.0), "stop", 5)
    assert [r["id"] for r in small.snapshot(recent=10)["recent"]] == [3, 4]


def test_job_is_the_newest_tagged_request_and_survives_untagged_ones():
    rec, mono, wall = _recorder()
    assert rec.snapshot()["job"] is None
    rec.note_job(parse_job_tag("code review: money answer key", "3/5"))
    wall.t += 7
    rec.record(_req(created=0.0, start=0.0, first=0.0), "stop", 5)         # untagged request
    j = rec.snapshot()["job"]
    assert j["label"] == "code review: money answer key" and (j["unit"], j["of"]) == (3, 5)
    assert j["updated_at"] == 1_800_000_000.0


@pytest.mark.parametrize("label,progress,want", [
    ("code review: money answer key", "3/5", {"label": "code review: money answer key", "unit": 3, "of": 5}),
    ("  spaced  ", " 12 / 40 ", {"label": "spaced", "unit": 12, "of": 40}),
    ("x" * 500, None, {"label": "x" * JOB_LABEL_MAX, "unit": None, "of": None}),
    ("line1\nline2\x00\x07", "nonsense", {"label": "line1line2", "unit": None, "of": None}),
    ("", "1/2", None),
    ("\n\t ", "1/2", None),
    (None, "1/2", None),
    ("ok", "3/", {"label": "ok", "unit": None, "of": None}),
    ("ok", "99999999/5", {"label": "ok", "unit": None, "of": None}),
])
def test_job_tag_parsing(label, progress, want):
    assert parse_job_tag(label, progress) == want


# --------------------------------------------------------------------------
# over HTTP: /v1/bigbrain/stats and /health
# --------------------------------------------------------------------------

def _bytes_decode(ids):
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class FakeEngine:
    model_name = "fake-120b"

    def __init__(self, script, max_context=4096, delay=0.0):
        self.script, self.max_context, self.delay = list(script), max_context, delay
        self.template = ChatTemplate(None)
        self.eos_ids = {EOS}
        self.gate = None
        self.fail_at = None

    def encode(self, text, special=True):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return _bytes_decode(ids)

    def generate(self, prompt_ids, settings, max_new, request_id=None, return_logits=False):
        for i, tok in enumerate(self.script[:max_new]):
            if self.gate is not None:
                assert self.gate.wait(10)
            if self.fail_at == i:
                raise RuntimeError("pipeline exploded")
            if self.delay:
                time.sleep(self.delay)
            yield tok, None

    def health(self):
        return {"ok": True}


@pytest.fixture
def make_server():
    made = []

    def _make(engine, **kw):
        kw.setdefault("api_key", KEY)
        app = BigBrainApp(engine, **kw)
        httpd, port, _th = serve_in_thread(app, host="127.0.0.1", port=0)
        made.append((app, httpd))
        return app, port

    yield _make
    for app, httpd in made:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def _call(port, method, path, body=None, key=KEY, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    h = dict(headers or {})
    if key:
        h["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, (json.loads(raw) if raw else None)


CHAT = {"model": "x", "messages": [{"role": "user", "content": "Hi"}]}


def test_health_carries_the_stats_block_without_a_key_and_without_the_request_list(make_server):
    app, port = make_server(FakeEngine(list(b"hello") + [EOS], delay=0.01))
    status, h = _call(port, "GET", "/health", key=None)
    assert status == 200 and h["stats"]["requests"]["served"] == 0 and h["stats"]["last_request"] is None
    assert _call(port, "POST", "/v1/chat/completions", CHAT)[0] == 200
    status, h = _call(port, "GET", "/health", key=None)
    st = h["stats"]
    assert st["requests"]["served"] == 1 and h["served"] == 1          # one set of counters
    assert st["tokens"]["completion_total"] == 6                        # five bytes + the EOS
    assert st["last_request"]["finish"] == "stop" and st["last_request"]["completion_tokens"] == 6
    assert st["decode_tok_s"]["last"] and st["decode_tok_s"]["avg"] and st["decode_tok_s"]["avg_n"] == 1
    assert "recent" not in st


def test_stats_route_needs_the_key_and_lists_recent_requests(make_server):
    app, port = make_server(FakeEngine(list(b"abc") + [EOS]))
    assert _call(port, "GET", "/v1/bigbrain/stats", key=None)[0] == 401
    for _ in range(3):
        assert _call(port, "POST", "/v1/chat/completions", CHAT)[0] == 200
    assert _call(port, "POST", "/v1/completions", {"prompt": "go", "max_tokens": 3})[0] == 200
    status, st = _call(port, "GET", "/v1/bigbrain/stats")
    assert status == 200 and len(st["recent"]) == 4
    assert [r["route"] for r in st["recent"]] == ["chat", "chat", "chat", "completions"]
    assert all(r["prompt_tokens"] > 0 and r["finish"] in ("stop", "length") for r in st["recent"])
    status, st = _call(port, "GET", "/v1/bigbrain/stats?recent=2")
    assert len(st["recent"]) == 2 and st["requests"]["served"] == 4
    status, st = _call(port, "GET", "/v1/bigbrain/stats?recent=0")
    assert "recent" not in st
    assert _call(port, "GET", "/v1/bigbrain/stats?recent=abc")[0] == 400


def test_a_job_header_becomes_the_current_job(make_server):
    app, port = make_server(FakeEngine(list(b"ok") + [EOS]))
    assert _call(port, "GET", "/v1/bigbrain/stats")[1]["job"] is None
    hdr = {"X-BigBrain-Job": "code review: money answer key", "X-BigBrain-Job-Progress": "3/5"}
    assert _call(port, "POST", "/v1/chat/completions", CHAT, headers=hdr)[0] == 200
    st = _call(port, "GET", "/v1/bigbrain/stats")[1]
    assert st["job"]["label"] == "code review: money answer key" and (st["job"]["unit"], st["job"]["of"]) == (3, 5)
    assert st["recent"][-1]["job"] == "code review: money answer key"
    # an untagged request does not erase it
    assert _call(port, "POST", "/v1/chat/completions", CHAT)[0] == 200
    assert _call(port, "GET", "/health", key=None)[1]["stats"]["job"]["label"] == "code review: money answer key"


def test_a_failing_request_is_counted_and_does_not_break_the_stats(make_server):
    eng = FakeEngine(list(b"abcdef") + [EOS])
    eng.fail_at = 2
    app, port = make_server(eng)
    assert _call(port, "POST", "/v1/chat/completions", CHAT)[0] == 500
    st = _call(port, "GET", "/v1/bigbrain/stats")[1]
    assert st["requests"]["failed"] == 1 and st["recent"][-1]["finish"] == "error"
    assert st["recent"][-1]["completion_tokens"] == 2


def test_requests_in_flight_and_queued_are_live_numbers(make_server):
    eng = FakeEngine(list(b"zz") + [EOS])
    eng.gate = threading.Event()
    app, port = make_server(eng, queue_size=4)
    results = []

    def go():
        results.append(_call(port, "POST", "/v1/chat/completions", CHAT)[0])

    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads:
        t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        st = _call(port, "GET", "/v1/bigbrain/stats")[1]["requests"]
        if st["in_flight"] == 1 and st["queued"] == 2:
            break
        time.sleep(0.05)
    assert (st["in_flight"], st["queued"]) == (1, 2)
    eng.gate.set()
    for t in threads:
        t.join(15)
    assert results == [200, 200, 200]
    st = _call(port, "GET", "/v1/bigbrain/stats")[1]["requests"]
    assert (st["served"], st["in_flight"], st["queued"]) == (3, 0, 0)
