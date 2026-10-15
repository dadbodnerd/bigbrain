# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Tests for bigbrain/server.py -- the OpenAI-compatible endpoint on node 0.

No GPU, no exllamav3. A byte-level FakeEngine (token id = one UTF-8 byte, so
a multi-byte character really is split across tokens) stands in for the
pipeline for the HTTP/streaming/stop/queue tests; one end-to-end test drives
the real SliceWorker pipeline over loopback with the CPU-torch fake model from
test_bigbrain_worker.py. Everything binds 127.0.0.1.
"""
from __future__ import annotations

import base64
import http.client
import json
import socket
import threading
import time
from contextlib import closing

import numpy as np
import pytest

from bigbrain.server import (
    BigBrainApp,
    ChatTemplate,
    IncrementalDetokenizer,
    StopMatcher,
    WorkerEngine,
    build_arg_parser,
    load_generation_defaults,
    normalize_messages,
    resolve_api_key,
    serve_in_thread,
)
from bigbrain.worker import SamplingSettings

KEY = "test-key-0123456789abcdef"
EOS = 256
SPECIAL = {"<|im_start|>": 257, "<|im_end|>": 258}


def _bytes_decode(ids):
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# incremental detokenizer / stop strings
# --------------------------------------------------------------------------

def test_detokenizer_holds_back_a_split_multibyte_character():
    text = "ok \U0001F600 fine \u00e9"
    ids = list(text.encode("utf-8"))
    d = IncrementalDetokenizer(_bytes_decode)
    pieces = [d.push(t) for t in ids]
    assert "".join(pieces) + d.flush() == text
    assert not any("\ufffd" in p for p in pieces)
    # the 4-byte emoji came out in ONE piece, only after its last byte
    emoji_end = ids.index(0x80) + 1  # 0xF0 0x9F 0x98 0x80
    assert pieces[emoji_end - 1] == "\U0001F600"
    assert pieces[emoji_end - 4:emoji_end - 1] == ["", "", ""]


def test_detokenizer_flushes_a_truncated_character_at_the_end():
    d = IncrementalDetokenizer(_bytes_decode)
    assert d.push(0xF0) == "" and d.push(0x9F) == ""
    assert d.flush() == "\ufffd"


def test_detokenizer_uses_prompt_context_for_leading_spaces():
    """A SentencePiece-style decode drops a sequence-initial space; the
    window starts inside the prompt, so the first token keeps its space."""
    vocab = {1: "\u2581Hello", 2: "\u2581world"}

    def sp_decode(ids):
        return "".join(vocab[i] for i in ids).replace("\u2581", " ").lstrip(" ")

    d = IncrementalDetokenizer(sp_decode, context_ids=[1])
    assert d.push(2) == " world"


@pytest.mark.parametrize("deltas,stops,want_text,want_hit", [
    (["Hello wor", "ld STOP more"], ["STOP"], "Hello world ", True),
    (["abc<", "|im", "_end|>tail"], ["<|im_end|>"], "abc", True),
    (["a<", "b"], ["<|x"], "a<b", False),             # a partial prefix is released
    (["xx END", " yy"], ["END", "xx"], "", True),       # the earliest stop wins
    (["no stops here"], [], "no stops here", False),
])
def test_stop_matcher(deltas, stops, want_text, want_hit):
    m = StopMatcher(stops)
    out, hit = "", False
    for d in deltas:
        emit, hit = m.feed(d)
        out += emit
        if hit:
            break
    if not hit:
        out += m.flush()
    assert (out, hit) == (want_text, want_hit)


def test_stop_matcher_never_emits_part_of_a_stop_string():
    m = StopMatcher(["###"])
    assert m.feed("ab#") == ("ab", False)
    assert m.feed("#") == ("", False)
    assert m.feed("#") == ("", True)


# --------------------------------------------------------------------------
# chat template
# --------------------------------------------------------------------------

QWEN_LIKE = (  # the shape of Qwen's own template (newlines inside {{- '...' }})
    "{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
    "{%- endif %}")


def test_template_from_jinja_file_wins(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(QWEN_LIKE, encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": "IGNORED", "eos_token": "<|im_end|>", "bos_token": None}))
    t = ChatTemplate.from_model_dir(str(tmp_path))
    assert t.source == "chat_template.jinja"
    out = t.render([{"role": "user", "content": "hi"}])
    assert out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    out = t.render([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_template_from_tokenizer_config_and_named_list(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": [{"name": "tool_use", "template": "TOOLS"},
                          {"name": "default", "template": "{{ bos_token }}{{ messages[0].content }}"}],
        "bos_token": {"content": "<s>"}}))
    t = ChatTemplate.from_model_dir(str(tmp_path))
    assert t.source == "tokenizer_config.json"
    assert t.render([{"role": "user", "content": "x"}]) == "<s>x"


def test_template_raise_exception_and_tojson(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% if messages[0].role != 'user' %}{{ raise_exception('user first') }}{% endif %}"
        "{{ messages[0] | tojson }}", encoding="utf-8")
    t = ChatTemplate.from_model_dir(str(tmp_path))
    assert json.loads(t.render([{"role": "user", "content": "\u00e9"}])) == \
        {"role": "user", "content": "\u00e9"}
    with pytest.raises(ValueError, match="user first"):
        t.render([{"role": "system", "content": "s"}])


def test_template_fallback_is_chatml(tmp_path):
    t = ChatTemplate.from_model_dir(str(tmp_path))  # no template files at all
    assert t.source == "fallback-chatml"
    assert t.render([{"role": "system", "content": "be brief"},
                     {"role": "user", "content": "hi"}]) == (
        "<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\n")


def test_template_fallback_when_jinja2_is_missing(tmp_path, monkeypatch):
    import bigbrain.server as srv
    (tmp_path / "chat_template.jinja").write_text("{{ 'JINJA' }}", encoding="utf-8")
    monkeypatch.setattr(srv, "_jinja_env", lambda: None)
    t = ChatTemplate.from_model_dir(str(tmp_path))
    assert t.render([{"role": "user", "content": "q"}]).startswith("<|im_start|>user\nq")


def test_normalize_messages():
    msgs = normalize_messages([
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]}])
    assert msgs[0]["content"] == "ab"
    assert msgs[1]["content"] == "" and msgs[1]["tool_calls"] == [{"id": "1"}]
    for bad in ([], "x", [{"content": "no role"}], [{"role": "user", "content": 5}],
                [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]):
        with pytest.raises(ValueError):
            normalize_messages(bad)


def test_generation_defaults(tmp_path):
    assert load_generation_defaults(str(tmp_path)) == {}
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "do_sample": True, "eos_token_id": [1]}))
    assert load_generation_defaults(str(tmp_path)) == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False, "temperature": 0.7}))
    assert load_generation_defaults(str(tmp_path)) == {"temperature": 0.0}


# --------------------------------------------------------------------------
# auth key resolution
# --------------------------------------------------------------------------

def test_resolve_api_key(tmp_path, monkeypatch):
    f = tmp_path / "key"
    f.write_text("  from-file \n")
    monkeypatch.setenv("BB_KEY", "from-env")
    assert resolve_api_key(str(f), "BB_KEY", insecure=False) == "from-file"
    assert resolve_api_key(None, "BB_KEY", insecure=False) == "from-env"
    monkeypatch.delenv("BB_KEY")
    with pytest.raises(SystemExit):
        resolve_api_key(None, "BB_KEY", insecure=False)
    assert resolve_api_key(None, "BB_KEY", insecure=True) is None


def test_server_flags_accept_the_controller_node0_flags():
    args = build_arg_parser().parse_args(
        "--model-dir /m --device cuda:0 --slice 0 32 --rank 0 --total 2 --prompt Hi "
        "--max-new 32 --max-num-tokens 8192 --next-host 10.0.0.11 --next-port 16900 "
        "--token-port 16901 --http-port 5099 --queue-size 4".split())
    assert args.http_port == 5099 and args.queue_size == 4 and args.rank == 0
    assert args.host == "0.0.0.0" and args.retry_after == 5


# --------------------------------------------------------------------------
# the HTTP endpoint over a FakeEngine
# --------------------------------------------------------------------------

class FakeEngine:
    model_name = "fake-27b"

    def __init__(self, script, max_context=64, template=None):
        self.script = list(script)
        self.max_context = max_context
        self.template = template or ChatTemplate(None)
        self.eos_ids = {EOS}
        self.calls = []
        self.gate = None
        self.fail_at = None
        self.delay = 0.0
        self.yielded = 0
        self.closed = 0
        self.alive = True

    def encode(self, text, special=True):
        out, i = [], 0
        while i < len(text):
            for s, tid in SPECIAL.items():
                if special and text.startswith(s, i):
                    out.append(tid)
                    i += len(s)
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids):
        return _bytes_decode(ids)

    def generate(self, prompt_ids, settings, max_new, request_id=None, return_logits=False):
        self.calls.append({"prompt_ids": list(prompt_ids), "settings": settings,
                           "max_new": max_new, "request_id": request_id})
        try:
            for i, tok in enumerate(self.script[:max_new]):
                if self.gate is not None:
                    assert self.gate.wait(10)
                if self.fail_at == i:
                    raise RuntimeError("pipeline exploded")
                if self.delay:
                    time.sleep(self.delay)
                self.yielded += 1
                yield tok, (np.full(4, float(i), np.float32) if return_logits else None)
        finally:
            self.closed += 1

    def health(self):
        return {"ok": self.alive}


def _script(text, eos=True):
    return list(text.encode("utf-8")) + ([EOS] if eos else [])


@pytest.fixture
def make_server():
    servers = []

    def _make(engine, **kw):
        kw.setdefault("api_key", KEY)
        app = BigBrainApp(engine, **kw)
        httpd, port, th = serve_in_thread(app, host="127.0.0.1", port=0)
        servers.append((app, httpd))
        return app, port

    yield _make
    for app, httpd in servers:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def _req(port, method, path, body=None, key=KEY, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    h = dict(headers or {})
    if key:
        h["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, dict(r.getheaders()), raw


def _json(port, method, path, body=None, **kw):
    status, headers, raw = _req(port, method, path, body, **kw)
    return status, headers, json.loads(raw) if raw else None


def _sse(raw: bytes):
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                payload = line[6:]
                events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


CHAT = {"model": "x", "messages": [{"role": "user", "content": "Hi"}]}


def test_health_is_open_and_models_need_auth(make_server):
    eng = FakeEngine(_script("x"))
    app, port = make_server(eng)
    status, _, body = _json(port, "GET", "/health", key=None)
    assert status == 200 and body["status"] == "ok" and body["model"] == "fake-27b"
    assert _json(port, "GET", "/v1/models", key=None)[0] == 401
    assert _json(port, "GET", "/v1/models", key="wrong")[0] == 401
    status, _, body = _json(port, "GET", "/v1/models")
    assert status == 200 and body["data"][0]["id"] == "fake-27b"
    status, _, _ = _json(port, "GET", "/v1/models", key=None, headers={"x-api-key": KEY})
    assert status == 200
    eng.alive = False
    status, _, body = _json(port, "GET", "/health", key=None)
    assert status == 503 and body["status"] == "degraded"


def test_chat_completion_non_stream(make_server):
    eng = FakeEngine(_script("Hello \U0001F600!"))
    app, port = make_server(eng)
    status, _, body = _json(port, "POST", "/v1/chat/completions",
                            dict(CHAT, temperature=0.5, top_p=0.9, top_k=7, seed=42, min_p=0.1))
    assert status == 200, body
    assert body["object"] == "chat.completion" and body["model"] == "fake-27b"
    ch = body["choices"][0]
    assert ch["message"] == {"role": "assistant", "content": "Hello \U0001F600!"}
    assert ch["finish_reason"] == "stop"
    prompt = eng.calls[0]["prompt_ids"]
    assert prompt == eng.encode("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
    assert body["usage"] == {"prompt_tokens": len(prompt),
                             "completion_tokens": len(_script("Hello \U0001F600!")),
                             "total_tokens": len(prompt) + len(_script("Hello \U0001F600!"))}
    s = eng.calls[0]["settings"]
    assert (s.temperature, s.top_p, s.top_k, s.seed, s.min_p) == (0.5, 0.9, 7, 42, 0.1)


def test_chat_completion_stream(make_server):
    text = "Stream \u00e9\U0001F680 ok"
    eng = FakeEngine(_script(text))
    app, port = make_server(eng)
    status, headers, raw = _req(port, "POST", "/v1/chat/completions",
                                dict(CHAT, stream=True, stream_options={"include_usage": True}))
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    ev = _sse(raw)
    assert ev[-1] == "[DONE]"
    chunks = ev[:-1]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(c["choices"][0]["delta"].get("content", "")
                      for c in chunks if c["choices"])
    assert content == text
    assert not any("\ufffd" in c["choices"][0]["delta"].get("content", "")
                   for c in chunks if c["choices"])
    finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finals[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == len(_script(text))
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_stop_strings_and_length(make_server):
    eng = FakeEngine(_script("alpha beta<|im_end|> gamma", eos=False))
    app, port = make_server(eng)
    status, _, body = _json(port, "POST", "/v1/chat/completions",
                            dict(CHAT, stop=["<|im_end|>", "zzz"]))
    assert body["choices"][0]["message"]["content"] == "alpha beta"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert eng.closed == 1 and eng.yielded < len(eng.script)  # stopped early
    status, _, body = _json(port, "POST", "/v1/chat/completions", dict(CHAT, max_tokens=5))
    assert body["choices"][0]["message"]["content"] == "alpha"
    assert body["choices"][0]["finish_reason"] == "length"
    assert eng.calls[-1]["max_new"] == 5


def test_max_tokens_clamped_to_context_and_overlong_prompt_refused(make_server):
    eng = FakeEngine(_script("x" * 100, eos=False), max_context=64)
    app, port = make_server(eng)
    status, _, body = _json(port, "POST", "/v1/chat/completions", dict(CHAT, max_tokens=500))
    prompt_len = len(eng.calls[0]["prompt_ids"])
    assert status == 200 and eng.calls[0]["max_new"] == 64 - prompt_len
    status, _, body = _json(port, "POST", "/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "y" * 80}]})
    assert status == 400 and "context" in body["error"]["message"]


@pytest.mark.parametrize("body,frag", [
    ({"messages": []}, "messages"),
    ({"messages": [{"role": "user", "content": "x"}], "temperature": -1}, "temperature"),
    ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
    ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, "max_tokens"),
    ({"messages": [{"role": "user", "content": "x"}], "stop": [1]}, "stop"),
    ({"messages": [{"role": "user", "content": "x"}], "logprobs": True}, "logprobs"),
    ({"messages": [{"role": "user", "content": "x"}],
      "response_format": {"type": "json_object"}}, "response_format"),
])
def test_bad_requests_are_400(make_server, body, frag):
    app, port = make_server(FakeEngine(_script("x")))
    status, _, out = _json(port, "POST", "/v1/chat/completions", body)
    assert status == 400 and frag in out["error"]["message"]
    status, _, out = _json(port, "POST", "/v1/chat/completions", b"{not json")
    assert status == 400


def test_queue_full_is_503_with_retry_after(make_server):
    eng = FakeEngine(_script("abc"))
    eng.gate = threading.Event()
    app, port = make_server(eng, queue_size=1, retry_after=7)
    results = {}

    def call(name):
        results[name] = _json(port, "POST", "/v1/chat/completions", CHAT)

    a = threading.Thread(target=call, args=("a",))
    a.start()
    deadline = time.monotonic() + 5
    while not app.scheduler.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    b = threading.Thread(target=call, args=("b",))
    b.start()
    while app.scheduler.depth() < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    status, headers, body = _json(port, "POST", "/v1/chat/completions", CHAT)
    assert status == 503 and headers["Retry-After"] == "7"
    assert "busy" in body["error"]["message"]
    eng.gate.set()
    a.join(10)
    b.join(10)
    assert results["a"][0] == 200 and results["b"][0] == 200
    # FIFO: a was served first
    assert [c["request_id"] for c in eng.calls] == sorted(c["request_id"] for c in eng.calls)


def test_client_disconnect_mid_stream_cancels_generation(make_server):
    eng = FakeEngine(_script("z" * 60, eos=False), max_context=200)
    eng.gate = threading.Event()
    eng.delay = 0.03  # a real pipeline takes time per token
    app, port = make_server(eng)
    s = socket.create_connection(("127.0.0.1", port))
    body = json.dumps(dict(CHAT, stream=True, max_tokens=60)).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
              b"Authorization: Bearer " + KEY.encode() + b"\r\nContent-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    eng.gate.set()
    s.recv(200)  # the stream has started
    s.close()
    deadline = time.monotonic() + 10
    while eng.closed == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert eng.closed == 1
    assert eng.yielded < 60


def test_engine_error_is_500_and_an_sse_error_event(make_server):
    eng = FakeEngine(_script("abcdef"))
    eng.fail_at = 3
    app, port = make_server(eng)
    status, _, body = _json(port, "POST", "/v1/chat/completions", CHAT)
    assert status == 500 and "pipeline exploded" in body["error"]["message"]
    status, _, raw = _req(port, "POST", "/v1/chat/completions", dict(CHAT, stream=True))
    ev = _sse(raw)
    assert any(isinstance(e, dict) and "error" in e for e in ev)
    # the scheduler survives and serves the next request
    eng.fail_at = None
    assert _json(port, "POST", "/v1/chat/completions", CHAT)[0] == 200


def test_text_completions(make_server):
    eng = FakeEngine(_script("world"))
    app, port = make_server(eng)
    status, _, body = _json(port, "POST", "/v1/completions", {"prompt": "Hello ", "max_tokens": 9})
    assert status == 200 and body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "world"
    assert eng.calls[0]["prompt_ids"] == list(b"Hello ")
    status, _, raw = _req(port, "POST", "/v1/completions", {"prompt": "Hi", "stream": True})
    ev = _sse(raw)
    assert "".join(e["choices"][0]["text"] for e in ev[:-1] if e["choices"]) == "world"


def test_bigbrain_generate_debug_endpoint_returns_tokens_and_logits(make_server):
    eng = FakeEngine(_script("abc", eos=False))
    app, port = make_server(eng)
    status, _, body = _json(port, "POST", "/v1/bigbrain/generate",
                            {"prompt_ids": [5, 6, 7], "max_tokens": 3, "temperature": 0,
                             "ignore_eos": True, "return_logits": True})
    assert status == 200, body
    assert body["tokens"] == list(b"abc") and body["prompt_ids"] == [5, 6, 7]
    lg = body["logits"]
    arr = np.frombuffer(base64.b64decode(lg["b64"]), dtype=lg["dtype"]).reshape(lg["shape"])
    np.testing.assert_array_equal(arr[:, 0], [0, 1, 2])
    assert eng.calls[0]["settings"].greedy


def test_ignore_eos_runs_past_eos(make_server):
    eng = FakeEngine(list(b"ab") + [EOS] + list(b"cd"))
    app, port = make_server(eng)
    status, _, body = _json(port, "POST", "/v1/bigbrain/generate",
                            {"prompt_ids": [1], "max_tokens": 5, "ignore_eos": True})
    assert body["tokens"] == list(b"ab") + [EOS] + list(b"cd")
    status, _, body = _json(port, "POST", "/v1/bigbrain/generate",
                            {"prompt_ids": [1], "max_tokens": 5})
    assert body["tokens"] == list(b"ab") + [EOS] and body["finish_reason"] == "stop"


def test_generation_defaults_apply_when_the_request_omits_them(make_server):
    eng = FakeEngine(_script("q"))
    app, port = make_server(eng, generation_defaults={"temperature": 0.6, "top_k": 20, "top_p": 0.95})
    _json(port, "POST", "/v1/chat/completions", CHAT)
    s = eng.calls[0]["settings"]
    assert (s.temperature, s.top_k, s.top_p) == (0.6, 20, 0.95)
    _json(port, "POST", "/v1/chat/completions", dict(CHAT, temperature=0, top_k=-1))
    s = eng.calls[1]["settings"]
    assert s.greedy and s.top_k == 0


def test_parity_check_http_mode_against_a_running_server(make_server, tmp_path, monkeypatch, capsys):
    """parity_check http: the reference dump's prompt ids go to
    /v1/bigbrain/generate (greedy, ignore_eos, return_logits) and the answer is
    compared token-for-token and logit-for-logit."""
    from bigbrain.parity_check import main as parity_main
    from bigbrain.worker import write_dump
    script = list(b"abcd") + [EOS] + list(b"ef")
    eng = FakeEngine(script)
    app, port = make_server(eng)
    monkeypatch.setenv("BB_PARITY_KEY", KEY)
    rows = np.stack([np.full(4, float(i), np.float32) for i in range(len(script))])
    ref = str(tmp_path / "ref.npz")
    write_dump(ref, [5, 6, 7], script, rows)
    url = f"http://127.0.0.1:{port}"
    out = str(tmp_path / "srv.npz")
    rc = parity_main(["http", "--url", url, "--reference", ref,
                      "--api-key-env", "BB_PARITY_KEY", "--out", out])
    assert rc == 0, capsys.readouterr().out
    assert "VERDICT: PASS" in capsys.readouterr().out
    assert eng.calls[-1]["prompt_ids"] == [5, 6, 7] and eng.calls[-1]["settings"].greedy
    assert eng.calls[-1]["max_new"] == len(script)  # ran past the EOS (ignore_eos)
    bad_rows = rows.copy()
    bad_rows[2, 1] += 0.5
    write_dump(ref, [5, 6, 7], script, bad_rows)
    assert parity_main(["http", "--url", url, "--reference", ref,
                        "--api-key-env", "BB_PARITY_KEY"]) == 1
    assert "max |logit diff|" in capsys.readouterr().out


# --------------------------------------------------------------------------
# end to end: HTTP -> server -> real SliceWorker pipeline (CPU fake model)
# --------------------------------------------------------------------------

class _LetterTokenizer:
    """id i <-> letter chr(97+i) for the fake model's 12-token vocab."""

    def encode(self, text, add_bos=False, encode_special_tokens=False):
        import torch
        return torch.tensor([[ord(c) - 97 for c in text]])

    def decode(self, ids, decode_special_tokens=False):
        return "".join(chr(97 + int(i)) for i in ids.reshape(-1).tolist())


def test_end_to_end_over_the_fake_pipeline(make_server, tmp_path):
    pytest.importorskip("torch")
    from test_bigbrain_worker import KINDS_HYBRID, _Cluster, reference_generate
    (tmp_path / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m.content }}{% endfor %}", encoding="utf-8")
    c = _Cluster(KINDS_HYBRID, [(0, 3), (3, 6)])
    try:
        c.head.tokenizer = _LetterTokenizer()
        eng = WorkerEngine(c.head, model_dir=str(tmp_path), model_name="fake-hybrid")
        eng.eos_ids = set()  # run to max_tokens
        app, port = make_server(eng)
        status, _, body = _json(port, "POST", "/v1/chat/completions",
                                {"messages": [{"role": "user", "content": "bfcid"}],
                                 "max_tokens": 6, "temperature": 0})
        assert status == 200, body
        ref, _ = reference_generate([1, 5, 2, 8, 3], 6)
        assert body["choices"][0]["message"]["content"] == "".join(chr(97 + t) for t in ref)
        status, _, raw = _req(port, "POST", "/v1/chat/completions",
                              {"messages": [{"role": "user", "content": "bfcid"}],
                               "max_tokens": 6, "temperature": 0, "stream": True})
        ev = _sse(raw)
        streamed = "".join(e["choices"][0]["delta"].get("content", "")
                           for e in ev[:-1] if e["choices"])
        assert streamed == "".join(chr(97 + t) for t in ref)
        status, _, body = _json(port, "POST", "/v1/bigbrain/generate",
                                {"prompt_ids": [1, 5, 2, 8, 3], "max_tokens": 6,
                                 "temperature": 0, "return_logits": True, "ignore_eos": True})
        ref_toks, ref_rows = reference_generate([1, 5, 2, 8, 3], 6)
        lg = body["logits"]
        rows = np.frombuffer(base64.b64decode(lg["b64"]), dtype=lg["dtype"]).reshape(lg["shape"])
        assert body["tokens"] == ref_toks
        np.testing.assert_array_equal(rows, ref_rows)
        assert eng.health()["ok"]
    finally:
        c.shutdown()
