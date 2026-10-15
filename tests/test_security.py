# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""v0.2.1 security fixes. One section per finding of the 2026-10-15 review:

M1  host and user values never reach a shell or ssh unchecked (shell injection, ssh option injection)
M2  the HTTP endpoint drops idle connections and caps how many it holds (pre-auth thread exhaustion)
M3  the pipe refuses an oversized or malformed frame, closes the link and marks the engine dead (no silent 600 s hang)
M4  the stage dir is private by default, and a stage dir or node.env someone else could write is refused
M5  the API key never appears on a command line or in the controller's output
M6  the chat template is never rendered with a jinja2 older than 3.1.6
plus the low-severity items fixed in the same release (loopback-only --insecure-no-auth, a minimum key length, /health detail,
stop-string length, ./site.toml).

Loopback sockets and fakes only: no GPU, no ssh, no network.
"""
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from bigbrain import controller as ctl
from bigbrain import site

ROOT = Path(__file__).resolve().parent.parent
PAYLOAD_HOST = "10.0.0.2;touch /tmp/pwned;#"            # the review's proof of concept
OPTION_USER = "-oProxyCommand=sh -c 'touch /tmp/pwned'"   # a user that ssh would read as an option
KEY = "k3y-" + "0123456789abcdef" * 2                     # a test key long enough for the minimum length


def _node(rank, host, gpu, start, end, in_port, **kw):
    return ctl.NodeSpec(rank=rank, host=host, gpu=gpu, slice_start=start, slice_end=end, in_port=in_port, **kw)


def _two_node_plan(**kw):
    return ctl.ClusterPlan(model_dir="/m", nodes=[_node(0, "alpha.lan", 0, 0, 10, 0, user="bigbrain"),
                                                   _node(1, "10.0.0.12", 0, 10, 20, 16900, user="bigbrain")],
                           total_blocks=20, token_port=16901, **kw)


# == M1: hosts and users ===========================================================================================================
class TestM1HostAndUserInjection:
    def test_a_host_with_shell_metacharacters_is_refused(self):
        with pytest.raises(ValueError, match="host"):
            _node(1, PAYLOAD_HOST, 0, 10, 20, 16900)
        for bad in ("$(id)", "`id`", "a b", "-oProxyCommand=x", "", "h0|id", "h0\nid", "fe80::1%$(id)"):
            with pytest.raises(ValueError):
                _node(0, bad, 0, 0, 10, 0)

    def test_a_user_that_ssh_would_read_as_an_option_is_refused(self):
        with pytest.raises(ValueError, match="user"):
            _node(0, "10.0.0.11", 0, 0, 10, 0, user=OPTION_USER)
        for bad in ("-l", "a;b", "a b", "$(id)", ""):
            with pytest.raises(ValueError):
                _node(0, "10.0.0.11", 0, 0, 10, 0, user=bad)

    def test_ordinary_hosts_and_users_are_accepted(self):
        for host in ("h0", "alpha", "alpha.lan", "gpu-box-2.example.org", "10.0.0.11", "::1", "fe80::1%eth0", "my_host"):
            assert _node(0, host, 0, 0, 10, 0, user="bigbrain").host == host
        for user in ("bigbrain", "bb_2", "a.b-c", "Alice", "_svc"):
            assert _node(0, "h0", 0, 0, 10, 0, user=user).user == user

    def test_a_host_changed_after_construction_is_reported_by_validate(self):
        plan = _two_node_plan()
        plan.nodes[1].host = PAYLOAD_HOST
        assert any("host" in p for p in plan.validate())
        plan.nodes[1].host, plan.nodes[1].user = "10.0.0.12", OPTION_USER
        assert any("user" in p for p in plan.validate())

    def test_every_host_in_a_node_command_is_one_shell_word(self):
        """Even a host that slipped past validation cannot break out of the remote command: --next-host and --token-host are
        quoted, so shlex.split hands each back as exactly one word."""
        plan = _two_node_plan()
        c0, c1 = plan.build_commands()
        w0, w1 = shlex.split(c0), shlex.split(c1)
        assert w0[w0.index("--next-host") + 1] == "10.0.0.12" and w1[w1.index("--token-host") + 1] == "alpha.lan"
        plan.nodes[1].host = PAYLOAD_HOST                       # bypass __post_init__: the quoting is the second line of defence
        plan.nodes[0].host = "$(touch /tmp/pwned)"
        c0, c1 = plan.build_commands()
        w0, w1 = shlex.split(c0), shlex.split(c1)
        assert w0[w0.index("--next-host") + 1] == PAYLOAD_HOST and w0[w0.index("--next-host") + 2] == "--next-port"
        assert w1[w1.index("--token-host") + 1] == "$(touch /tmp/pwned)"
        assert "--next-host 10.0.0.2;touch" not in c0

    def test_the_cli_refuses_a_bad_host(self):
        with pytest.raises((SystemExit, ValueError)):
            ctl.plan_from_args(["--model-dir", "/m", "--total-blocks", "20", "--gpus", f"{PAYLOAD_HOST}:0", "--gpus", "h1:0"])
        with pytest.raises((SystemExit, ValueError)):
            ctl.plan_from_args(["--model-dir", "/m", "--total-blocks", "20", "--token-port", "7000",
                                "--node", "$(id):0:0:10:0", "--node", "h1:0:10:20:6001"])

    def test_every_ssh_argv_ends_its_options_with_a_double_dash(self, monkeypatch, tmp_path):
        """`ssh ... -- user@host cmd`: whatever the destination is, ssh never reads it as an option."""
        seen = []

        class _Done:
            returncode, stdout, stderr = 0, "{}", ""

        def fake_run(argv, *a, **kw):
            seen.append(list(argv))
            return _Done()

        def fake_popen(argv, *a, **kw):
            seen.append(list(argv))
            return object()

        monkeypatch.setattr(ctl.subprocess, "run", fake_run)
        monkeypatch.setattr(ctl.subprocess, "Popen", fake_popen)
        ctl._default_stager("10.0.0.11", "bigbrain", "~/.cache/bigbrain/stage/bigbrain/pipe.py", "x")
        ctl._ssh_start("10.0.0.11", "bigbrain", "true")
        ctl._ssh_capture("bigbrain", "10.0.0.11", "true")
        assert ctl._read_model_config(str(tmp_path / "no-such-model"), "10.0.0.11", "bigbrain") == {}   # not local: over ssh
        from bigbrain import build_registry as br
        monkeypatch.setattr(br.subprocess, "run", fake_run)
        br.measure("10.0.0.11", "/data/models", user="bigbrain")
        assert len(seen) == 5
        for argv in seen:
            assert argv[0] == "ssh" and "--" in argv, argv
            i = argv.index("--")
            assert argv[i + 1] == "bigbrain@10.0.0.11", argv
            assert all(not a.startswith("bigbrain@") for a in argv[:i]), argv

    def test_the_ssh_helpers_refuse_a_bad_destination_before_running_anything(self, monkeypatch):
        monkeypatch.setattr(ctl.subprocess, "run", lambda *a, **k: pytest.fail("ssh must not run"))
        monkeypatch.setattr(ctl.subprocess, "Popen", lambda *a, **k: pytest.fail("ssh must not run"))
        with pytest.raises(ValueError):
            ctl._default_stager(PAYLOAD_HOST, "bigbrain", "/x/y", "x")
        with pytest.raises(ValueError):
            ctl._ssh_start("10.0.0.11", OPTION_USER, "true")
        with pytest.raises(ValueError):
            ctl._ssh_capture("bigbrain", "-oProxyCommand=id", "true")

    def test_the_site_file_refuses_a_bad_address_or_user(self, tmp_path, monkeypatch):
        for body in (f'[hosts.alpha]\naddress = "{PAYLOAD_HOST}"\n', '[hosts.alpha]\naddress = "10.0.0.11"\nuser = "-oX=y"\n',
                     '[ssh]\nuser = "a;b"\n'):
            f = tmp_path / "site.toml"
            f.write_text(body, encoding="utf-8")
            monkeypatch.setenv("BIGBRAIN_SITE", str(f))
            with pytest.raises(site.SiteError):
                site.load()
            site._cache.clear()

    def test_ssh_target_never_returns_an_option(self):
        assert site.ssh_target("bigbrain@10.0.0.11") == "bigbrain@10.0.0.11"
        assert site.ssh_target("10.0.0.11") == "10.0.0.11"
        for bad in ("-oProxyCommand=id", "bigbrain@-x", "a b@h", f"bigbrain@{PAYLOAD_HOST}"):
            with pytest.raises(site.SiteError):
                site.ssh_target(bad)

    def test_build_registry_skips_a_server_url_whose_host_is_not_a_host(self, capsys):
        from bigbrain import build_registry as br
        cfg = {"tabby": {"servers": {"A": {"url": "http://$(id)/", "model_dir": "/data/models"},
                                     "B": {"url": "http://10.0.0.12:5000", "model_dir": "/srv/models"}}}}
        assert [n["host"] for n in br.nodes_from_config(cfg)] == ["10.0.0.12"]
        assert "$(id)" in capsys.readouterr().err

    def test_the_repo_tools_put_a_double_dash_before_the_destination(self):
        """deploy/release.py, deploy/verify_stage.py and patches/verify.py build their ssh argv the same way."""
        for rel in ("deploy/release.py", "deploy/verify_stage.py", "patches/verify.py"):
            p = ROOT / rel
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8")
            uses = [line for line in text.splitlines() if "SSH, " in line and "[*" in line]
            assert uses, rel
            for line in uses:
                assert 'SSH, "--",' in line, (rel, line)


# == M2: idle and excess HTTP connections ==========================================================================================
class _Engine:
    """The least a BigBrainApp needs: a model that answers 'ok' to one prompt."""
    model_name = "fake"
    max_context = 64

    def __init__(self, dead=None):
        from bigbrain.server import ChatTemplate
        self.template = ChatTemplate(None)
        self.eos_ids = {0}
        self.dead = dead

    def encode(self, text, special=True):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(t) for t in ids).decode("utf-8", "replace")

    def generate(self, prompt_ids, settings, max_new, request_id=None, return_logits=False):
        for t in b"ok":
            yield t, None
        yield 0, None

    def health(self):
        return {"ok": self.dead is None, "dead": self.dead}


@pytest.fixture
def http_server():
    from bigbrain.server import BigBrainApp, serve_in_thread
    made = []

    def make(engine=None, **kw):
        app = BigBrainApp(engine or _Engine(), api_key=KEY)
        httpd, port, _ = serve_in_thread(app, host="127.0.0.1", port=0, **kw)
        made.append((app, httpd))
        return app, httpd, port

    yield make
    for app, httpd in made:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def _get(port, path, key=None, timeout=10):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("GET", path, headers={"Authorization": f"Bearer {key}"} if key else {})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, json.loads(body or b"{}")


def _idle(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.settimeout(10)
    return s


class TestM2HttpConnections:
    def test_an_idle_connection_is_dropped_after_the_timeout(self, http_server):
        _, httpd, port = http_server(conn_timeout_s=0.5)
        s = _idle(port)
        t0 = time.monotonic()
        try:
            assert s.recv(1) == b""                     # the server closed it; before v0.2.1 this blocked forever
        finally:
            s.close()
        assert time.monotonic() - t0 < 5.0

    def test_connections_beyond_the_cap_are_refused_with_503_and_the_slots_come_back(self, http_server):
        _, httpd, port = http_server(max_connections=2, conn_timeout_s=30)
        held = [_idle(port), _idle(port)]
        try:
            deadline = time.monotonic() + 5
            while httpd.active_connections < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert httpd.active_connections == 2
            third = _idle(port)
            try:
                assert third.recv(4096).startswith(b"HTTP/1.0 503")   # refused before any handler thread was spent on it
            finally:
                third.close()
        finally:
            for s in held:
                s.close()
        deadline = time.monotonic() + 5
        while httpd.active_connections and time.monotonic() < deadline:
            time.sleep(0.02)
        assert httpd.active_connections == 0
        assert _get(port, "/health")[0] == 200            # the slots were released: a normal request is served again

    def test_the_defaults_are_a_timeout_and_a_cap(self):
        from bigbrain import server as srv
        a = srv.build_arg_parser().parse_args(
            "--model-dir /m --slice 0 1 --rank 0 --total 1".split())
        assert a.max_connections == srv.DEFAULT_MAX_CONNECTIONS == 64
        assert a.conn_timeout == srv.DEFAULT_CONN_TIMEOUT_S == 60.0
        httpd = srv.make_http_server(srv.BigBrainApp(_Engine(), api_key=KEY), "127.0.0.1", 0)
        try:
            assert httpd.max_connections == 64 and httpd.conn_timeout_s == 60.0
        finally:
            httpd.server_close()
            httpd.app.close()


# == M3: wire frames ===============================================================================================================
def _raw_and_link(**kw):
    """A raw client socket and a BigBrainLink on the accepting side: hand-made bytes go on the wire."""
    from bigbrain.pipe import BigBrainLink
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(1)
    raw = socket.create_connection(lsock.getsockname())
    conn, _ = lsock.accept()
    lsock.close()
    link = BigBrainLink.from_socket(conn)
    for k, v in kw.items():
        setattr(link, k, v)
    return raw, link


def _hdr(msgtype, *, batch=0, seq_len=0, hidden=0, dtype=0, flags=1, payload_len=0):
    """A 55-byte v2 header with every field chosen by the caller (struct, not pack_header: no client-side checks)."""
    import struct
    from bigbrain.pipe import HEADER_FMT
    return struct.pack(HEADER_FMT, b"BGBN", 2, msgtype, 1, batch, seq_len, hidden, dtype, flags, payload_len, 0, 0)


def _closed_by_peer(raw, within=5.0):
    raw.settimeout(within)
    try:
        while True:
            chunk = raw.recv(65536)
            if not chunk:
                return True                  # EOF: the link closed the connection
    except (ConnectionResetError, ConnectionAbortedError):
        return True
    except socket.timeout:
        return False


class TestM3Frames:
    def test_a_2_pow_40_payload_len_is_refused_before_any_allocation(self):
        from bigbrain.pipe import MSG_ACTIVATION, DT_FP16, BigBrainProtocolError
        raw, link = _raw_and_link()
        try:
            raw.sendall(_hdr(MSG_ACTIVATION, batch=1, seq_len=1, hidden=1, dtype=DT_FP16, payload_len=2 ** 40))
            t0 = time.monotonic()
            with pytest.raises(BigBrainProtocolError, match="limit"):       # not MemoryError (the review's PoC)
                link.recv_message()
            assert time.monotonic() - t0 < 2.0
            assert link.is_dead and _closed_by_peer(raw)
        finally:
            raw.close()
            link.close()

    def test_a_shape_whose_size_wraps_in_int64_is_refused(self):
        """batch=4, seq_len=2**31, hidden=2**31, fp32 and an empty payload: np.prod wrapped to 0 and the size check passed."""
        from bigbrain.pipe import MSG_ACTIVATION, DT_FP32, BigBrainProtocolError
        raw, link = _raw_and_link()
        try:
            raw.sendall(_hdr(MSG_ACTIVATION, batch=4, seq_len=2 ** 31, hidden=2 ** 31, dtype=DT_FP32, payload_len=0))
            with pytest.raises(BigBrainProtocolError, match="does not match"):    # a protocol error, not a ValueError
                link.recv_message()
            assert link.is_dead and _closed_by_peer(raw)
        finally:
            raw.close()
            link.close()

    @pytest.mark.parametrize("case", ["zero_batch", "zero_hidden", "control_over_cap", "reset_over_cap", "heartbeat_payload",
                                      "token_dtype", "unknown_type"])
    def test_other_malformed_headers_are_refused_before_the_payload_is_read(self, case):
        from bigbrain import pipe
        hdr = {
            "zero_batch": _hdr(pipe.MSG_ACTIVATION, batch=0, seq_len=1, hidden=4, dtype=pipe.DT_FP16, payload_len=0),
            "zero_hidden": _hdr(pipe.MSG_ACTIVATION, batch=1, seq_len=1, hidden=0, dtype=pipe.DT_FP16, payload_len=0),
            "control_over_cap": _hdr(pipe.MSG_CONTROL, payload_len=pipe.MAX_CONTROL_BYTES + 1),
            "reset_over_cap": _hdr(pipe.MSG_RESET, payload_len=pipe.MAX_SMALL_PAYLOAD + 1),
            "heartbeat_payload": _hdr(pipe.MSG_HEARTBEAT, payload_len=pipe.MAX_SMALL_PAYLOAD + 1),
            "token_dtype": _hdr(pipe.MSG_TOKEN, batch=1, seq_len=1, hidden=1, dtype=pipe.DT_FP32, payload_len=4),
            "unknown_type": _hdr(99, payload_len=8),
        }[case]
        raw, link = _raw_and_link()
        try:
            raw.sendall(hdr)                 # no payload follows: a check after reading it would hang until the watchdog
            t0 = time.monotonic()
            with pytest.raises(pipe.BigBrainProtocolError):
                link.recv_message()
            assert time.monotonic() - t0 < 2.0 and link.is_dead
        finally:
            raw.close()
            link.close()

    def test_a_frame_over_the_links_own_limit_is_refused_and_one_at_it_passes(self):
        import numpy as np
        from bigbrain.pipe import MSG_ACTIVATION, DT_FP16, BigBrainProtocolError
        arr = np.arange(8, dtype=np.float16).reshape(1, 2, 4)
        raw, link = _raw_and_link(max_frame_bytes=arr.nbytes)
        try:
            raw.sendall(_hdr(MSG_ACTIVATION, batch=1, seq_len=2, hidden=4, dtype=DT_FP16, payload_len=arr.nbytes)
                        + arr.tobytes())
            np.testing.assert_array_equal(link.recv_message().array, arr)
            raw.sendall(_hdr(MSG_ACTIVATION, batch=1, seq_len=2, hidden=5, dtype=DT_FP16, payload_len=20) + b"\0" * 20)
            with pytest.raises(BigBrainProtocolError, match="limit"):
                link.recv_message()
            assert link.is_dead
        finally:
            raw.close()
            link.close()

    def test_mismatched_chunks_and_deep_json_are_protocol_errors(self):
        import numpy as np
        from bigbrain.pipe import MSG_ACTIVATION, MSG_CONTROL, DT_FP16, BigBrainProtocolError
        a, b = np.zeros((1, 2, 4), np.float16), np.zeros((1, 2, 8), np.float16)
        raw, link = _raw_and_link()
        try:
            raw.sendall(_hdr(MSG_ACTIVATION, batch=1, seq_len=2, hidden=4, dtype=DT_FP16, flags=0, payload_len=a.nbytes)
                        + a.tobytes()
                        + _hdr(MSG_ACTIVATION, batch=1, seq_len=2, hidden=8, dtype=DT_FP16, flags=1, payload_len=b.nbytes)
                        + b.tobytes())
            with pytest.raises(BigBrainProtocolError):                          # np.concatenate's ValueError, converted
                link.recv_message()
            assert link.is_dead
        finally:
            raw.close()
            link.close()
        deep = b"[" * 100000 + b"]" * 100000
        raw, link = _raw_and_link()
        try:
            raw.sendall(_hdr(MSG_CONTROL, payload_len=len(deep)) + deep)
            with pytest.raises(BigBrainProtocolError):                          # json's RecursionError, converted
                link.recv_message()
            assert link.is_dead
        finally:
            raw.close()
            link.close()

    def test_the_frame_limit_comes_from_the_model_and_covers_a_real_frame(self):
        from bigbrain import pipe
        lim = pipe.frame_limit(hidden_size=2880, max_tokens=2048, vocab_size=201088)
        assert lim >= 2048 * 2880 * 4                     # a full fp32 prefill chunk of gpt-oss-120b fits
        assert lim >= 201088 * 4                          # one fp32 logits row fits
        assert lim < 2 ** 40 and lim >= pipe.MIN_FRAME_LIMIT
        assert pipe.frame_limit(hidden_size=4, max_tokens=1) == pipe.MIN_FRAME_LIMIT
        assert pipe.BigBrainLink().max_frame_bytes == pipe.DEFAULT_MAX_FRAME_BYTES

    def test_a_node_narrows_its_links_to_the_models_frame_limit(self):
        from types import SimpleNamespace
        from bigbrain import pipe
        from bigbrain.worker import SliceWorker
        inbound, tok = pipe.BigBrainLink(), pipe.BigBrainLink()
        cfg = SimpleNamespace(hidden_size=4096, vocab_size=32000)
        none = lambda *a, **k: None                      # noqa: E731  (no exllamav3 here)
        w = SliceWorker.from_parts(model=SimpleNamespace(), cache=SimpleNamespace(), rank=1, total=2, config=cfg,
                                   inbound=inbound, token_link=tok, prefill_chunk=2048, slots=4, max_draft=4, advance_fn=none)
        want = pipe.frame_limit(4096, 2048, vocab_size=32000)
        assert w.max_frame_bytes == want and inbound.max_frame_bytes == want and tok.max_frame_bytes == want
        w2 = SliceWorker.from_parts(model=SimpleNamespace(), cache=SimpleNamespace(), rank=1, total=2, config=cfg,
                                    inbound=pipe.BigBrainLink(), max_frame_mib=3, advance_fn=none)
        assert w2.max_frame_bytes == 3 * 1024 * 1024 and w2.inbound.max_frame_bytes == 3 * 1024 * 1024

    def test_a_worker_reader_thread_never_dies_silently(self):
        """Any exception in the receive loop (not only a protocol error) becomes a 'dead' message for serve()."""
        import queue
        from types import SimpleNamespace
        from bigbrain.worker import SliceWorker

        def boom(*a, **k):
            raise MemoryError("2**40 bytes")
        fake = SimpleNamespace(inbound=SimpleNamespace(recv_message=boom, close=lambda: None), _inq=queue.Queue(), rank=3)
        SliceWorker._recv_loop(fake)
        msg = fake._inq.get_nowait()
        assert msg.kind == "dead" and "MemoryError" in msg.reason

    def _engine(self, link):
        from types import SimpleNamespace
        from bigbrain.engine import PipelineEngine
        w = SimpleNamespace(is_first=True, n_slots=2, max_num_tokens=64, total=2, token_link=link,
                            stats=SimpleNamespace(wait_s=0.0))
        eng = PipelineEngine(w)
        eng.ready.set()
        th = threading.Thread(target=eng._recv_loop, daemon=True)
        th.start()
        return eng, th

    def test_a_malformed_frame_on_the_token_link_marks_the_engine_dead_at_once(self):
        from bigbrain.engine import EngineDead
        from bigbrain.pipe import MSG_ACTIVATION, DT_FP16
        from bigbrain.worker import SamplingSettings
        raw, link = _raw_and_link()
        try:
            eng, th = self._engine(link)
            raw.sendall(_hdr(MSG_ACTIVATION, batch=1, seq_len=1, hidden=1, dtype=DT_FP16, payload_len=2 ** 40))
            th.join(timeout=5)
            assert not th.is_alive() and eng.dead and "limit" in eng.dead
            assert eng.health()["ok"] is False
            with pytest.raises(EngineDead):                                     # a new request fails at once, not after 600 s
                eng.submit([1, 2, 3], SamplingSettings(), 4)
        finally:
            raw.close()
            link.close()

    def test_any_exception_in_the_engine_reader_marks_the_engine_dead(self):
        from types import SimpleNamespace

        def boom(*a, **k):
            raise ValueError("cannot reshape array of size 0")
        eng, th = self._engine(SimpleNamespace(recv_message=boom, close=lambda: None, is_dead=False))
        th.join(timeout=5)
        assert not th.is_alive() and eng.dead and "ValueError" in eng.dead and eng.health()["ok"] is False

    def test_a_dead_link_in_single_sequence_mode_is_degraded_health_and_a_fast_503(self, http_server):
        import http.client
        from types import SimpleNamespace
        from bigbrain.pipe import BigBrainLink
        from bigbrain.server import WorkerEngine
        link = BigBrainLink()
        link._dead = True                                   # what a malformed frame leaves behind
        we = WorkerEngine.__new__(WorkerEngine)
        we.batch, we.w = None, SimpleNamespace(token_link=link, outbound=None, total=2, recurrent=False)
        assert we.health()["ok"] is False and we.dead_reason()
        eng = _Engine(dead="the token link is down")
        eng.dead_reason = lambda: eng.dead
        _, _, port = http_server(eng)
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        t0 = time.monotonic()
        c.request("POST", "/v1/completions", body=json.dumps({"prompt": "hi", "max_tokens": 2}),
                  headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        r = c.getresponse()
        assert r.status == 503 and b"down" in r.read() and time.monotonic() - t0 < 5
        c.close()


# == M4: the stage dir and node.env ================================================================================================
SH = shutil.which("sh")
needs_sh = pytest.mark.skipif(SH is None, reason="no sh on this machine")
posix_only = pytest.mark.skipif(os.name != "posix", reason="needs POSIX permissions (chmod g+w, symlinks)")


def _fake_find(tmp_path, writable_suffix):
    """A `find` that reports `writable_suffix` as group/other-writable (what the real one does for a chmod 777 dir). Windows' MSYS
    cannot make a directory group-writable, so the refusal path is driven this way there; the POSIX tests below use chmod."""
    d = tmp_path / "fakebin"
    d.mkdir(exist_ok=True)
    f = d / "find"
    f.write_text('#!/bin/sh\ncase "$1" in *"' + writable_suffix + '") echo "$1" ;; esac\nexit 0\n', encoding="utf-8",
                 newline="\n")
    f.chmod(0o755)
    return {**os.environ, "PATH": str(d) + os.pathsep + os.environ.get("PATH", "")}


def _stage(tmp_path):
    stage = tmp_path / "stage"
    (stage / "bigbrain").mkdir(parents=True)
    (stage / "bigbrain" / "__init__.py").write_text("")
    (stage / "bigbrain" / "worker.py").write_text("print('the node ran')\n")
    if os.name == "posix":
        stage.chmod(0o700)
    return stage


def _one_node_plan(stage_root):
    return ctl.ClusterPlan(model_dir="/m", nodes=[_node(0, "h0", 0, 0, 20, 0, user="bigbrain")], total_blocks=20,
                           stage_root=stage_root, python_bin=sys.executable.replace("\\", "/"))


class TestM4StageDir:
    def test_the_default_stage_root_is_private_and_per_user(self):
        assert ctl.DEFAULT_STAGE_ROOT == "~/.cache/bigbrain/stage"
        assert ctl.ClusterPlan(model_dir="/m", nodes=[]).stage_root == ctl.DEFAULT_STAGE_ROOT
        assert ctl.plan_auto("/m", [("h0", 0)], total_blocks=4).stage_root == ctl.DEFAULT_STAGE_ROOT
        plan = ctl.plan_from_args(["--model-dir", "/m", "--total-blocks", "20", "--gpus", "h0:0"])
        assert plan.stage_root == ctl.DEFAULT_STAGE_ROOT
        import inspect
        assert inspect.signature(ctl.plan_capacity).parameters["stage_root"].default == ctl.DEFAULT_STAGE_ROOT
        for src in ("controller.py", "worker.py", "server.py"):
            assert "/tmp/bigbrain" not in (ROOT / "src" / "bigbrain" / src).read_text(encoding="utf-8"), src

    def test_a_home_relative_stage_root_expands_on_the_node_and_is_checked_before_cd(self):
        c0 = ctl.ClusterPlan(model_dir="/m", nodes=[_node(0, "h0", 0, 0, 20, 0)], total_blocks=20).build_commands(
            redirect_logs=True)[0]
        assert c0.startswith('umask 077 && bb_stage="$HOME"/.cache/bigbrain/stage && ')
        assert c0.index('[ -O "$bb_stage" ]') < c0.index('cd "$bb_stage" && ')
        assert c0.endswith('> "$bb_stage"/node_0.log 2>&1')
        t = ctl.teardown_command(ctl.DEFAULT_STAGE_ROOT)
        assert '"$HOME"/.cache/bigbrain/stage' in t and "~" not in t

    def test_a_relative_stage_root_is_refused(self):
        plan = _two_node_plan(stage_root="bigbrain-stage")
        assert any("stage" in p for p in plan.validate())
        assert _two_node_plan(stage_root="/srv/bigbrain/stage").validate() == []

    @needs_sh
    def test_the_node_command_runs_from_a_private_stage_dir(self, tmp_path):
        stage = _stage(tmp_path)
        cmd = _one_node_plan(stage.as_posix()).build_commands()[0]
        p = subprocess.run([SH, "-c", cmd], capture_output=True, text=True, timeout=60)
        assert p.returncode == 0 and "the node ran" in p.stdout, p.stderr

    @needs_sh
    def test_the_node_command_refuses_a_stage_dir_others_can_write(self, tmp_path):
        stage = _stage(tmp_path)
        cmd = _one_node_plan(stage.as_posix()).build_commands()[0]
        p = subprocess.run([SH, "-c", cmd], capture_output=True, text=True, timeout=60, env=_fake_find(tmp_path, "/stage"))
        assert p.returncode != 0 and "the node ran" not in p.stdout
        assert "refusing the stage dir" in p.stderr

    @posix_only
    @needs_sh
    def test_the_node_command_refuses_a_world_writable_or_symlinked_stage_dir(self, tmp_path):
        stage = _stage(tmp_path)
        stage.chmod(0o777)
        cmd = _one_node_plan(stage.as_posix()).build_commands()[0]
        p = subprocess.run([SH, "-c", cmd], capture_output=True, text=True, timeout=60)
        assert p.returncode != 0 and "refusing the stage dir" in p.stderr
        stage.chmod(0o700)
        link = tmp_path / "link"
        link.symlink_to(stage)
        cmd = _one_node_plan(link.as_posix()).build_commands()[0]
        p = subprocess.run([SH, "-c", cmd], capture_output=True, text=True, timeout=60)
        assert p.returncode != 0 and "refusing the stage dir" in p.stderr

    @needs_sh
    def test_the_stager_creates_a_private_dir_and_refuses_one_others_can_write(self, tmp_path, monkeypatch):
        """The stager's remote command, run here by sh instead of over ssh."""
        env = {"env": None}
        real_run = subprocess.run                     # ctl.subprocess IS the subprocess module: keep the real one

        def run_locally(argv, input=None, check=False, **kw):
            assert argv[0] == "ssh" and argv[argv.index("--") + 1] == "bigbrain@h0"
            p = real_run([SH, "-c", argv[-1]], input=input, capture_output=True, env=env["env"], timeout=60)
            if check and p.returncode:
                raise subprocess.CalledProcessError(p.returncode, argv, p.stdout, p.stderr)
            return p
        monkeypatch.setattr(ctl.subprocess, "run", run_locally)
        root = (tmp_path / "fresh" / "stage").as_posix()
        ctl._default_stager("h0", "bigbrain", root + "/bigbrain/pipe.py", "# pipe\n")
        assert (tmp_path / "fresh" / "stage" / "bigbrain" / "pipe.py").read_text() == "# pipe\n"
        if os.name == "posix":
            assert (tmp_path / "fresh" / "stage").stat().st_mode & 0o077 == 0
            assert (tmp_path / "fresh" / "stage" / "bigbrain" / "pipe.py").stat().st_mode & 0o077 == 0
        env["env"] = _fake_find(tmp_path, "/stage")
        with pytest.raises(subprocess.CalledProcessError) as e:
            ctl._default_stager("h0", "bigbrain", root + "/bigbrain/worker.py", "# evil twin\n")
        assert b"refusing the stage dir" in e.value.stderr
        assert not (tmp_path / "fresh" / "stage" / "bigbrain" / "worker.py").exists()

    @needs_sh
    def test_run_node_sh_refuses_a_stage_dir_or_node_env_others_can_write(self, tmp_path):
        script = ROOT / "deploy" / "run_node.sh"
        stage = _stage(tmp_path)
        (stage / "node.args").write_text("--rank 1 --total 2\n")
        (stage / "node.env").write_text(f'BB_PYTHON="{sys.executable.replace(chr(92), "/")}"\n')
        ok = subprocess.run([SH, str(script), str(stage)], capture_output=True, text=True, timeout=60)
        assert ok.returncode == 0 and "the node ran" in ok.stdout, ok.stderr
        bad_dir = subprocess.run([SH, str(script), str(stage)], capture_output=True, text=True, timeout=60,
                                 env=_fake_find(tmp_path, "stage"))
        assert bad_dir.returncode == 77 and "refusing the stage dir" in bad_dir.stderr and "the node ran" not in bad_dir.stdout
        bad_env = subprocess.run([SH, str(script), str(stage)], capture_output=True, text=True, timeout=60,
                                 env=_fake_find(tmp_path, "node.env"))
        assert bad_env.returncode == 77 and "refusing node.env" in bad_env.stderr and "the node ran" not in bad_env.stdout

    @posix_only
    @needs_sh
    def test_run_node_sh_refuses_a_world_writable_node_env_or_a_symlinked_one(self, tmp_path):
        script = ROOT / "deploy" / "run_node.sh"
        stage = _stage(tmp_path)
        (stage / "node.args").write_text("--rank 1\n")
        real = tmp_path / "elsewhere.env"
        real.write_text(f'BB_PYTHON="{sys.executable}"\n')
        (stage / "node.env").symlink_to(real)
        p = subprocess.run([SH, str(script), str(stage)], capture_output=True, text=True, timeout=60)
        assert p.returncode == 77 and "refusing node.env" in p.stderr
        (stage / "node.env").unlink()
        (stage / "node.env").write_text(f'BB_PYTHON="{sys.executable}"\n')
        (stage / "node.env").chmod(0o666)
        p = subprocess.run([SH, str(script), str(stage)], capture_output=True, text=True, timeout=60)
        assert p.returncode == 77 and "refusing node.env" in p.stderr


# == M5: the API key ===============================================================================================================
def _serving_plan(key=KEY):
    return ctl.ClusterPlan(model_dir="/m", nodes=[_node(0, "alpha.lan", 0, 0, 10, 0, user="bigbrain"),
                                                   _node(1, "10.0.0.12", 0, 10, 20, 16900, user="bigbrain")],
                           total_blocks=20, token_port=16901, http_port=16980, api_token=key)


class _Runner:
    def __init__(self):
        self.calls = []

    def __call__(self, host, user, cmd):
        self.calls.append((host, user, cmd))

        class H:
            @staticmethod
            def poll():
                return 0
        return H()


SERVE_ARGV = ["--model-dir", "/m", "--total-blocks", "20", "--token-port", "16901",
              "--node", "10.0.0.11:0:0:10:0", "--node", "10.0.0.12:0:10:20:16900", "--serve-port", "16980"]


class TestM5ApiKey:
    def test_node_0_gets_a_key_file_never_the_key(self):
        c0, c1 = _serving_plan().build_commands(redirect_logs=True)
        assert KEY not in c0 and KEY not in c1 and "--api-token" not in c0
        w0 = shlex.split(c0)
        assert w0[w0.index("--api-key-file") + 1] == ctl.API_KEY_FILE == "api_key"     # relative to the stage dir (the cwd)
        assert "--api-key-file" not in c1

    def test_launch_stages_the_key_as_a_file_on_node_0_only_and_runs_no_command_with_it(self):
        staged = []
        runner = _Runner()
        c = ctl.Controller(_serving_plan(), runner=runner, dry_run=False,
                           stager=lambda h, u, path, content: staged.append((h, u, path, content)))
        c.launch()
        keys = [s for s in staged if s[2].endswith("/api_key")]
        assert keys == [("alpha.lan", "bigbrain", ctl.DEFAULT_STAGE_ROOT + "/api_key", KEY + "\n")]
        assert all(KEY not in cmd for _, _, cmd in runner.calls)
        assert all(KEY not in s[2] for s in staged)

    def test_the_default_stager_sends_the_key_on_stdin_never_in_argv(self, monkeypatch):
        seen = []
        monkeypatch.setattr(ctl.subprocess, "run", lambda argv, input=None, **kw: seen.append((argv, input)))
        ctl._default_stager("alpha.lan", "bigbrain", ctl.DEFAULT_STAGE_ROOT + "/api_key", KEY + "\n")
        (argv, data), = seen
        assert KEY not in " ".join(argv) and data == (KEY + "\n").encode()
        assert argv[-1].startswith("umask 077 && ")                                   # created 0600 inside a 0700 dir

    @pytest.mark.parametrize("how", ["file", "env", "deprecated-flag"])
    def test_no_printed_line_carries_the_key_dry_run_included(self, how, tmp_path, monkeypatch, capsys, caplog):
        argv = list(SERVE_ARGV)
        if how == "file":
            f = tmp_path / "api_key"
            f.write_text(KEY + "\n", encoding="utf-8")
            argv += ["--api-key-file", str(f)]
        elif how == "env":
            monkeypatch.setenv("BIGBRAIN_API_KEY", KEY)
        else:
            argv += ["--api-token", KEY]
        caplog.set_level("DEBUG")
        assert ctl.main(argv) == 0
        out = capsys.readouterr()
        assert KEY not in out.out and KEY not in out.err and KEY not in caplog.text
        assert "--api-key-file api_key" in out.out and "stage api_key" in out.out
        if how == "deprecated-flag":
            assert "--api-token" in out.err and "deprecated" in out.err

    def test_the_cli_reads_the_key_from_a_file_or_the_environment(self, tmp_path, monkeypatch):
        f = tmp_path / "k"
        f.write_text(KEY + "\n", encoding="utf-8")
        assert ctl.plan_from_args(SERVE_ARGV + ["--api-key-file", str(f)]).api_token == KEY
        monkeypatch.setenv("BIGBRAIN_API_KEY", KEY)
        assert ctl.plan_from_args(SERVE_ARGV).api_token == KEY
        f.write_text("\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            ctl.plan_from_args(SERVE_ARGV + ["--api-key-file", str(f)])

    def test_a_redacting_print_runner(self, capsys):
        c = ctl.Controller(_serving_plan(), dry_run=True)
        c._print_runner("alpha.lan", "bigbrain", f"echo {KEY}")
        assert KEY not in capsys.readouterr().out

    @posix_only
    def test_the_server_warns_about_a_key_file_others_can_read(self, tmp_path, caplog):
        from bigbrain.server import resolve_api_key
        f = tmp_path / "api_key"
        f.write_text(KEY + "\n", encoding="utf-8")
        f.chmod(0o644)
        caplog.set_level("WARNING")
        assert resolve_api_key(str(f), None, insecure=False) == KEY
        assert "chmod 600" in caplog.text and KEY not in caplog.text


# == M6: jinja2 ====================================================================================================================
NODE0 = ["--model-dir", "/m", "--slice", "0", "10", "--rank", "0", "--total", "2", "--next-host", "10.0.0.12",
         "--next-port", "16900", "--token-port", "16901", "--http-port", "16980", "--host", "127.0.0.1"]


class TestM6Jinja2:
    def test_the_package_requires_a_jinja2_without_the_known_sandbox_escapes(self):
        import tomllib
        deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
        assert "jinja2>=3.1.6" in deps and "numpy>=1.24" in deps

    def test_the_installed_jinja2_passes_and_older_ones_are_named(self, monkeypatch):
        from bigbrain import server as srv
        assert srv.jinja2_version_problem() is None                         # the test environment has a current jinja2
        for old in ("3.1.5", "3.1.4", "3.0.3", "2.11.3"):
            monkeypatch.setattr(srv, "_jinja2_version", lambda v=old: v)
            msg = srv.jinja2_version_problem()
            assert msg and old in msg and "3.1.6" in msg and "pip install" in msg
        for ok in ("3.1.6", "3.2.0", "4.0.0rc1", "3.1.6.post1"):
            monkeypatch.setattr(srv, "_jinja2_version", lambda v=ok: v)
            assert srv.jinja2_version_problem() is None, ok
        monkeypatch.setattr(srv, "_jinja2_version", lambda: "unknown")
        assert "unknown" in srv.jinja2_version_problem()                    # an unreadable version is refused, not trusted

    def test_a_template_is_never_rendered_with_an_old_jinja2(self, monkeypatch):
        from bigbrain import server as srv
        monkeypatch.setattr(srv, "_jinja2_version", lambda: "3.1.4")
        t = srv.ChatTemplate("{{ messages[0].content }}", source="tokenizer_config.json")
        with pytest.raises(RuntimeError, match="3.1.6"):
            t.render([{"role": "user", "content": "hi"}])

    def test_node_0_refuses_to_start_with_an_old_jinja2_before_loading_anything(self, monkeypatch):
        from bigbrain import server as srv
        monkeypatch.setattr(srv, "_jinja2_version", lambda: "3.1.4")
        monkeypatch.setattr(srv, "worker_from_args", lambda a: pytest.fail("the model must not be loaded"))
        monkeypatch.setenv("BIGBRAIN_API_KEY", KEY)
        with pytest.raises(SystemExit) as e:
            srv.main(NODE0)
        assert "3.1.4" in str(e.value) and "3.1.6" in str(e.value)


# == found while documenting v0.2.1: the registry's per-node ssh user in every split mode ==========================================
def test_the_registry_user_is_used_by_the_even_and_explicit_splits_too(tmp_path, monkeypatch):
    reg = {"version": 1,
           "nodes": {"alpha": {"host": "10.0.0.11", "user": "alice", "python": "/opt/a/bin/python",
                               "gpus": [{"index": 0, "name": "G", "total_mib": 24576}]},
                     "beta": {"host": "10.0.0.12", "user": "bob", "python": "/opt/b/bin/python",
                              "gpus": [{"index": 0, "name": "G", "total_mib": 24576}]}},
           "models": {"M": {"layers": 20, "kv_elements_per_token": 1024, "size_gib": 10.0,
                            "paths": {"alpha": "/data/M", "beta": "/srv/M"}}}}
    f = tmp_path / "reg.json"
    f.write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setenv("BIGBRAIN_REGISTRY", str(f))
    even = ctl.plan_from_args(["--model", "M", "--all-cards", "--split", "even", "--total-blocks", "20"])
    assert [(n.host, n.user, n.python_bin) for n in even.nodes] == [("10.0.0.11", "alice", "/opt/a/bin/python"),
                                                                    ("10.0.0.12", "bob", "/opt/b/bin/python")]
    explicit = ctl.plan_from_args(["--model", "M", "--total-blocks", "20", "--token-port", "16901",
                                   "--node", "10.0.0.11:0:0:12:0", "--node", "10.0.0.12:0:12:20:16900"])
    assert [n.user for n in explicit.nodes] == ["alice", "bob"]


# == the low-severity items fixed in v0.2.1 ========================================================================================
class _Loaded(Exception):
    """Raised by a stubbed worker_from_args: the server got past its start-up checks."""


class TestLowSeverity:
    # S2 -------------------------------------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.11", "::"])
    def test_insecure_no_auth_is_refused_off_loopback(self, host, monkeypatch):
        from bigbrain import server as srv
        monkeypatch.delenv("BIGBRAIN_API_KEY", raising=False)
        monkeypatch.setattr(srv, "worker_from_args", lambda a: pytest.fail("must refuse before loading"))
        argv = [a if a != "127.0.0.1" else host for a in NODE0] + ["--insecure-no-auth"]
        with pytest.raises(SystemExit, match="loopback"):
            srv.main(argv)

    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
    def test_insecure_no_auth_still_works_on_loopback(self, host, monkeypatch):
        from bigbrain import server as srv
        monkeypatch.delenv("BIGBRAIN_API_KEY", raising=False)

        def loaded(a):
            raise _Loaded()
        monkeypatch.setattr(srv, "worker_from_args", loaded)
        argv = [a if a != "127.0.0.1" else host for a in NODE0] + ["--insecure-no-auth"]
        with pytest.raises(_Loaded):
            srv.main(argv)

    # S3 -------------------------------------------------------------------------------------------------------------------------
    def test_a_short_key_is_refused_by_the_server_and_the_plan(self, tmp_path, monkeypatch):
        from bigbrain.server import resolve_api_key
        f = tmp_path / "k"
        f.write_text("short\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="16"):
            resolve_api_key(str(f), None, insecure=False)
        monkeypatch.setenv("BB_K", "x" * 15)
        with pytest.raises(SystemExit, match="token_hex"):
            resolve_api_key(None, "BB_K", insecure=False)
        assert resolve_api_key(None, None, insecure=False, token="x" * 16) == "x" * 16
        assert any("16" in p for p in _serving_plan(key="t").validate())
        assert _serving_plan().validate() == []

    # S4 -------------------------------------------------------------------------------------------------------------------------
    def test_unauthenticated_health_does_not_echo_the_failure_detail(self, http_server):
        secret_detail = "the token link died: socket error: [Errno 104] reset by 10.0.0.12:16901 (/home/bigbrain/x.py)"
        _, _, port = http_server(_Engine(dead=secret_detail))
        status, body = _get(port, "/health")
        assert status == 503 and body["status"] == "degraded"
        assert secret_detail not in json.dumps(body) and "10.0.0.12" not in json.dumps(body)
        assert body["engine"]["dead"]                                       # still says THAT it is down
        status, body = _get(port, "/health", key=KEY)
        assert status == 503 and body["engine"]["dead"] == secret_detail   # the key holder gets the reason

    # S5 -------------------------------------------------------------------------------------------------------------------------
    def test_a_long_stop_string_is_refused(self, http_server):
        import http.client
        from bigbrain.server import MAX_STOP_LENGTH
        assert MAX_STOP_LENGTH == 256
        _, _, port = http_server()
        for stop, want in (("x" * 257, 400), (["ok", "y" * 4_000_000], 400), ("x" * 256, 200)):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            c.request("POST", "/v1/completions", body=json.dumps({"prompt": "hi", "max_tokens": 4, "stop": stop}),
                      headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = c.getresponse()
            body = r.read()
            c.close()
            assert r.status == want, (r.status, body[:200])

    # S6 -------------------------------------------------------------------------------------------------------------------------
    @needs_sh
    def test_run_node_sh_never_glob_expands_node_args(self, tmp_path):
        stage = _stage(tmp_path)
        (stage / "bigbrain" / "worker.py").write_text("import sys\nprint('argv=' + '|'.join(sys.argv[1:]))\n")
        (stage / "node.args").write_text("--prompt * --model-dir /data/m?del\n")
        (stage / "node.env").write_text(f'BB_PYTHON="{sys.executable.replace(chr(92), "/")}"\n')
        p = subprocess.run([SH, str(ROOT / "deploy" / "run_node.sh"), str(stage)], capture_output=True, text=True, timeout=60)
        assert p.returncode == 0, p.stderr
        assert "argv=--prompt|*|--model-dir|/data/m?del" in p.stdout

    # S7 -------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _settings(path):
        return {line.split("=", 1)[0]: line.split("=", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()
                if "=" in line and not line.lstrip().startswith("#")}

    def test_the_systemd_unit_is_hardened(self):
        """What works in every user manager is in the unit; the namespace sandbox (a user unit needs unprivileged user namespaces
        for it, which e.g. Ubuntu 24.04 restricts) is a drop-in you add where it works."""
        unit_path = ROOT / "deploy" / "bigbrain-node@.service"
        body = self._settings(unit_path)
        assert body.get("NoNewPrivileges") == "yes" and body.get("UMask") == "0077"
        assert body.get("RestrictSUIDSGID") == "yes" and body.get("LockPersonality") == "yes"
        assert "/tmp/bigbrain" not in unit_path.read_text(encoding="utf-8")   # the example instance is the private default
        drop = ROOT / "deploy" / "bigbrain-node@.service.d" / "sandbox.conf"
        sb = self._settings(drop)
        assert drop.read_text(encoding="utf-8").count("[Service]") == 1
        assert sb.get("PrivateTmp") == "yes" and sb.get("ProtectSystem") == "strict" and sb.get("ProtectHome") == "read-only"
        assert "%h/.cache/bigbrain" in sb.get("ReadWritePaths", "")
        assert "*.conf" in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")   # the drop-in ships in the sdist too

    # S8 -------------------------------------------------------------------------------------------------------------------------
    def test_a_site_toml_in_the_current_directory_is_not_read_implicitly(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "site.toml").write_text('[ssh]\nuser = "mallory"\n', encoding="utf-8")
        monkeypatch.setattr(site, "SITE_FILE", "site.toml")
        monkeypatch.setattr(site, "user_site_file", lambda: tmp_path / "no-such-dir" / "site.toml")
        monkeypatch.chdir(tmp_path)
        site._warned.clear()
        assert site.path() is None and site.ssh_user() != "mallory"
        err = capsys.readouterr().err
        assert "site.toml" in err and "BIGBRAIN_SITE" in err and "ignored" in err
        monkeypatch.setenv("BIGBRAIN_SITE", str(tmp_path / "site.toml"))   # explicit: read
        assert site.ssh_user() == "mallory"

    def test_the_per_user_site_file_is_read_without_an_env_var(self, tmp_path, monkeypatch):
        conf = tmp_path / "config" / "bigbrain"
        conf.mkdir(parents=True)
        (conf / "site.toml").write_text('[ssh]\nuser = "carol"\n', encoding="utf-8")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.setattr(site, "user_site_file", site._default_user_site_file)
        assert site.path() == conf / "site.toml" and site.ssh_user() == "carol"

    # S10 ------------------------------------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("member", ["../evil.py", "/abs/evil.py", "link"])
    def test_the_release_extraction_fallback_refuses_escaping_members(self, member, tmp_path):
        import io
        import tarfile
        sys.path.insert(0, str(ROOT / "deploy"))
        try:
            import release
        finally:
            sys.path.remove(str(ROOT / "deploy"))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            ok = tarfile.TarInfo("src/ok.py")
            ok.size = 3
            tf.addfile(ok, io.BytesIO(b"ok\n"))
            bad = tarfile.TarInfo(member)
            if member == "link":
                bad.type, bad.linkname = tarfile.SYMTYPE, "../../outside"
            else:
                bad.size = 3
            tf.addfile(bad, io.BytesIO(b"no\n") if member != "link" else None)
        buf.seek(0)
        dest = tmp_path / "rel"
        dest.mkdir()
        with tarfile.open(fileobj=buf) as tf:
            with pytest.raises(ValueError, match="refusing"):
                release.extract_checked(tf, dest)
        assert not (tmp_path / "evil.py").exists()
