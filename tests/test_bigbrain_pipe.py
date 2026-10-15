# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Tests for the BigBrain wire protocol (tools/bigbrain/pipe.py).

Loopback sockets only: no GPU, no external network, no production queue.
"""

import socket
import struct
import threading
import time

import numpy as np
import pytest

from tools.bigbrain.pipe import (
    FLAG_PREFILL,
    HEADER_FMT,
    HEADER_FMT_V1,
    HEADER_SIZE,
    HEADER_SIZE_V1,
    MSG_ACTIVATION,
    MSG_BYE,
    MSG_CONTROL,
    MSG_HEARTBEAT,
    MSG_LOGITS,
    MSG_RESET,
    MSG_TOKEN,
    DT_FP16,
    DT_FP32,
    DT_U32,
    VERSION,
    BigBrainLink,
    BigBrainPeerDead,
    BigBrainProtocolError,
    BigBrainRing,
    pack_header,
    unpack_header,
)

FLAG_LAST_CHUNK = 1 << 0


def _pair():
    """Two connected loopback links. Returns (server_side, client_side)."""
    server = BigBrainLink()
    port = server.listen_inbound(0)
    client = BigBrainLink()
    client.connect("127.0.0.1", port, timeout_s=5.0)
    server.accept_peer(timeout_s=5.0)
    return server, client


class TestHeader:
    def test_size_and_roundtrip(self):
        # v2 = the 39-byte v1 header (every v1 field at its v1 offset) + a
        # 16-byte tail: request_id uint64, position uint64.
        assert VERSION == 2
        assert HEADER_SIZE_V1 == 39
        assert HEADER_SIZE == 55
        assert struct.calcsize(HEADER_FMT) == HEADER_SIZE
        assert HEADER_FMT.startswith(HEADER_FMT_V1)
        h = pack_header(MSG_ACTIVATION, batch=2, seq_len=16, hidden_dim=512,
                        dtype=DT_FP16, flags=FLAG_LAST_CHUNK, payload_len=1024,
                        seq_id=7, request_id=41, position=1234)
        assert len(h) == HEADER_SIZE
        d = unpack_header(h)
        assert d["version"] == 2
        assert d["msgtype"] == MSG_ACTIVATION
        assert d["batch"] == 2
        assert d["seq_len"] == 16
        assert d["hidden_dim"] == 512
        assert d["dtype"] == DT_FP16
        assert d["flags"] == FLAG_LAST_CHUNK
        assert d["payload_len"] == 1024
        assert d["seq_id"] == 7
        assert d["request_id"] == 41
        assert d["position"] == 1234

    def test_v1_header_still_parses(self):
        """An old (v1, 39-byte) header parses; it carries no request/position,
        which read as 0."""
        h = struct.pack(HEADER_FMT_V1, b"BGBN", 1, MSG_ACTIVATION, 3, 1, 2, 4,
                        DT_FP32, FLAG_LAST_CHUNK, 32)
        assert len(h) == HEADER_SIZE_V1
        d = unpack_header(h)
        assert d["version"] == 1
        assert d["seq_len"] == 2 and d["hidden_dim"] == 4 and d["payload_len"] == 32
        assert d["request_id"] == 0 and d["position"] == 0
        # pack_header can still emit v1 for an old peer, but v1 cannot carry
        # the v2 fields (refuse instead of silently dropping them).
        assert pack_header(MSG_ACTIVATION, version=1) == pack_header(MSG_ACTIVATION, version=1)
        assert len(pack_header(MSG_ACTIVATION, version=1)) == HEADER_SIZE_V1
        with pytest.raises(BigBrainProtocolError, match="v1"):
            pack_header(MSG_ACTIVATION, version=1, request_id=5)

    def test_bad_magic(self):
        h = bytearray(pack_header(MSG_ACTIVATION))
        h[0:4] = b"BADM"
        with pytest.raises(BigBrainProtocolError, match="bad magic"):
            unpack_header(bytes(h))

    def test_bad_version(self):
        h = bytearray(pack_header(MSG_ACTIVATION))
        h[4] = 9
        with pytest.raises(BigBrainProtocolError, match="unsupported version 9"):
            unpack_header(bytes(h))

    def test_incomplete_header(self):
        with pytest.raises(BigBrainProtocolError, match="incomplete"):
            unpack_header(b"\x00" * 10)

    def test_incomplete_v2_tail(self):
        h = pack_header(MSG_ACTIVATION, request_id=1)
        with pytest.raises(BigBrainProtocolError, match="incomplete"):
            unpack_header(h[:HEADER_SIZE_V1 + 3])


class TestActivation:
    def test_fp16_roundtrip(self):
        server, client = _pair()
        try:
            arr = np.arange(2 * 4 * 8, dtype=np.float16).reshape(2, 4, 8)
            client.send_activation(arr, last_chunk=True)
            got = server.recv_activation(expected_shape=(2, 4, 8))
            np.testing.assert_array_equal(got, arr)
            assert got.dtype == np.float16
        finally:
            client.close()
            server.close()

    def test_fp32_roundtrip(self):
        server, client = _pair()
        try:
            arr = np.random.default_rng(0).random((1, 3, 16), dtype=np.float32)
            client.send_activation(arr, last_chunk=True)
            got = server.recv_activation(expected_shape=(1, 3, 16))
            np.testing.assert_array_equal(got, arr)
            assert got.dtype == np.float32
        finally:
            client.close()
            server.close()

    def test_multi_chunk_assembly(self):
        server, client = _pair()
        try:
            arr = np.arange(2 * 12 * 8, dtype=np.float16).reshape(2, 12, 8)
            for i in range(3):
                chunk = arr[:, i * 4:(i + 1) * 4, :]
                client.send_activation(chunk, last_chunk=(i == 2))
            got = server.recv_activation(expected_shape=(2, 12, 8))
            np.testing.assert_array_equal(got, arr)
        finally:
            client.close()
            server.close()

    def test_shape_mismatch_raises(self):
        server, client = _pair()
        try:
            arr = np.zeros((1, 2, 4), dtype=np.float16)
            client.send_activation(arr, last_chunk=True)
            with pytest.raises(BigBrainProtocolError, match="shape mismatch"):
                server.recv_activation(expected_shape=(1, 4, 4))
        finally:
            client.close()
            server.close()

    def test_bf16_refused_v1(self):
        server, client = _pair()
        try:
            client.send_header(MSG_ACTIVATION, batch=1, seq_len=1, hidden_dim=2,
                               dtype=1, flags=FLAG_LAST_CHUNK,
                               payload=b"\x00\x00\x00\x00")
            with pytest.raises(BigBrainProtocolError, match="not supported"):
                server.recv_activation()
        finally:
            client.close()
            server.close()


class TestControl:
    def test_reset_carries_reason(self):
        server, client = _pair()
        try:
            client.reset("worker down")
            msgtype, header, payload = server.recv_frame()
            assert msgtype == MSG_RESET
            assert payload.decode("utf-8") == "worker down"
        finally:
            client.close()
            server.close()

    def test_abrupt_close_surfaces_peer_dead(self):
        server, client = _pair()
        try:
            server._sock.close()  # abrupt, no BYE
            with pytest.raises(BigBrainPeerDead):
                client.recv_frame()
        finally:
            client.close()
            server.close()

    def test_bye_marks_dead(self):
        server, client = _pair()
        try:
            client.close()  # sends BYE
            msgtype, _, _ = server.recv_frame()
            assert msgtype == MSG_BYE
            assert server.is_dead
        finally:
            server.close()

    def test_a_longer_watchdog_survives_a_stall_the_default_does_not(self):
        """A peer that goes silent (no heartbeat, no frame) for longer than the watchdog is declared dead; the same
        stall is survived when the receiver's --watchdog-s is larger (a tunnelled hop, a long compute step)."""
        def stall(watchdog_s):
            server = BigBrainLink(watchdog_s=watchdog_s)
            port = server.listen_inbound(0)
            client = BigBrainLink()
            client.connect("127.0.0.1", port, timeout_s=5.0)
            server.accept_peer(timeout_s=5.0)
            box = {}

            def receive():                               # blocked in recv() BEFORE the stall, as a real node is
                try:
                    box["got"] = server.recv_activation(expected_shape=(1, 2, 4))
                except Exception as exc:                 # noqa: BLE001
                    box["err"] = exc
            try:
                client._hb_stop.set()                    # the sender stops heartbeating, like a node stuck in one long step
                client._hb_thread.join(2.0)
                rx = threading.Thread(target=receive, daemon=True)
                rx.start()
                time.sleep(2.6)
                if "err" not in box:
                    client.send_activation(np.ones((1, 2, 4), dtype=np.float16), last_chunk=True)
                rx.join(5.0)
                if "err" in box:
                    raise box["err"]
                return box["got"]
            finally:
                client.close()
                server.close()
        with pytest.raises(BigBrainPeerDead, match="no frame for 2.0s"):
            stall(2.0)
        got = stall(8.0)
        assert got.shape == (1, 2, 4)

    def test_a_long_silence_is_logged_and_measured(self, caplog):
        """The stall that killed the token link on 2026-10-14 was never measured: a gap between frames is now logged
        (GAP_WARN_S) and the longest one kept, so a --watchdog-s decision can be made from a number."""
        import logging
        server = BigBrainLink(watchdog_s=10.0)
        port = server.listen_inbound(0)
        client = BigBrainLink()
        client.connect("127.0.0.1", port, timeout_s=5.0)
        server.accept_peer(timeout_s=5.0)
        try:
            client._hb_stop.set()
            client._hb_thread.join(2.0)
            box = {}
            rx = threading.Thread(target=lambda: box.setdefault("a", server.recv_activation(expected_shape=(1, 2, 4))), daemon=True)
            with caplog.at_level(logging.WARNING, logger="tools.bigbrain.pipe"):
                rx.start()
                time.sleep(2.3)
                client.send_activation(np.ones((1, 2, 4), dtype=np.float16), last_chunk=True)
                rx.join(5.0)
            assert box["a"].shape == (1, 2, 4)
            assert 2.0 <= server.max_gap_s < 6.0
            assert any("between frames" in r.getMessage() for r in caplog.records)
            assert client.max_gap_s < 2.0                      # the other direction was never silent
        finally:
            client.close()
            server.close()

    def test_idle_link_survives_on_heartbeats(self):
        server, client = _pair()
        try:
            time.sleep(2.5)  # > one watchdog window's worth of missed beats
            assert not client.is_dead
            assert not server.is_dead
            # still functional after the idle period
            arr = np.ones((1, 2, 4), dtype=np.float16)
            client.send_activation(arr, last_chunk=True)
            np.testing.assert_array_equal(
                server.recv_activation(expected_shape=(1, 2, 4)), arr)
        finally:
            client.close()
            server.close()


class TestRing:
    def test_first_node_has_no_inbound(self):
        ring = BigBrainRing("n0", inbound_port=None, last=False)
        assert ring.listen_inbound() is None
        with pytest.raises(BigBrainProtocolError):
            ring.recv_inbound()
        ring.close()

    def test_two_node_ring_forward(self):
        n0 = BigBrainRing("n0", inbound_port=None,
                          outbound_peers=[("127.0.0.1", 0)], last=False)
        n1 = BigBrainRing("n1", inbound_port=0, last=True)
        try:
            port = n1.listen_inbound()
            n0.outbound_peers = [("127.0.0.1", port)]
            n0.connect_outbound()
            n1.accept_inbound()
            arr = np.arange(1 * 6 * 4, dtype=np.float16).reshape(1, 6, 4)
            n0.send_outbound(arr)
            got = n1.recv_inbound(expected_shape=(1, 6, 4))
            np.testing.assert_array_equal(got, arr)
        finally:
            n0.close()
            n1.close()


def _raw_pair():
    """A raw client socket + a BigBrainLink on the accepting side (to put
    hand-made bytes on the wire, e.g. a v1 peer's frames)."""
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(1)
    raw = socket.create_connection(lsock.getsockname())
    conn, _ = lsock.accept()
    lsock.close()
    return raw, BigBrainLink.from_socket(conn)


class TestV2Messages:
    """v2 frames: request id + position in every header, the CONTROL / TOKEN /
    LOGITS message types, the PREFILL flag, and recv_message() which assembles
    any of them (heartbeats skipped)."""

    def test_activation_carries_request_position_and_prefill(self):
        server, client = _pair()
        try:
            arr = np.arange(1 * 3 * 4, dtype=np.float32).reshape(1, 3, 4)
            client.send_activation(arr, request_id=9, position=17, flags=FLAG_PREFILL)
            msg = server.recv_message()
            assert msg.kind == "activation"
            assert msg.request_id == 9 and msg.position == 17
            assert msg.is_prefill
            np.testing.assert_array_equal(msg.array, arr)
            # A decode-step activation has no PREFILL flag.
            client.send_activation(arr[:, :1], request_id=9, position=20)
            msg = server.recv_message()
            assert not msg.is_prefill and msg.position == 20
        finally:
            client.close()
            server.close()

    def test_chunked_activation_with_heartbeat_between_chunks(self):
        server, client = _pair()
        try:
            arr = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2)
            client.send_activation(arr[:, :3], last_chunk=False, request_id=2, position=0)
            client.send_header(MSG_HEARTBEAT)  # the daemon may interleave one
            client.send_activation(arr[:, 3:], last_chunk=True, request_id=2, position=3)
            msg = server.recv_message()
            assert msg.kind == "activation" and msg.position == 0
            np.testing.assert_array_equal(msg.array, arr)
        finally:
            client.close()
            server.close()

    def test_hyper_connection_residual_keeps_its_streams(self):
        """(batch, seq, hc_mult, hidden) -- e.g. 4 x 2560 for Qwen3.8-Flash-Next --
        travels as-is, chunked too; the stream count rides in flag bits 8..15."""
        server, client = _pair()
        try:
            arr = np.arange(1 * 5 * 4 * 3, dtype=np.float32).reshape(1, 5, 4, 3)
            client.send_activation(arr, request_id=1, position=0, flags=FLAG_PREFILL)
            msg = server.recv_message()
            assert msg.array.shape == (1, 5, 4, 3) and msg.is_prefill
            np.testing.assert_array_equal(msg.array, arr)
            client.send_activation(arr[:, :2], last_chunk=False, request_id=1, position=0)
            client.send_activation(arr[:, 2:], last_chunk=True, request_id=1, position=2)
            np.testing.assert_array_equal(server.recv_message().array, arr)
            with pytest.raises(BigBrainProtocolError, match="3-D or 4-D"):
                client.send_activation(np.zeros((1, 1, 1, 1, 1), np.float32))
        finally:
            client.close()
            server.close()

    def test_control_roundtrip(self):
        server, client = _pair()
        try:
            body = {"op": "begin", "request_id": 5,
                    "sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 1}}
            client.send_control(body, request_id=5)
            msg = server.recv_message()
            assert msg.kind == "control"
            assert msg.request_id == 5
            assert msg.control == body
        finally:
            client.close()
            server.close()

    def test_control_must_be_a_json_object(self):
        server, client = _pair()
        try:
            client.send_header(MSG_CONTROL, payload=b"[1, 2]")
            with pytest.raises(BigBrainProtocolError, match="JSON object"):
                server.recv_message()
        finally:
            client.close()
            server.close()

    def test_token_roundtrip_full_vocab_range(self):
        server, client = _pair()
        try:
            client.send_tokens([248319, 0, 151645], request_id=3, position=12)
            msg = server.recv_message()
            assert msg.kind == "token"
            assert msg.tokens == [248319, 0, 151645]
            assert msg.request_id == 3 and msg.position == 12
            assert msg.header["dtype"] == DT_U32
        finally:
            client.close()
            server.close()

    def test_token_payload_size_checked(self):
        server, client = _pair()
        try:
            client.send_header(MSG_TOKEN, batch=1, seq_len=2, hidden_dim=1,
                               dtype=DT_U32, payload=b"\x01\x00\x00\x00")
            with pytest.raises(BigBrainProtocolError, match="token"):
                server.recv_message()
        finally:
            client.close()
            server.close()

    def test_logits_roundtrip(self):
        server, client = _pair()
        try:
            row = np.linspace(-3, 3, 11, dtype=np.float32)
            client.send_logits(row, request_id=4, position=8)
            msg = server.recv_message()
            assert msg.kind == "logits"
            assert msg.request_id == 4 and msg.position == 8
            np.testing.assert_array_equal(msg.array.reshape(-1), row)
        finally:
            client.close()
            server.close()

    def test_reset_and_bye_messages(self):
        server, client = _pair()
        try:
            client.reset("new request")
            msg = server.recv_message()
            assert msg.kind == "reset" and msg.reason == "new request"
            client.close()
            msg = server.recv_message()
            assert msg.kind == "bye"
            assert server.is_dead
        finally:
            server.close()

    def test_heartbeats_visible_on_request(self):
        """include_heartbeats=True lets a caller with a deadline wake up on
        every beat (the token wait on node 0)."""
        server, client = _pair()
        try:
            client.send_header(MSG_HEARTBEAT)
            msg = server.recv_message(include_heartbeats=True)
            assert msg.kind == "heartbeat"
        finally:
            client.close()
            server.close()

    def test_last_rx_advances_on_every_frame(self):
        server, client = _pair()
        try:
            before = server.last_rx
            time.sleep(0.05)
            client.send_header(MSG_HEARTBEAT)
            server.recv_frame()
            assert server.last_rx > before
        finally:
            client.close()
            server.close()

    def test_v1_frame_from_an_old_peer_still_parses(self):
        """A v1 peer's 39-byte frame is read correctly off the wire (the
        receiver reads the 16-byte v2 tail only when the version says so)."""
        raw, link = _raw_pair()
        try:
            arr = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
            for seq in (1, 2):
                raw.sendall(struct.pack(HEADER_FMT_V1, b"BGBN", 1, MSG_ACTIVATION, seq, 1, 2, 4,
                                        DT_FP32, FLAG_LAST_CHUNK, arr.nbytes) + arr.tobytes())
            np.testing.assert_array_equal(link.recv_activation(expected_shape=(1, 2, 4)), arr)
            msg = link.recv_message()
            assert msg.kind == "activation" and msg.header["version"] == 1
            assert msg.request_id == 0 and msg.position == 0
            np.testing.assert_array_equal(msg.array, arr)
        finally:
            raw.close()
            link.close()

    def test_unknown_version_on_the_wire_fails_clearly_and_kills_the_link(self):
        raw, link = _raw_pair()
        try:
            raw.sendall(struct.pack(HEADER_FMT_V1, b"BGBN", 7, MSG_ACTIVATION, 1, 1, 1, 1,
                                    DT_FP32, FLAG_LAST_CHUNK, 4) + b"\x00" * 4)
            with pytest.raises(BigBrainProtocolError, match="unsupported version 7"):
                link.recv_message()
            # The stream is desynchronised after a bad header: the link is dead.
            assert link.is_dead
        finally:
            raw.close()
            link.close()

    def test_unknown_msgtype_fails_clearly(self):
        server, client = _pair()
        try:
            client.send_header(99)
            with pytest.raises(BigBrainProtocolError, match="msgtype 99"):
                server.recv_message()
        finally:
            client.close()
            server.close()

    def test_connected_flag(self):
        lst = BigBrainLink()
        port = lst.listen_inbound(0, bind_host="127.0.0.1")
        assert not lst.connected
        cli = BigBrainLink().connect("127.0.0.1", port)
        lst.accept_peer(timeout_s=5.0)
        try:
            assert lst.connected and cli.connected
        finally:
            cli.close()
            lst.close()
        assert not lst.connected

    def test_drain_discards_peer_heartbeats_and_detects_close(self):
        """A link one side only SENDS on drains the peer's heartbeats (so the
        kernel buffers never fill) and goes dead when the peer closes."""
        server, client = _pair()
        try:
            client.start_drain()
            before = client.last_rx
            time.sleep(1.2)  # >= 2 peer heartbeats arrive and are discarded
            assert client.last_rx > before
            assert not client.is_dead
            # sending still works while draining
            client.send_activation(np.ones((1, 1, 2), np.float32))
            assert server.recv_message().kind == "activation"
            server.close()
            deadline = time.monotonic() + 3
            while not client.is_dead and time.monotonic() < deadline:
                time.sleep(0.02)
            assert client.is_dead
        finally:
            client.close()
            server.close()

    def test_connect_through_a_dead_tunnel_is_refused(self):
        """An ssh -L/-R forward ACCEPTS the connect even when its far side is
        not listening yet, then closes it at once. connect(settle_s) sees that
        EOF and fails like a refused connect, so the caller retries."""
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(4)
        port = lsock.getsockname()[1]

        def tunnel_without_far_side():
            conn, _ = lsock.accept()
            conn.close()

        th = threading.Thread(target=tunnel_without_far_side, daemon=True)
        th.start()
        with pytest.raises(ConnectionError, match="closed"):
            BigBrainLink().connect("127.0.0.1", port, settle_s=1.0)
        th.join(5)
        # A real listener that has not accepted yet (still loading its model)
        # sends nothing: the settle times out and the connect stands.
        link = BigBrainLink().connect("127.0.0.1", port, settle_s=0.3)
        try:
            assert link.connected and not link.is_dead
        finally:
            link.close()
            lsock.close()

    def test_recv_activation_rejects_control(self):
        """The legacy activation-only reader refuses a control frame loudly."""
        server, client = _pair()
        try:
            client.send_control({"op": "end"})
            with pytest.raises(BigBrainProtocolError, match="unexpected msgtype"):
                server.recv_activation()
        finally:
            client.close()
            server.close()