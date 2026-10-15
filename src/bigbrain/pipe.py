# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""BigBrain wire protocol for cross-machine pipeline-parallel LLM inference.

Each BigBrain node holds a slice of a model's transformer layers. Between
nodes, the hidden-state activation tensor (batch, seq_len, hidden_dim) hops
over a plain TCP socket using this protocol.

Header (little-endian)
----------------------
* v1 (39 bytes, ``<4sBBQIIIBIQ``): magic b'BGBN' (4s), version uint8,
  msgtype uint8, seq_id uint64, batch uint32, seq_len uint32,
  hidden_dim uint32, dtype uint8, flags uint32, payload_len uint64.
* v2 (55 bytes, ``<4sBBQIIIBIQQQ``) = the v1 header, every field at its v1
  offset, plus a 16-byte tail: request_id uint64, position uint64.
  ``request_id`` names the generation request a frame belongs to (a stale frame
  of an aborted request is recognisable); ``position`` is the sequence
  position of the FIRST token the frame covers (an activation's past_len, a
  token's index in the sequence). This build SENDS v2 and still PARSES v1
  frames (request_id/position read as 0); any other version fails with
  "unsupported version N" and kills the link (the stream is desynchronised).

Message types
-------------
* ACTIVATION (1): raw bytes of a contiguous fp16/fp32 (batch, seq, hidden)
  array -- or (batch, seq, streams, hidden) for hyper-connection models, the
  stream count in flag bits 8..15. A large activation may be split into
  several frames along the seq axis; the final frame carries FLAG_LAST_CHUNK.
  FLAG_PREFILL marks a prompt chunk (the last node discards its result).
* HEARTBEAT (2), RESET (3, utf-8 reason), READY (4), BYE (5).
* CONTROL (6, v2): a utf-8 JSON object, e.g. ``{"op": "begin",
  "request_id": 7, "sampling": {...}}``. Relayed down the chain.
* TOKEN (7, v2): uint32 token ids (dtype DT_U32): the sampled token (last
  node -> node 0), and on the activation chain the INPUT ids of the
  activation that follows (a layer such as PLE needs them on any node).
* LOGITS (8, v2): fp32 (1, 1, vocab) row, last node -> node 0 (parity/debug
  only, ~1 MB per token for a 248k vocab).

A daemon heartbeat (every 0.5 s) keeps idle links alive. A receiver whose
socket sees no bytes for the watchdog window (default 5 s) declares the peer
dead and raises BigBrainPeerDead.

Stdlib + numpy only: no torch, no asyncio. The GPU-side workers convert
their CUDA tensors to/from numpy at the boundary (``.cpu().numpy()``).
"""

from __future__ import annotations

import json
import logging
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HEADER_FMT_V1 = "<4sBBQIIIBIQ"
HEADER_SIZE_V1 = struct.calcsize(HEADER_FMT_V1)  # 39
HEADER_EXT_FMT = "<QQ"  # v2 tail: request_id, position
HEADER_EXT_SIZE = struct.calcsize(HEADER_EXT_FMT)  # 16
HEADER_FMT = HEADER_FMT_V1 + "QQ"  # the current (v2) full header
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 55
MAGIC = b"BGBN"
VERSION = 2
SUPPORTED_VERSIONS = (1, 2)

MSG_ACTIVATION = 1
MSG_HEARTBEAT = 2
MSG_RESET = 3
MSG_READY = 4
MSG_BYE = 5
MSG_CONTROL = 6
MSG_TOKEN = 7
MSG_LOGITS = 8

DT_FP16 = 0
DT_BF16 = 1  # reserved; not supported (raise on receipt)
DT_FP32 = 2
DT_U32 = 3  # token ids

FLAG_LAST_CHUNK = 1 << 0
FLAG_PREFILL = 1 << 1
# Bits 8..15: the number of parallel residual STREAMS (hyper-connection models
# carry (batch, seq, hc_mult, hidden) fp32; 0 or 1 = a plain (batch, seq, hidden)
# array). hidden_dim is then the per-stream width.
FLAG_STREAMS_SHIFT = 8
FLAG_STREAMS_MASK = 0xFF << FLAG_STREAMS_SHIFT
_FLAG_LAST_CHUNK = FLAG_LAST_CHUNK  # v1 name, kept for importers

_DTYPE_TO_NP = {DT_FP16: np.float16, DT_FP32: np.float32}

# Frame size caps (v0.2.1, security review M3). Every limit is checked on the HEADER, before the payload is read, so a peer cannot
# make a node allocate what it names. A node narrows DEFAULT_MAX_FRAME_BYTES to what its model can legitimately send it
# (frame_limit(); `--max-frame-mib` overrides).
DEFAULT_MAX_FRAME_BYTES = 1 << 30      # 1 GiB: a link nobody narrowed (tests, embedding)
MIN_FRAME_LIMIT = 64 << 20             # frame_limit() never answers less: room for a logits row and odd configs
MAX_CONTROL_BYTES = 16 << 20           # one CONTROL frame (JSON)
MAX_SMALL_PAYLOAD = 64 << 10           # a RESET reason; HEARTBEAT / READY / BYE carry nothing


def frame_limit(hidden_size: int, max_tokens: int, *, streams: int = 1, vocab_size: int = 0, slack: int = 8) -> int:
    """The largest frame a node of this model may receive: one activation of `max_tokens` x `hidden_size` x `streams` in fp32
    (the widest wire dtype), or one fp32 logits row, times `slack` (a peer started with a larger --prefill-chunk, a stream count
    the config does not name), and never less than MIN_FRAME_LIMIT."""
    act = int(max_tokens) * int(hidden_size) * max(1, int(streams)) * 4
    logits = int(vocab_size or 0) * 4
    return max(MIN_FRAME_LIMIT, int(slack) * max(act, logits))


_DEFAULT_WATCHDOG_S = 5.0
_HEARTBEAT_INTERVAL_S = 0.5
GAP_WARN_S = 2.0           # a frame arriving this long after the previous one is logged (a heartbeat is due every 0.5 s)
_MAX_U64 = (1 << 64) - 1


class BigBrainProtocolError(Exception):
    """Raised on protocol violations (bad magic, version, shape, dtype)."""


class BigBrainPeerDead(ConnectionError):
    """Raised when the peer is gone (EOF, send/recv timeout, reset)."""


def pack_header(msgtype: int, *, batch: int = 0, seq_len: int = 0,
                hidden_dim: int = 0, dtype: int = DT_FP16, flags: int = 0,
                payload_len: int = 0, seq_id: int = 0, request_id: int = 0,
                position: int = 0, version: int = VERSION) -> bytes:
    """Pack one header: 55 bytes (v2, the default) or 39 bytes (v1, for an
    old peer -- which cannot carry request_id/position)."""
    if version == 1:
        if request_id or position:
            raise BigBrainProtocolError(
                "a v1 header cannot carry request_id/position (v2 fields)")
        return struct.pack(HEADER_FMT_V1, MAGIC, 1, msgtype, seq_id,
                           batch, seq_len, hidden_dim, dtype, flags, payload_len)
    if version != 2:
        raise BigBrainProtocolError(f"cannot pack version {version}")
    if not (0 <= request_id <= _MAX_U64 and 0 <= position <= _MAX_U64):
        raise BigBrainProtocolError(f"request_id/position out of range: {request_id}/{position}")
    return struct.pack(HEADER_FMT, MAGIC, 2, msgtype, seq_id, batch, seq_len,
                       hidden_dim, dtype, flags, payload_len, request_id, position)


def unpack_header(data: bytes) -> Dict:
    """Unpack and validate one v1 or v2 header. Returns a dict of its fields
    (request_id/position are 0 for a v1 header)."""
    if len(data) < HEADER_SIZE_V1:
        raise BigBrainProtocolError(
            f"incomplete header: got {len(data)} bytes, need {HEADER_SIZE_V1}")
    (magic, version, msgtype, seq_id, batch, seq_len,
     hidden_dim, dtype, flags, payload_len) = struct.unpack(HEADER_FMT_V1, data[:HEADER_SIZE_V1])
    if magic != MAGIC:
        raise BigBrainProtocolError(f"bad magic {magic!r} (expected {MAGIC!r})")
    if version not in SUPPORTED_VERSIONS:
        raise BigBrainProtocolError(
            f"unsupported version {version} (this build speaks "
            f"{', '.join(str(v) for v in SUPPORTED_VERSIONS)}; are both ends "
            f"staged from the same bigbrain/pipe.py?)")
    request_id = position = 0
    size = HEADER_SIZE_V1
    if version >= 2:
        if len(data) < HEADER_SIZE:
            raise BigBrainProtocolError(
                f"incomplete v2 header: got {len(data)} bytes, need {HEADER_SIZE}")
        request_id, position = struct.unpack(HEADER_EXT_FMT, data[HEADER_SIZE_V1:HEADER_SIZE])
        size = HEADER_SIZE
    return {"magic": magic, "version": version, "msgtype": msgtype,
            "seq_id": seq_id, "batch": batch, "seq_len": seq_len,
            "hidden_dim": hidden_dim, "dtype": dtype, "flags": flags,
            "payload_len": payload_len, "request_id": request_id,
            "position": position, "header_size": size}


_KIND = {MSG_ACTIVATION: "activation", MSG_LOGITS: "logits", MSG_CONTROL: "control",
         MSG_TOKEN: "token", MSG_RESET: "reset", MSG_BYE: "bye", MSG_READY: "ready",
         MSG_HEARTBEAT: "heartbeat"}


@dataclass
class Message:
    """One assembled message (see BigBrainLink.recv_message).

    kind: activation | logits | control | token | reset | bye | ready |
    heartbeat. ``header`` is the header of the message's FIRST frame (for a
    chunked activation: the chunk that starts at ``position``)."""
    kind: str
    header: Dict = field(default_factory=dict)
    array: Optional[np.ndarray] = None
    control: Optional[Dict] = None
    tokens: Optional[List[int]] = None
    reason: str = ""

    @property
    def request_id(self) -> int:
        return int(self.header.get("request_id", 0))

    @property
    def position(self) -> int:
        return int(self.header.get("position", 0))

    @property
    def flags(self) -> int:
        return int(self.header.get("flags", 0))

    @property
    def is_prefill(self) -> bool:
        return bool(self.flags & FLAG_PREFILL)


def _array_shape(header: Dict) -> Tuple[Tuple[int, ...], int]:
    """(shape, itemsize) an ACTIVATION / LOGITS header announces. Raises BigBrainProtocolError for an unsupported dtype or a zero
    batch / hidden dimension (seq_len 0 is the terminator, an empty activation)."""
    np_dtype = _DTYPE_TO_NP.get(header["dtype"])
    if np_dtype is None:
        raise BigBrainProtocolError(
            f"dtype {header['dtype']} not supported (use fp16/fp32)")
    streams = (int(header["flags"]) & FLAG_STREAMS_MASK) >> FLAG_STREAMS_SHIFT
    want = (int(header["batch"]), int(header["seq_len"]), int(header["hidden_dim"]))
    if streams > 1:
        want = (want[0], want[1], streams, want[2])
    if want[0] < 1 or want[-1] < 1:
        raise BigBrainProtocolError(f"activation shape {want} has a zero dimension")
    return want, np.dtype(np_dtype).itemsize


def _decode_array(header: Dict, payload: bytes) -> np.ndarray:
    want, itemsize = _array_shape(header)
    # math.prod on Python ints: np.prod wraps in int64, which let a huge shape pass this check with an empty payload
    if len(payload) != math.prod(want) * itemsize:
        raise BigBrainProtocolError(
            f"payload {len(payload)} bytes does not match shape {want} x {itemsize} bytes")
    try:
        return np.frombuffer(payload, dtype=_DTYPE_TO_NP[header["dtype"]]).reshape(want)
    except (ValueError, TypeError) as exc:
        raise BigBrainProtocolError(f"cannot decode a {want} array: {exc}") from None


class BigBrainLink:
    """One directed TCP link carrying BigBrain frames.

    Usage (client side):  link = BigBrainLink(); link.connect(host, port)
    Usage (server side):  link = BigBrainLink(); port = link.listen_inbound(0)
                          ... peer connects ...
                          link.accept_peer()

    Both sides then use send_header / send_activation / send_control /
    send_tokens / send_logits / recv_frame / recv_activation / recv_message.
    A daemon heartbeat thread runs on every live link; the socket's receive
    timeout doubles as the peer watchdog. ``last_rx`` is the monotonic time
    the last frame (heartbeats included) arrived.
    """

    def __init__(self, watchdog_s: float = _DEFAULT_WATCHDOG_S,
                 max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> None:
        self._sock: Optional[socket.socket] = None
        self._watchdog_s = float(watchdog_s)
        # the largest frame (and the largest chunked message) this link accepts; checked on the header, before the payload is read
        self.max_frame_bytes = int(max_frame_bytes)
        self._seq_id = 0
        self._send_lock = threading.Lock()
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()
        self._dead = False
        self._connected = False
        self._drain_thread: Optional[threading.Thread] = None
        self.last_rx = time.monotonic()
        self.max_gap_s = 0.0          # the longest silence between two received frames on this link

    # ------------------------------------------------------------------ setup

    def _peer_label(self) -> str:
        try:
            return "%s:%s" % self._sock.getpeername()[:2]
        except (OSError, AttributeError, TypeError):
            return "?"

    def connect(self, host: str, port: int, timeout_s: float = 10.0,
                settle_s: float = 0.0) -> "BigBrainLink":
        """Connect to a peer's listening port. Returns self.

        settle_s > 0: then watch the new connection for that long without
        consuming anything. An ssh -L/-R forward accepts a connect even when its
        far side is not listening yet and closes it at once; that EOF raises
        ConnectionRefusedError here (retryable), so nodes may start in any
        order through tunnels. A real peer that has not accepted yet (still
        loading) sends nothing, and the connect stands."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        try:
            sock.connect((host, port))
            if settle_s > 0:
                sock.settimeout(settle_s)
                try:
                    if sock.recv(1, socket.MSG_PEEK) == b"":
                        raise ConnectionRefusedError(
                            f"{host}:{port} accepted and closed at once (an ssh tunnel "
                            f"whose far side is not listening yet?)")
                except socket.timeout:
                    pass
        except Exception:
            sock.close()
            raise
        self._adopt(sock)
        return self

    def listen_inbound(self, port: int = 0, bind_host: str = "0.0.0.0") -> int:
        """Bind and listen; returns the actual bound port (0 -> ephemeral)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
        sock.listen(1)
        self._sock = sock
        return sock.getsockname()[1]

    def accept_peer(self, timeout_s: Optional[float] = None) -> Tuple["BigBrainLink", Tuple[str, int]]:
        """Accept the single inbound peer. Returns (self, peer_addr)."""
        if self._sock is None:
            raise BigBrainProtocolError("listen_inbound() was not called")
        self._sock.settimeout(timeout_s if timeout_s is not None else self._watchdog_s)
        try:
            conn, addr = self._sock.accept()
        except socket.timeout:
            self.close()
            raise BigBrainPeerDead("timed out waiting for a peer")
        self._sock.close()
        self._adopt(conn)
        return self, addr

    @classmethod
    def from_socket(cls, sock: socket.socket,
                    watchdog_s: float = _DEFAULT_WATCHDOG_S) -> "BigBrainLink":
        """Wrap an already-connected socket (tests, advanced use)."""
        link = cls(watchdog_s=watchdog_s)
        link._adopt(sock)
        return link

    def _adopt(self, sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # The receive timeout doubles as the peer watchdog: any frame (even a
        # heartbeat) resets it by arriving; a dead peer never does.
        sock.settimeout(self._watchdog_s)
        self._sock = sock
        self._dead = False
        self._connected = True
        self._seq_id = 0
        self.last_rx = time.monotonic()
        self.max_gap_s = 0.0
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()

    # ------------------------------------------------------------------ io

    def _recv_exact(self, n: int) -> bytearray:
        # A bytearray, not bytes: arrays decoded from it are WRITABLE (torch
        # warns on read-only numpy memory) and a 100+ MB prefill payload is
        # not copied a second time.
        if n == 0:
            return bytearray()
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self._sock.recv(n - len(buf))
            except socket.timeout:
                raise BigBrainPeerDead(
                    f"no frame for {self._watchdog_s:.1f}s; declaring peer dead")
            except OSError as exc:
                raise BigBrainPeerDead(f"socket error: {exc}")
            if not chunk:
                raise BigBrainPeerDead("connection closed by peer")
            buf.extend(chunk)
        return buf

    def send_header(self, msgtype: int, *, batch: int = 0, seq_len: int = 0,
                    hidden_dim: int = 0, dtype: int = DT_FP16, flags: int = 0,
                    payload: bytes = b"", request_id: int = 0,
                    position: int = 0) -> None:
        """Send one header (+ optional payload) atomically."""
        if self._dead:
            raise BigBrainPeerDead("link is dead")
        with self._send_lock:
            self._seq_id += 1
            seq_id = self._seq_id
            data = pack_header(msgtype, batch=batch, seq_len=seq_len,
                               hidden_dim=hidden_dim, dtype=dtype, flags=flags,
                               payload_len=len(payload), seq_id=seq_id,
                               request_id=request_id, position=position) + payload
            # One sender at a time: the heartbeat thread must not splice its
            # frame into the middle of a large activation's sendall().
            try:
                self._sock.sendall(data)
            except socket.timeout:
                raise BigBrainPeerDead(f"send timed out after {self._watchdog_s:.1f}s")
            except (OSError, AttributeError) as exc:
                raise BigBrainPeerDead(f"socket error: {exc}")

    def send_activation(self, arr: np.ndarray, *, last_chunk: bool = True,
                        request_id: int = 0, position: int = 0, flags: int = 0,
                        msgtype: int = MSG_ACTIVATION) -> None:
        """Send one activation frame. arr shape: (batch, seq_len, hidden_dim),
        or (batch, seq_len, streams, hidden_dim) for a hyper-connection
        residual (streams <= 255). ``position`` = the sequence position of
        arr's first token; ``flags`` may carry FLAG_PREFILL (FLAG_LAST_CHUNK is
        set from last_chunk, the stream count from the shape)."""
        streams = 0
        if arr.ndim == 4:
            streams = int(arr.shape[2])
            if not 1 <= streams <= 255:
                raise BigBrainProtocolError(f"stream count {streams} out of range 1..255")
        elif arr.ndim != 3:
            raise BigBrainProtocolError(f"expected a 3-D or 4-D array, got {arr.ndim}-D")
        if arr.dtype == np.float32:
            dtype = DT_FP32
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        else:
            dtype = DT_FP16
            arr = np.ascontiguousarray(arr, dtype=np.float16)
        flags = (flags & ~(FLAG_LAST_CHUNK | FLAG_STREAMS_MASK))             | (FLAG_LAST_CHUNK if last_chunk else 0) | (streams << FLAG_STREAMS_SHIFT)
        self.send_header(msgtype, batch=int(arr.shape[0]),
                         seq_len=int(arr.shape[1]), hidden_dim=int(arr.shape[-1]),
                         dtype=dtype, flags=flags, payload=arr.tobytes(),
                         request_id=request_id, position=position)

    def send_control(self, obj: Dict, *, request_id: int = 0, position: int = 0) -> None:
        """Send one CONTROL frame: a JSON object (utf-8)."""
        if not isinstance(obj, dict):
            raise BigBrainProtocolError("a control payload must be a JSON object")
        payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_header(MSG_CONTROL, payload=payload, request_id=request_id,
                         position=position)

    def send_tokens(self, ids: Sequence[int], *, request_id: int = 0,
                    position: int = 0) -> None:
        """Send token ids (uint32). ``position`` = the sequence position of
        the first id."""
        arr = np.asarray(list(ids), dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() > 0xFFFFFFFF):
            raise BigBrainProtocolError(f"token id out of uint32 range: {ids}")
        self.send_header(MSG_TOKEN, batch=1, seq_len=int(arr.size), hidden_dim=1,
                         dtype=DT_U32, payload=arr.astype("<u4").tobytes(),
                         request_id=request_id, position=position)

    def send_logits(self, row: np.ndarray, *, request_id: int = 0,
                    position: int = 0) -> None:
        """Send one logits row as fp32 (1, 1, vocab) (parity/debug path)."""
        arr = np.asarray(row, dtype=np.float32).reshape(1, 1, -1)
        self.send_activation(arr, request_id=request_id, position=position,
                             msgtype=MSG_LOGITS)

    def _malformed(self, reason: str) -> BigBrainProtocolError:
        """A peer sent something this build cannot accept: the stream can no longer be trusted (or is desynchronised), so the link
        is marked dead and its connection closed at once -- the next call fails fast and the peer sees EOF. Returns the error for
        the caller to raise (security review M3)."""
        self._dead = True
        self._hb_stop.set()
        logger.error("link %s: %s; closing it", self._peer_label(), reason)
        sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.close()
        return BigBrainProtocolError(reason)

    def _check_frame(self, header: Dict) -> None:
        """Header-level checks, done BEFORE the payload is read: its size against this link's limit and against what the message
        type can carry. Raises BigBrainProtocolError."""
        mt, n = header["msgtype"], int(header["payload_len"])
        if n > self.max_frame_bytes:
            raise BigBrainProtocolError(
                f"a {_KIND.get(mt, mt)} frame of {n} bytes is over this link's limit of {self.max_frame_bytes} bytes "
                f"(--max-frame-mib raises it)")
        if mt in (MSG_ACTIVATION, MSG_LOGITS):
            want, itemsize = _array_shape(header)
            if math.prod(want) * itemsize != n:
                raise BigBrainProtocolError(
                    f"payload {n} bytes does not match shape {want} x {itemsize} bytes")
        elif mt == MSG_TOKEN:
            if header["dtype"] != DT_U32 or n != 4 * int(header["seq_len"]):
                raise BigBrainProtocolError(
                    f"bad token frame: dtype {header['dtype']}, {header['seq_len']} ids, {n} payload bytes")
        elif mt == MSG_CONTROL:
            if n > MAX_CONTROL_BYTES:
                raise BigBrainProtocolError(f"a control frame of {n} bytes is over the limit of {MAX_CONTROL_BYTES}")
        elif mt in (MSG_HEARTBEAT, MSG_READY, MSG_BYE, MSG_RESET):
            if n > MAX_SMALL_PAYLOAD:
                raise BigBrainProtocolError(f"a {_KIND[mt]} frame of {n} bytes is over the limit of {MAX_SMALL_PAYLOAD}")
        else:
            raise BigBrainProtocolError(f"unknown msgtype {mt}")

    def recv_frame(self) -> Tuple[int, Dict, bytes]:
        """Receive one frame. Returns (msgtype, header_dict, payload). A malformed header (bad magic or version, a payload over
        the limit or not matching its shape, an unknown type) closes the link and raises BigBrainProtocolError."""
        if self._dead:
            raise BigBrainPeerDead("link is dead")
        base = self._recv_exact(HEADER_SIZE_V1)
        if base[:4] == MAGIC and base[4] >= 2 and base[4] in SUPPORTED_VERSIONS:
            base += self._recv_exact(HEADER_EXT_SIZE)
        try:
            header = unpack_header(base)
            self._check_frame(header)
        except BigBrainProtocolError as exc:
            # Unknown framing, or a frame this node will not buffer: everything after this point is unreadable.
            raise self._malformed(str(exc)) from None
        payload = self._recv_exact(int(header["payload_len"]))
        now = time.monotonic()
        gap = now - self.last_rx
        if gap > self.max_gap_s:
            self.max_gap_s = gap
        if gap >= GAP_WARN_S:
            # a peer heartbeats every 0.5 s, so a long gap is a stall (a long compute step, a stalled tunnel) that the
            # watchdog would have called a dead peer at --watchdog-s: say so BEFORE it kills a link, with the length
            logger.warning("link %s: %.1f s between frames (heartbeats come every %.1f s; watchdog %.1f s)",
                           self._peer_label(), gap, _HEARTBEAT_INTERVAL_S, self._watchdog_s)
        self.last_rx = now
        if header["msgtype"] == MSG_BYE:
            self._dead = True
        return header["msgtype"], header, payload

    def recv_activation(self, expected_shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Receive a (possibly multi-frame) activation; HEARTBEATs are skipped.
        Any other message type is a protocol error here (use recv_message)."""
        chunks: List[np.ndarray] = []
        while True:
            msgtype, header, payload = self.recv_frame()
            if msgtype == MSG_HEARTBEAT:
                continue
            if msgtype == MSG_RESET:
                raise BigBrainProtocolError(
                    f"peer reset: {payload.decode('utf-8', 'replace')}")
            if msgtype != MSG_ACTIVATION:
                raise BigBrainProtocolError(
                    f"unexpected msgtype {msgtype} in activation stream")
            chunks.append(self._decode(header, payload))
            if header["flags"] & FLAG_LAST_CHUNK:
                break
            self._check_assembled(chunks)
        result = self._assemble(chunks)
        if expected_shape is not None and tuple(result.shape) != tuple(expected_shape):
            raise BigBrainProtocolError(
                f"shape mismatch: got {tuple(result.shape)}, expected {tuple(expected_shape)}")
        return result

    def recv_message(self, include_heartbeats: bool = False) -> Message:
        """Receive one whole message of ANY type: a chunked ACTIVATION/LOGITS
        is assembled, a CONTROL payload parsed (must be a JSON object), TOKEN
        ids decoded. HEARTBEATs are skipped unless include_heartbeats."""
        chunks: List[np.ndarray] = []
        first: Optional[Dict] = None
        chunk_type: Optional[int] = None
        while True:
            msgtype, header, payload = self.recv_frame()
            if msgtype == MSG_HEARTBEAT:
                if include_heartbeats and not chunks:
                    return Message("heartbeat", header)
                continue
            if chunks and msgtype != chunk_type:
                raise self._malformed(
                    f"msgtype {msgtype} interleaved inside a chunked "
                    f"{_KIND.get(chunk_type, chunk_type)} message")
            if msgtype in (MSG_ACTIVATION, MSG_LOGITS):
                chunks.append(self._decode(header, payload))
                if first is None:
                    first, chunk_type = header, msgtype
                if header["flags"] & FLAG_LAST_CHUNK:
                    return Message(_KIND[msgtype], first, array=self._assemble(chunks))
                self._check_assembled(chunks)
                continue
            if msgtype == MSG_CONTROL:
                try:
                    obj = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, ValueError, RecursionError) as exc:
                    raise self._malformed(f"control payload is not JSON: {type(exc).__name__}: {str(exc)[:200]}") from None
                if not isinstance(obj, dict):
                    raise self._malformed(
                        f"control payload must be a JSON object, got {type(obj).__name__}")
                return Message("control", header, control=obj)
            if msgtype == MSG_TOKEN:
                n = int(header["seq_len"])
                if header["dtype"] != DT_U32 or len(payload) != 4 * n:
                    raise self._malformed(
                        f"bad token frame: dtype {header['dtype']}, {n} ids, "
                        f"{len(payload)} payload bytes")
                ids = np.frombuffer(payload, dtype="<u4").astype(np.int64).tolist()
                return Message("token", header, tokens=ids)
            if msgtype == MSG_RESET:
                return Message("reset", header, reason=payload.decode("utf-8", "replace"))
            if msgtype == MSG_BYE:
                return Message("bye", header)
            if msgtype == MSG_READY:
                return Message("ready", header)
            raise self._malformed(f"unknown msgtype {msgtype}")

    def _decode(self, header: Dict, payload: bytes) -> np.ndarray:
        try:
            return _decode_array(header, payload)
        except BigBrainProtocolError as exc:
            raise self._malformed(str(exc)) from None

    def _check_assembled(self, chunks: List[np.ndarray]) -> None:
        """A chunked message may not grow past the frame limit either (chunks without a last one would otherwise buffer forever)."""
        total = sum(int(c.nbytes) for c in chunks)
        if total > self.max_frame_bytes:
            raise self._malformed(f"a chunked message of {total} bytes so far is over this link's limit of "
                                  f"{self.max_frame_bytes} bytes")

    def _assemble(self, chunks: List[np.ndarray]) -> np.ndarray:
        if len(chunks) == 1:
            return chunks[0]
        try:
            return np.concatenate(chunks, axis=1)
        except ValueError as exc:            # chunks whose shapes do not line up
            raise self._malformed(f"the chunks of one message do not fit together: {exc}") from None

    # ------------------------------------------------------------- lifecycle

    def reset(self, reason: str = "", *, request_id: int = 0) -> None:
        """Tell the peer to abort the current generation."""
        self.send_header(MSG_RESET, payload=reason.encode("utf-8"),
                         request_id=request_id)

    def start_drain(self) -> None:
        """For a link this side only SENDS on (node i's outbound, the last
        node's token link): read and discard what the peer sends back (its
        heartbeats), so the kernel buffers never fill and the peer's heartbeat
        thread never blocks. EOF or a socket error marks the link dead -- the
        next send then fails fast instead of the far end timing out.
        Never use it on a link this side also reads (it would eat frames)."""
        if self._drain_thread is not None or not self._connected or self._sock is None:
            return
        self._drain_thread = threading.Thread(target=self._drain_loop,
                                              args=(self._sock,), daemon=True)
        self._drain_thread.start()

    def _drain_loop(self, sock: socket.socket) -> None:
        while not self._hb_stop.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue  # an idle peer between heartbeats is not a dead one
            except OSError:
                break
            if not data:
                break
            self.last_rx = time.monotonic()
        self._dead = True

    def close(self) -> None:
        """Send BYE and close. Idempotent."""
        self._hb_stop.set()
        if self._sock is not None:
            try:
                if not self._dead:
                    self.send_header(MSG_BYE)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._dead = True
        self._connected = False
        for attr in ("_hb_thread", "_drain_thread"):
            th = getattr(self, attr)
            if th is not None and th is not threading.current_thread():
                th.join(timeout=1.0)
                setattr(self, attr, None)

    @property
    def is_dead(self) -> bool:
        return self._dead

    @property
    def connected(self) -> bool:
        """True once a peer is attached (connect/accept) and until close()."""
        return self._connected

    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(_HEARTBEAT_INTERVAL_S):
            try:
                self.send_header(MSG_HEARTBEAT)
            except Exception:
                break


class BigBrainRing:
    """Ordered-pipeline bookkeeping: inbound link + outbound link(s).

    Node 0 has inbound_port=None and is the coordinator; the last node has
    last=True and no outbound peers.
    """

    def __init__(self, name: str, inbound_port: Optional[int] = None,
                 outbound_peers: Optional[List[Tuple[str, int]]] = None,
                 last: bool = False,
                 watchdog_s: float = _DEFAULT_WATCHDOG_S) -> None:
        self.name = name
        self.inbound_port = inbound_port
        self.outbound_peers = list(outbound_peers or [])
        self.last = last
        self._watchdog_s = watchdog_s
        self.inbound: Optional[BigBrainLink] = None
        self.outbounds: List[BigBrainLink] = []
        self._closed = False

    def listen_inbound(self) -> Optional[int]:
        """Bind the inbound port (None for the first node). Returns bound port."""
        if self.inbound_port is None:
            return None
        link = BigBrainLink(watchdog_s=self._watchdog_s)
        port = link.listen_inbound(self.inbound_port)
        self.inbound = link
        return port

    def accept_inbound(self, timeout_s: Optional[float] = None) -> Tuple[str, int]:
        if self.inbound is None:
            raise BigBrainProtocolError("no inbound port configured")
        _, addr = self.inbound.accept_peer(timeout_s)
        return addr

    def connect_outbound(self, timeout_s: float = 10.0) -> None:
        for host, port in self.outbound_peers:
            link = BigBrainLink(watchdog_s=self._watchdog_s)
            link.connect(host, port, timeout_s=timeout_s)
            self.outbounds.append(link)

    def recv_inbound(self, expected_shape: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if self.inbound is None:
            raise BigBrainProtocolError("no inbound link")
        return self.inbound.recv_activation(expected_shape)

    def send_outbound(self, arr: np.ndarray, last_chunk: bool = True) -> None:
        for link in self.outbounds:
            link.send_activation(arr, last_chunk=last_chunk)

    def broadcast_reset(self, reason: str = "") -> None:
        for link in ([self.inbound] if self.inbound else []) + self.outbounds:
            try:
                link.reset(reason)
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for link in ([self.inbound] if self.inbound else []) + self.outbounds:
            link.close()
