# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""tools/bigbrain/server.py -- an OpenAI-compatible endpoint on BigBrain node 0.

Runs IN PLACE OF `python -m tools.bigbrain.worker` on node 0 (it takes every
node-placement flag the worker takes, plus the HTTP flags below) and drives
the same SliceWorker, so any OpenAI-compatible client calls the big model like any
other endpoint. Nodes 1..T-1 run the plain worker, which serves any number of
requests until node 0 sends its terminator.

Routes
  GET  /health                 open (no auth): status, queue depth, link health
  GET  /v1/models              the one model
  POST /v1/chat/completions    messages -> the model folder's chat template;
                               stream (SSE) or not; temperature, top_p, top_k,
                               min_p, seed, max_tokens / max_completion_tokens,
                               stop, stream_options.include_usage,
                               chat_template_kwargs, tools (passed to the template)
  POST /v1/completions         raw prompt (string, or token ids)
  POST /v1/bigbrain/generate   debug/parity: prompt_ids in, token ids (+ every
                               step's fp32 logits, base64) out; greedy by default

Auth: every /v1 route needs `Authorization: Bearer <key>` (or `x-api-key`),
compared with secrets.compare_digest. The key comes from --api-key-file, else
the env var named by --api-key-env (default BIGBRAIN_API_KEY); with neither the
server refuses to start unless --insecure-no-auth (loopback tests only).

Scheduling. Single-sequence mode (--slots 1, no speculation: the default):
one engine thread serves requests in FIFO order from a bounded queue
(--queue-size waiting requests). BATCHED mode (--slots N > 1 and/or
--spec-draft ngram|model): up to N requests run at once through
engine.PipelineEngine, which keeps the pipeline full (several sequences in
flight) and, for greedy requests, speculates -- ngram (prompt lookup) or a
draft model. Requests beyond N wait in the same bounded queue. Either way,
when the queue is full the answer is 503 with Retry-After (--retry-after). A
client that disconnects cancels its request (between tokens; queued requests
are skipped). Request extensions (any route): "spec": "none"|"ngram"|"draft",
"spec_k": 1..15 -- greedy requests only.

Extra route: GET /v1/bigbrain/metrics (decode/prefill tok/s, first-token
latency, speculation acceptance, per-node busy/wait/send % and VRAM; POST the
same path with {"reset": true} to read and zero the window).

Extra route: GET /v1/bigbrain/stats[?recent=N] -- the per-request record the
status viewer reads (docs/API.md): prompt and
completion tokens and times of every finished request (last 200 kept), the last
request's decode/prefill tok/s, the average over the last 20, the aggregate over
all slots, served/failed/cancelled and the tag of the job that is using the
model. Unlike /metrics it costs the pipeline nothing (node 0's memory only), so
it is safe to poll every few seconds. /health carries the same without `recent`
as `stats`, so a reader without the key can show it too. A client names its job
with two request headers: X-BigBrain-Job (label) and X-BigBrain-Job-Progress (n/m).

Chat template: chat_template.jinja in the model folder, else
tokenizer_config.json "chat_template" (a string, or a named list -> "default"),
rendered with jinja2 in a sandbox the way transformers renders it
(trim_blocks, lstrip_blocks, loopcontrols, raise_exception, tojson,
strftime_now; bos_token/eos_token/... and chat_template_kwargs as variables).
FALLBACK when there is no template or jinja2 is missing: ChatML
(<|im_start|>role\\ncontent<|im_end|>\\n ... <|im_start|>assistant\\n) -- exact
for the Qwen family, an approximation for anything else.

Sampling defaults when a request omits them: the model's generation_config.json
(temperature/top_p/top_k/min_p; do_sample=false -> greedy), else temperature 1,
top_p 1, top_k off. top_k <= 0 means off. Sampling itself runs on the LAST node.

Output: the last node returns token ids; this node detokenizes INCREMENTALLY
(a window from the last emitted boundary, held back while it ends in U+FFFD, so
a multi-byte character split across byte-level tokens is emitted whole) and
stops on any EOS id (generation_config + tokenizer) or a stop string (text that
could still become a stop string is held back, never streamed).
usage.completion_tokens counts every sampled token, the EOS included.
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime
import itertools
import json
import logging
import os
import queue
import re
import secrets
import select
import signal
import socket
import sys
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .engine import EngineConfig, EngineDead, PipelineEngine
from .spec import SpecConfig
from .worker import (SamplingSettings, add_node_args, effective_max_draft, effective_slots,
                     encode_prompt, eos_token_ids, worker_from_args)

log = logging.getLogger("bigbrain.server")

DEFAULT_HTTP_PORT = 16890
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_STOP_STRINGS = 16
MAX_LOGIT_TOKENS = 512
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


# --------------------------------------------------------------------------
# incremental detokenization + stop strings
# --------------------------------------------------------------------------

class IncrementalDetokenizer:
    """Streams text from token ids. Decodes the window from the last emitted
    boundary (prefix) to the end and emits what is new past the previous read
    point -- but holds it back while it ends in U+FFFD (a UTF-8 character
    split across byte-level tokens). context_ids (the prompt's tail) seed the
    window so a SentencePiece-style first token keeps its leading space."""

    def __init__(self, decode_fn: Callable[[List[int]], str], context_ids: Sequence[int] = ()):
        self.decode = decode_fn
        self.ids: List[int] = [int(t) for t in context_ids]
        self.prefix = 0
        self.read = len(self.ids)

    def push(self, tok: int) -> str:
        self.ids.append(int(tok))
        prefix_text = self.decode(self.ids[self.prefix:self.read])
        new_text = self.decode(self.ids[self.prefix:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            self.prefix, self.read = self.read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""

    def flush(self) -> str:
        """End of stream: whatever is still held (even an incomplete char)."""
        prefix_text = self.decode(self.ids[self.prefix:self.read])
        new_text = self.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return new_text[len(prefix_text):] if len(new_text) > len(prefix_text) else ""


class StopMatcher:
    """Finds stop strings across streamed deltas. feed() returns (text safe to
    emit, hit). Text that is a prefix of some stop string is held back until it
    is decided; on a hit everything from the stop string on is dropped."""

    def __init__(self, stops: Sequence[str]):
        self.stops = [s for s in stops if s]
        self.buf = ""

    def feed(self, text: str) -> Tuple[str, bool]:
        if not self.stops:
            return text, False
        self.buf += text
        hits = [i for i in (self.buf.find(s) for s in self.stops) if i >= 0]
        if hits:
            out, self.buf = self.buf[:min(hits)], ""
            return out, True
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        cut = len(self.buf) - hold
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out, False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


# --------------------------------------------------------------------------
# chat template
# --------------------------------------------------------------------------

def _jinja_env():
    """A transformers-compatible sandboxed jinja2 environment, or None when
    jinja2 is not installed (then the ChatML fallback is used)."""
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        return None

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env


def _read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def _tok_str(v) -> str:
    if isinstance(v, dict):
        v = v.get("content")
    return v if isinstance(v, str) else ""


_RESERVED_TEMPLATE_VARS = {"messages", "add_generation_prompt", "tools"}


class ChatTemplate:
    def __init__(self, template: Optional[str], *, special: Optional[Dict[str, str]] = None,
                 source: str = "fallback-chatml"):
        self.template = template
        self.special = dict(special or {})
        self.source = source if template else "fallback-chatml"
        self._compiled = None

    @property
    def bos_token(self) -> str:
        return self.special.get("bos_token", "")

    @classmethod
    def from_model_dir(cls, model_dir: str) -> "ChatTemplate":
        tc = _read_json(os.path.join(model_dir, "tokenizer_config.json")) or {}
        special = {k: _tok_str(tc.get(k)) for k in ("bos_token", "eos_token", "unk_token", "pad_token")}
        jinja_path = os.path.join(model_dir, "chat_template.jinja")
        if os.path.isfile(jinja_path):
            with open(jinja_path, encoding="utf-8") as f:
                return cls(f.read(), special=special, source="chat_template.jinja")
        t = tc.get("chat_template")
        if isinstance(t, list):
            named = {d.get("name"): d.get("template") for d in t if isinstance(d, dict)}
            t = named.get("default") or next((v for v in named.values() if isinstance(v, str)), None)
        if isinstance(t, str) and t:
            return cls(t, special=special, source="tokenizer_config.json")
        return cls(None, special=special)

    def render(self, messages: List[Dict], add_generation_prompt: bool = True,
               tools: Optional[list] = None, **kwargs) -> str:
        env = _jinja_env() if self.template else None
        if env is None:
            return self._chatml(messages, add_generation_prompt)
        if self._compiled is None:
            self._compiled = env.from_string(self.template)
        variables = dict(self.special)
        variables.update({k: v for k, v in kwargs.items() if k not in _RESERVED_TEMPLATE_VARS})
        variables.update(messages=messages, add_generation_prompt=add_generation_prompt)
        if tools is not None:
            variables["tools"] = tools
        import jinja2
        try:
            return self._compiled.render(**variables)
        except jinja2.exceptions.TemplateError as exc:
            raise ValueError(f"chat template: {exc}") from exc

    @staticmethod
    def _chatml(messages: List[Dict], add_generation_prompt: bool) -> str:
        out = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n"
                      for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def normalize_messages(messages) -> List[Dict]:
    """OpenAI messages -> template input: content flattened to a string (text
    parts joined); a non-text part is refused (this endpoint is text-only)."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or not isinstance(m.get("role"), str) or not m["role"]:
            raise ValueError(f"messages[{i}] needs a string 'role'")
        m = dict(m)
        c = m.get("content")
        if c is None:
            m["content"] = ""
        elif isinstance(c, list):
            parts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                else:
                    kind = p.get("type") if isinstance(p, dict) else type(p).__name__
                    raise ValueError(f"messages[{i}]: '{kind}' content is not supported "
                                     f"(text-only endpoint)")
            m["content"] = "".join(parts)
        elif not isinstance(c, str):
            raise ValueError(f"messages[{i}].content must be a string or a list of text parts")
        out.append(m)
    return out


def load_generation_defaults(model_dir: str) -> Dict:
    """Sampling defaults from generation_config.json (do_sample=false -> greedy)."""
    g = _read_json(os.path.join(model_dir, "generation_config.json"))
    if not g:
        return {}
    if g.get("do_sample") is False:
        return {"temperature": 0.0}
    return {k: g[k] for k in ("temperature", "top_p", "top_k", "min_p")
            if isinstance(g.get(k), (int, float)) and not isinstance(g.get(k), bool)}


def resolve_api_key(key_file: Optional[str], env_name: Optional[str], insecure: bool,
                    token: Optional[str] = None) -> Optional[str]:
    if token:
        return token.strip()
    if key_file:
        with open(key_file, encoding="utf-8") as f:
            key = f.read().strip()
        if not key:
            raise SystemExit(f"API key file {key_file} is empty")
        return key
    key = os.environ.get(env_name, "").strip() if env_name else ""
    if key:
        return key
    if insecure:
        return None
    raise SystemExit(f"no API key: pass --api-key-file PATH or set ${env_name} "
                     f"(--insecure-no-auth only for a loopback test)")


# --------------------------------------------------------------------------
# engine adapter + scheduler
# --------------------------------------------------------------------------

class WorkerEngine:
    """The server's view of node 0's SliceWorker."""

    def __init__(self, worker, model_dir: str, model_name: Optional[str] = None,
                 token_timeout_s: float = 600.0, batch: Optional[PipelineEngine] = None):
        self.w = worker
        self.batch = batch          # the batched engine (several sequences, speculation) or None
        self.model_name = model_name or os.path.basename(os.path.normpath(model_dir))
        self.max_context = worker.max_num_tokens
        self.template = ChatTemplate.from_model_dir(model_dir)
        self.eos_ids = eos_token_ids(worker.config, worker.tokenizer)
        tc = _read_json(os.path.join(model_dir, "tokenizer_config.json")) or {}
        self.add_bos = bool(tc.get("add_bos_token"))
        self.token_timeout_s = token_timeout_s

    def encode(self, text: str, special: bool = True) -> List[int]:
        bos = self.template.bos_token
        add_bos = self.add_bos and not (bos and text.startswith(bos))
        return encode_prompt(self.w.tokenizer, text, add_bos=add_bos,
                             encode_special_tokens=special)

    def decode(self, ids: Sequence[int]) -> str:
        import torch
        return self.w.tokenizer.decode(torch.tensor(list(ids), dtype=torch.long))

    @property
    def concurrency(self) -> int:
        """How many requests can run at once."""
        return self.batch.slots if self.batch is not None else 1

    def generate(self, prompt_ids, settings, max_new, request_id=None, return_logits=False, *,
                 stop_ids=(), spec=None, spec_k=None, diag=False, info=None):
        if self.batch is not None:
            return self.batch.generate(prompt_ids, settings, max_new, request_id=request_id,
                                       return_logits=return_logits, token_timeout_s=self.token_timeout_s,
                                       stop_ids=stop_ids, spec=spec, spec_k=spec_k, diag=diag, info=info)
        return self.w.generate(prompt_ids, settings, max_new, request_id=request_id,
                               return_logits=return_logits, token_timeout_s=self.token_timeout_s)

    def report(self, reset: bool = False) -> Dict:
        """Metrics for GET/POST /v1/bigbrain/metrics."""
        if self.batch is not None:
            return {"mode": "batched", **self.batch.report(reset=reset)}
        return {"mode": "single-sequence", "engine": None,
                "nodes": [self.w.node_stats(reset=reset)]}

    def health(self) -> Dict:
        if self.batch is not None:
            h = self.batch.health()
            links = {}
            for name in ("outbound", "token_link"):
                link = getattr(self.w, name, None)
                if link is not None:
                    links[name] = "dead" if link.is_dead else "ok"
            return {"ok": bool(h["ok"]) and all(v == "ok" for v in links.values()),
                    "nodes": self.w.total, "links": links, "recurrent": self.w.recurrent,
                    "batched": True, "ready": self.batch.ready.is_set(),
                    **{k: h[k] for k in ("slots", "active", "queued", "dead", "spec", "inflight_cap")}}
        links = {}
        for name in ("outbound", "token_link"):
            link = getattr(self.w, name, None)
            if link is not None:
                links[name] = "dead" if link.is_dead else "ok"
        return {"ok": all(v == "ok" for v in links.values()), "nodes": self.w.total,
                "links": links, "recurrent": self.w.recurrent, "batched": False}


@dataclass
class GenRequest:
    rid: int
    prompt_ids: List[int]
    settings: SamplingSettings
    max_tokens: int
    stop: List[str] = field(default_factory=list)
    ignore_eos: bool = False
    return_logits: bool = False
    spec: Optional[str] = None          # none | ngram | draft (batched mode; greedy only)
    spec_k: Optional[int] = None
    diag: bool = False                  # report top-1 minus top-2 logit margins per token
    events: "queue.Queue" = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    tokens: List[int] = field(default_factory=list)
    logits: List[np.ndarray] = field(default_factory=list)
    info: Dict = field(default_factory=dict)
    # the stats record (StatsRecorder): clocks are time.monotonic(), set by the scheduler
    t_created: float = field(default_factory=time.monotonic)
    t_start: Optional[float] = None     # a slot took it
    t_first: Optional[float] = None     # its first token came back
    route: str = ""                     # chat | completions | generate
    job: Optional[Dict] = None          # {"label", "unit", "of"} from the X-BigBrain-Job headers


# --------------------------------------------------------------------------
# the per-request stats record
# --------------------------------------------------------------------------

JOB_LABEL_MAX = 120
_JOB_PROGRESS = re.compile(r"^\s*(\d{1,6})\s*/\s*(\d{1,6})\s*$")


def parse_job_tag(label, progress=None) -> Optional[Dict]:
    """The job a request names itself with (headers X-BigBrain-Job / -Progress), or None.
    The label is printable text only, at most JOB_LABEL_MAX characters; a progress that is
    not "n/m" is dropped (never an error: a tag must not fail a request)."""
    if not isinstance(label, str):
        return None
    text = "".join(ch for ch in label if ch.isprintable()).strip()[:JOB_LABEL_MAX].strip()
    if not text:
        return None
    unit = of = None
    if isinstance(progress, str):
        m = _JOB_PROGRESS.match(progress)
        if m:
            unit, of = int(m.group(1)), int(m.group(2))
    return {"label": text, "unit": unit, "of": of}


def _iso(epoch: float) -> str:
    try:
        return datetime.datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):   # a clock set to nonsense must not break /health
        return str(epoch)


class StatsRecorder:
    """What a status viewer shows: tokens and times of every finished request.

    The scheduler measures each request on its own monotonic clock (arrival, a slot took
    it, first token, last token) and calls record() once, from the engine thread that ran
    it. Everything is node 0's memory: reading it never touches the pipeline.

    Definitions (docs/API.md, GET /v1/bigbrain/stats):
      decode tok/s     (completion_tokens - 1) / (last token - first token): the rate after the
                       first token, one request's own sequence (needs >= 2 tokens)
      prefill tok/s    prompt_tokens / (first token - a slot took it): includes the first step
      first_token_s    arrival -> first token (the queue wait included)
      avg              token-weighted over the last `avg_n` finished requests that have a rate:
                       sum(completion_tokens - 1) / sum(decode seconds)
      aggregate        every slot together over the last minute OF ACTIVITY (anchored at the
                       newest finished request, so an idle server keeps showing its last
                       busy minute): generated tokens / the time at least one request decoded
      per_sequence     the plain mean of the per-request decode rates
    """

    RECENT = 200
    AVG_N = 20
    AGG_WINDOW_S = 60.0

    def __init__(self, recent: int = RECENT, avg_n: int = AVG_N, wall: Callable[[], float] = time.time,
                 mono: Callable[[], float] = time.monotonic):
        self._lock = threading.Lock()
        self._recent: "collections.deque[Dict]" = collections.deque(maxlen=max(1, int(recent)))
        self.avg_n = max(1, int(avg_n))
        self._wall, self._mono = wall, mono
        self.t0_wall, self.t0_mono = wall(), mono()
        self.cancelled = 0
        self.prompt_total = 0
        self.completion_total = 0
        self._ids = itertools.count(1)
        self._job: Optional[Dict] = None

    # ---- writing

    def note_job(self, job: Optional[Dict]) -> None:
        """A request that names a job arrived (or finished): that job is the current one."""
        if job:
            with self._lock:
                self._job = {"label": job["label"], "unit": job.get("unit"), "of": job.get("of"),
                             "updated_at": self._wall()}

    def record(self, req: "GenRequest", finish: str, completion_tokens: int) -> None:
        """One request ended (finish: stop | length | cancelled | error)."""
        end = self._mono()
        prompt = len(req.prompt_ids)
        c = max(0, int(completion_tokens))
        t_first, t_start = req.t_first, req.t_start
        queue_s = ((t_start if t_start is not None else end) - req.t_created)
        first_s = (t_first - req.t_created) if t_first is not None else None
        prefill_s = (t_first - t_start) if (t_first is not None and t_start is not None) else None
        dec_s = (end - t_first) if t_first is not None else None
        rate = ((c - 1) / dec_s) if (dec_s is not None and dec_s > 0 and c >= 2) else None
        prate = (prompt / prefill_s) if (prefill_s is not None and prefill_s > 0) else None
        rec = {"id": next(self._ids), "at": self._wall(), "route": req.route or None,
               "prompt_tokens": prompt, "completion_tokens": c, "queue_s": round(max(0.0, queue_s), 3),
               "first_token_s": round(first_s, 3) if first_s is not None else None,
               "total_s": round(end - req.t_created, 3),
               "decode_tok_s": round(rate, 2) if rate is not None else None,
               "prefill_tok_s": round(prate, 2) if prate is not None else None,
               "finish": finish, "spec": (req.info or {}).get("spec_mode") or req.spec or "none",
               "job": (req.job or {}).get("label"),
               # private (stripped on output): exact clocks for the averages
               "_dec_s": dec_s, "_prefill_s": prefill_s, "_t_first": t_first, "_t_end": end}
        with self._lock:
            self._recent.append(rec)
            if t_first is not None:       # a prompt that never produced a token was not served
                self.prompt_total += prompt
            self.completion_total += c
            if finish == "cancelled":
                self.cancelled += 1
        if req.job:
            self.note_job(req.job)

    # ---- reading

    @staticmethod
    def _public(rec: Dict) -> Dict:
        return {k: v for k, v in rec.items() if not k.startswith("_")}

    @staticmethod
    def _last(recs: List[Dict], key: str) -> Optional[float]:
        for r in reversed(recs):
            if r.get(key) is not None:
                return r[key]
        return None

    def snapshot(self, served: int = 0, failed: int = 0, running: int = 0, queued: int = 0,
                 recent: int = 0) -> Dict:
        """The stats record. served/failed/running/queued come from the scheduler (one set of
        counters for /health and for this); `recent` = how many finished requests to list."""
        with self._lock:
            recs = list(self._recent)
            cancelled, ptot, ctot = self.cancelled, self.prompt_total, self.completion_total
            job = dict(self._job) if self._job else None
        now = self._mono()
        window = [r for r in recs if r["decode_tok_s"] is not None][-self.avg_n:]
        dec_tok = sum(r["completion_tokens"] - 1 for r in window)
        dec_sec = sum(r["_dec_s"] for r in window)
        pre = [r for r in recs if r["prefill_tok_s"] is not None][-self.avg_n:]
        pre_sec = sum(r["_prefill_s"] for r in pre)
        first = [r["first_token_s"] for r in recs if r["first_token_s"] is not None][-self.avg_n:]
        out = {
            "since": _iso(self.t0_wall), "uptime_s": int(max(0.0, now - self.t0_mono)),
            "requests": {"served": int(served), "failed": int(failed), "cancelled": cancelled,
                         "in_flight": int(running), "queued": int(queued)},
            "tokens": {"prompt_total": ptot, "completion_total": ctot},
            "decode_tok_s": {
                "last": self._last(recs, "decode_tok_s"),
                "avg": round(dec_tok / dec_sec, 2) if dec_sec > 0 else None,
                "avg_n": len(window) if window else None,
                "aggregate": self._aggregate(recs),
                "per_sequence": round(sum(r["decode_tok_s"] for r in window) / len(window), 2) if window else None},
            "prefill_tok_s": {
                "last": self._last(recs, "prefill_tok_s"),
                "avg": round(sum(r["prompt_tokens"] for r in pre) / pre_sec, 2) if pre_sec > 0 else None},
            "first_token_s": {
                "last": self._last(recs, "first_token_s"),
                "avg": round(sum(first) / len(first), 3) if first else None},
            "last_request": ({"at": recs[-1]["at"], "prompt_tokens": recs[-1]["prompt_tokens"],
                              "completion_tokens": recs[-1]["completion_tokens"],
                              "total_s": recs[-1]["total_s"], "finish": recs[-1]["finish"]} if recs else None),
            "job": job,
        }
        if recent > 0:
            out["recent"] = [self._public(r) for r in recs[-int(recent):]]
        return out

    def _aggregate(self, recs: List[Dict]) -> Optional[float]:
        timed = [r for r in recs if r["_t_first"] is not None and r["_t_end"] > r["_t_first"]]
        if not timed:
            return None
        hi = max(r["_t_end"] for r in timed)
        lo = hi - self.AGG_WINDOW_S
        tokens, spans = 0.0, []
        for r in timed:
            a, b = max(r["_t_first"], lo), r["_t_end"]
            if b <= a:
                continue
            tokens += max(0, r["completion_tokens"] - 1) * (b - a) / (r["_t_end"] - r["_t_first"])
            spans.append((a, b))
        spans.sort()
        busy, cur_a, cur_b = 0.0, None, None
        for a, b in spans:
            if cur_b is None or a > cur_b:
                if cur_b is not None:
                    busy += cur_b - cur_a
                cur_a, cur_b = a, b
            else:
                cur_b = max(cur_b, b)
        if cur_b is not None:
            busy += cur_b - cur_a
        return round(tokens / busy, 2) if busy >= 1.0 and tokens > 0 else None


class Scheduler:
    """FIFO from a bounded queue into `workers` engine threads: 1 in single-sequence
    mode, one per slot in batched mode (each thread drives one request's generator; the
    PipelineEngine multiplexes them over the pipeline)."""

    def __init__(self, engine, queue_size: int = 8, workers: int = 1,
                 stats: Optional["StatsRecorder"] = None):
        self.engine = engine
        self.stats = stats
        self.workers = max(1, int(workers))
        self.q: "queue.Queue[Optional[GenRequest]]" = queue.Queue(maxsize=max(1, queue_size))
        self.running = 0
        self.served = 0
        self.failed = 0
        self._lock = threading.Lock()
        self._threads = [threading.Thread(target=self._loop, name=f"bigbrain-engine-{i}", daemon=True)
                         for i in range(self.workers)]
        for t in self._threads:
            t.start()

    @property
    def busy(self) -> bool:
        return self.running > 0

    def submit(self, req: GenRequest) -> None:
        self.q.put_nowait(req)  # queue.Full -> the caller answers 503

    def depth(self) -> int:
        return self.q.qsize()

    def alive(self) -> bool:
        return all(t.is_alive() for t in self._threads)

    def close(self) -> None:
        for _ in self._threads:
            try:
                self.q.put(None, timeout=1.0)
            except queue.Full:
                pass
        for t in self._threads:
            t.join(timeout=5.0)

    def _loop(self) -> None:
        while True:
            req = self.q.get()
            if req is None:
                return
            if req.cancel.is_set():
                req.t_start = time.monotonic()
                self._record(req, "cancelled", 0)
                continue  # the client left while queued
            with self._lock:
                self.running += 1
            try:
                self._run(req)
                with self._lock:
                    self.served += 1
            except Exception as exc:  # noqa: BLE001 - one request's failure, not the server's
                with self._lock:
                    self.failed += 1
                log.exception("request %d failed", req.rid)
                self._record(req, "error", len(req.tokens))
                req.events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                with self._lock:
                    self.running -= 1

    def _record(self, req: GenRequest, finish: str, completion_tokens: int) -> None:
        if self.stats is None:
            return
        try:
            self.stats.record(req, finish, completion_tokens)
        except Exception:  # noqa: BLE001 - a stats glitch must never fail a request
            log.exception("stats record failed for request %d", req.rid)

    def _run(self, req: GenRequest) -> None:
        req.t_start = time.monotonic()
        detok = IncrementalDetokenizer(self.engine.decode, req.prompt_ids[-6:])
        stopper = StopMatcher(req.stop)
        finish, stop_string = "length", False
        n = 0
        extra = {}
        if getattr(self.engine, "batch", None) is not None:
            extra = {"stop_ids": () if req.ignore_eos else tuple(self.engine.eos_ids),
                     "spec": req.spec, "spec_k": req.spec_k, "diag": req.diag, "info": req.info}
        gen = self.engine.generate(req.prompt_ids, req.settings, req.max_tokens,
                                   request_id=req.rid, return_logits=req.return_logits, **extra)
        with closing(gen):
            for tok, row in gen:
                n += 1
                if n == 1:
                    req.t_first = time.monotonic()
                req.tokens.append(int(tok))
                if row is not None:
                    req.logits.append(np.asarray(row, np.float32))
                if not req.ignore_eos and tok in self.engine.eos_ids:
                    finish = "stop"
                    break
                emit, hit = stopper.feed(detok.push(tok))
                if emit:
                    req.events.put(("delta", emit))
                if hit:
                    finish, stop_string = "stop", True
                    break
                if req.cancel.is_set():
                    finish = "cancelled"
                    break
        if not stop_string:
            emit, hit = stopper.feed(detok.flush())
            tail = emit + ("" if hit else stopper.flush())
            if hit:
                finish = "stop"
            if tail:
                req.events.put(("delta", tail))
        self._record(req, finish, n)       # before the client can see "done": a /health read after it counts it
        req.events.put(("done", {"finish_reason": finish, "prompt_tokens": len(req.prompt_ids),
                                 "completion_tokens": n}))


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class _HTTPError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error",
                 headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.status, self.message, self.kind, self.headers = status, message, kind, headers or {}


def _usage(done: Dict) -> Dict:
    p, c = done["prompt_tokens"], done["completion_tokens"]
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


class BigBrainApp:
    def __init__(self, engine, api_key: Optional[str], queue_size: int = 8, retry_after: int = 5,
                 default_max_tokens: int = 0, generation_defaults: Optional[Dict] = None,
                 keepalive_s: float = 5.0, max_body_bytes: int = MAX_BODY_BYTES):
        self.engine = engine
        self.api_key = api_key
        self.retry_after = int(retry_after)
        self.default_max_tokens = int(default_max_tokens or 0)
        self.generation_defaults = dict(generation_defaults or {})
        self.keepalive_s = keepalive_s
        self.max_body_bytes = max_body_bytes
        self.stats = StatsRecorder()
        self.scheduler = Scheduler(engine, queue_size, workers=getattr(engine, "concurrency", 1),
                                   stats=self.stats)
        self._rid = itertools.count(1)
        self._rid_lock = threading.Lock()
        self.created = int(time.time())

    def close(self) -> None:
        self.scheduler.close()

    # -------------------------------------------------------------- plumbing

    def _authorized(self, h) -> bool:
        if self.api_key is None:
            return True
        auth = h.headers.get("Authorization", "")
        token = auth[7:].strip() if auth[:7].lower() == "bearer " else h.headers.get("x-api-key", "")
        return secrets.compare_digest(token.encode("utf-8"), self.api_key.encode("utf-8"))

    @staticmethod
    def _send_json(h, status: int, obj, headers: Optional[Dict[str, str]] = None) -> None:
        data = json.dumps(obj).encode("utf-8")
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(data)

    def _error(self, h, e: _HTTPError) -> None:
        self._send_json(h, e.status, {"error": {"message": e.message, "type": e.kind,
                                                "code": e.status}}, e.headers)

    def _body(self, h) -> Dict:
        try:
            n = int(h.headers.get("Content-Length") or 0)
        except ValueError:
            raise _HTTPError(400, "bad Content-Length")
        if n <= 0:
            raise _HTTPError(400, "a JSON request body is required")
        if n > self.max_body_bytes:
            raise _HTTPError(413, f"request body over {self.max_body_bytes} bytes")
        try:
            body = json.loads(h.rfile.read(n))
        except ValueError as exc:
            raise _HTTPError(400, f"invalid JSON: {exc}")
        if not isinstance(body, dict):
            raise _HTTPError(400, "the request body must be a JSON object")
        return body

    @staticmethod
    def _client_gone(h) -> bool:
        try:
            readable, _, _ = select.select([h.connection], [], [], 0)
            if not readable:
                return False
            return h.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def route(self, h, method: str) -> None:
        path = urlsplit(h.path).path.rstrip("/") or "/"
        try:
            if method == "GET" and path == "/health":
                return self._health(h)
            if not self._authorized(h):
                raise _HTTPError(401, "invalid or missing API key", "authentication_error")
            if method == "GET" and path == "/v1/models":
                return self._send_json(h, 200, {"object": "list", "data": [
                    {"id": self.engine.model_name, "object": "model", "created": self.created,
                     "owned_by": "bigbrain"}]})
            if method == "POST" and path == "/v1/chat/completions":
                return self._chat(h)
            if method == "POST" and path == "/v1/completions":
                return self._completions(h)
            if method == "POST" and path == "/v1/bigbrain/generate":
                return self._debug_generate(h)
            if path == "/v1/bigbrain/metrics" and method in ("GET", "POST"):
                return self._metrics(h, method)
            if path == "/v1/bigbrain/stats" and method == "GET":
                return self._stats(h)
            raise _HTTPError(404, f"no route {method} {path}", "not_found_error")
        except _HTTPError as e:
            self._error(h, e)
        except _CLIENT_GONE:
            pass
        except Exception as exc:  # noqa: BLE001
            log.exception("unhandled error on %s %s", method, path)
            try:
                self._error(h, _HTTPError(500, f"{type(exc).__name__}: {exc}", "server_error"))
            except Exception:  # noqa: BLE001
                pass

    def _metrics(self, h, method: str) -> None:
        """GET: the engine's and every node's numbers for the current window. POST
        {"reset": true}: the same, then the window restarts."""
        reset = False
        if method == "POST":
            body = self._body(h) if int(h.headers.get("Content-Length") or 0) > 0 else {}
            reset = bool(body.get("reset"))
        rep = self.engine.report(reset=reset) if hasattr(self.engine, "report") else {}
        rep["scheduler"] = {"running": self.scheduler.running, "queued": self.scheduler.depth(),
                            "served": self.scheduler.served, "failed": self.scheduler.failed,
                            "workers": self.scheduler.workers}
        self._send_json(h, 200, rep)

    def _stats_snapshot(self, recent: int = 0) -> Dict:
        sch = self.scheduler
        return self.stats.snapshot(served=sch.served, failed=sch.failed, running=sch.running,
                                   queued=sch.depth(), recent=recent)

    def _stats(self, h) -> None:
        """GET /v1/bigbrain/stats[?recent=N]: the per-request record (see StatsRecorder)."""
        q = parse_qs(urlsplit(h.path).query)
        try:
            recent = int((q.get("recent") or [StatsRecorder.RECENT])[0])
        except ValueError:
            raise _HTTPError(400, "recent must be a whole number")
        self._send_json(h, 200, self._stats_snapshot(recent=max(0, min(recent, StatsRecorder.RECENT))))

    def _batch_stats(self) -> Dict:
        b = getattr(self.engine, "batch", None)
        if b is None:
            return {}
        try:
            st = b.stats()
            return {"tokens_per_s": st.get("decode_tok_s"), "slots": st.get("slots"),
                    "active": st.get("active")}
        except Exception:  # noqa: BLE001 - a health probe must not fail on a metrics glitch
            return {}

    def _health(self, h) -> None:
        eng = self.engine.health()
        ok = bool(eng.get("ok")) and self.scheduler.alive()
        self._send_json(h, 200 if ok else 503, {
            "status": "ok" if ok else "degraded", "model": self.engine.model_name,
            "busy": self.scheduler.busy, "running": self.scheduler.running,
            "slots": getattr(self.engine, "concurrency", 1), "queue_depth": self.scheduler.depth(),
            "queue_size": self.scheduler.q.maxsize, "served": self.scheduler.served,
            "requests_served": self.scheduler.served, **self._batch_stats(),
            "failed": self.scheduler.failed, "max_context": self.engine.max_context,
            "chat_template": getattr(getattr(self.engine, "template", None), "source", None),
            "stats": self._stats_snapshot(), "engine": eng})

    # -------------------------------------------------------- request parsing

    def _sampling(self, body: Dict, greedy_default: bool = False) -> SamplingSettings:
        d = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
        d.update({"temperature": 0.0} if greedy_default else self.generation_defaults)
        for k in ("temperature", "top_p", "top_k", "min_p", "seed"):
            if body.get(k) is not None:
                d[k] = body[k]
        if isinstance(d.get("top_k"), (int, float)) and not isinstance(d["top_k"], bool) \
                and d["top_k"] < 0:
            d["top_k"] = 0  # -1 = off
        try:
            return SamplingSettings.from_dict(d)
        except ValueError as exc:
            raise _HTTPError(400, str(exc))

    @staticmethod
    def _common_checks(body: Dict) -> None:
        if body.get("n") not in (None, 1):
            raise _HTTPError(400, "n must be 1 (one completion per request)")
        if body.get("logprobs") or body.get("top_logprobs"):
            raise _HTTPError(400, "logprobs are not supported")
        rf = body.get("response_format")
        if rf is not None and (not isinstance(rf, dict) or rf.get("type") not in (None, "text")):
            raise _HTTPError(400, "response_format other than text is not supported")

    @staticmethod
    def _max_tokens(body: Dict) -> Optional[int]:
        v = body.get("max_completion_tokens", body.get("max_tokens"))
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise _HTTPError(400, "max_tokens must be a positive integer")
        return v

    @staticmethod
    def _stops(body: Dict) -> List[str]:
        s = body.get("stop")
        if s is None:
            return []
        if isinstance(s, str):
            s = [s]
        if not isinstance(s, list) or not all(isinstance(x, str) for x in s):
            raise _HTTPError(400, "stop must be a string or a list of strings")
        s = [x for x in s if x]
        if len(s) > MAX_STOP_STRINGS:
            raise _HTTPError(400, f"at most {MAX_STOP_STRINGS} stop strings")
        return s

    def _budget(self, prompt_ids: List[int], requested: Optional[int]) -> int:
        S, ctx = len(prompt_ids), self.engine.max_context
        if S == 0:
            raise _HTTPError(400, "empty prompt")
        if S >= ctx:
            raise _HTTPError(400, f"prompt is {S} tokens; the context is {ctx} tokens")
        room = ctx - S
        return min(requested or self.default_max_tokens or room, room)

    @staticmethod
    def _spec(body: Dict) -> Tuple[Optional[str], Optional[int]]:
        """The request's speculation overrides: ("none"|"ngram"|"draft" | None, k | None)."""
        spec, k = body.get("spec"), body.get("spec_k")
        if spec is not None and spec not in ("none", "ngram", "draft"):
            raise _HTTPError(400, "spec must be one of none, ngram, draft")
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 15):
            raise _HTTPError(400, "spec_k must be an integer from 1 to 15")
        return spec, k

    @staticmethod
    def _job_tag(h) -> Optional[Dict]:
        """The job this request names itself with (X-BigBrain-Job[-Progress] headers)."""
        return parse_job_tag(h.headers.get("X-BigBrain-Job"), h.headers.get("X-BigBrain-Job-Progress"))

    def _submit(self, prompt_ids, settings, max_tokens, stop=(), ignore_eos=False,
                return_logits=False, body: Optional[Dict] = None, diag: bool = False,
                route: str = "", job: Optional[Dict] = None) -> GenRequest:
        spec, spec_k = self._spec(body or {})
        batch = getattr(self.engine, "batch", None)
        if batch is not None:
            if return_logits:
                raise _HTTPError(400, "return_logits is not available in batched mode (start the "
                                      "pipeline with --slots 1 and no speculation)")
            if batch.dead:
                raise _HTTPError(503, f"the pipeline is down: {batch.dead}", "server_error")
            if not batch.ready.is_set():
                raise _HTTPError(503, "the pipeline is still starting", "server_busy",
                                 headers={"Retry-After": str(self.retry_after)})
            if spec == "draft" and "draft" not in batch.drafters:
                raise _HTTPError(400, "spec 'draft' needs a draft model (--spec-draft model --spec-model DIR)")
        elif spec not in (None, "none"):
            raise _HTTPError(400, "speculation needs a batched pipeline (--spec-draft ngram|model)")
        with self._rid_lock:
            rid = next(self._rid)
        req = GenRequest(rid=rid, prompt_ids=list(prompt_ids), settings=settings,
                         max_tokens=max_tokens, stop=list(stop), ignore_eos=ignore_eos,
                         return_logits=return_logits, spec=spec, spec_k=spec_k, diag=diag,
                         route=route, job=job)
        try:
            self.scheduler.submit(req)
            self.stats.note_job(job)
        except queue.Full:
            raise _HTTPError(503, f"busy: {self.scheduler.depth()} request(s) already queued; "
                                  f"retry later", "server_busy",
                             headers={"Retry-After": str(self.retry_after)})
        return req

    def _collect(self, h, req: GenRequest) -> Optional[Tuple[str, Dict]]:
        parts = []
        while True:
            try:
                ev = req.events.get(timeout=1.0)
            except queue.Empty:
                if self._client_gone(h):
                    req.cancel.set()
                    return None
                continue
            if ev[0] == "delta":
                parts.append(ev[1])
            elif ev[0] == "done":
                return "".join(parts), ev[1]
            else:
                raise _HTTPError(500, ev[1], "server_error")

    def _stream(self, h, req: GenRequest, first: Optional[Dict], chunk: Callable[..., Dict],
                usage_chunk: Optional[Callable[[Dict], Dict]]) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.send_header("X-Accel-Buffering", "no")
        h.end_headers()

        def send(obj):
            h.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
            h.wfile.flush()

        try:
            if first is not None:
                send(first)
            while True:
                try:
                    ev = req.events.get(timeout=self.keepalive_s)
                except queue.Empty:
                    h.wfile.write(b": keep-alive\n\n")  # also detects a vanished client
                    h.wfile.flush()
                    continue
                if ev[0] == "delta":
                    send(chunk(text=ev[1]))
                elif ev[0] == "done":
                    send(chunk(finish=ev[1]["finish_reason"]))
                    if usage_chunk is not None:
                        send(usage_chunk(ev[1]))
                    break
                else:
                    send({"error": {"message": ev[1], "type": "server_error"}})
                    break
            h.wfile.write(b"data: [DONE]\n\n")
            h.wfile.flush()
        except _CLIENT_GONE + (OSError,):
            req.cancel.set()

    # ----------------------------------------------------------------- routes

    def _chat(self, h) -> None:
        body = self._body(h)
        self._common_checks(body)
        try:
            msgs = normalize_messages(body.get("messages"))
            kw = body.get("chat_template_kwargs") or {}
            if not isinstance(kw, dict):
                raise ValueError("chat_template_kwargs must be an object")
            text = self.engine.template.render(
                msgs, add_generation_prompt=body.get("add_generation_prompt", True) is not False,
                tools=body.get("tools"), **{str(k): v for k, v in kw.items()})
        except ValueError as exc:
            raise _HTTPError(400, str(exc))
        settings = self._sampling(body)
        stops = self._stops(body)
        requested = self._max_tokens(body)
        prompt_ids = self.engine.encode(text, special=True)
        max_tokens = self._budget(prompt_ids, requested)
        req = self._submit(prompt_ids, settings, max_tokens, stops, body=body, route="chat",
                           job=self._job_tag(h))
        cid, created, model = f"chatcmpl-{uuid.uuid4().hex}", int(time.time()), self.engine.model_name

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            def chunk(text=None, finish=None):
                delta = {"content": text} if text is not None else {}
                return {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model, "choices": [{"index": 0, "delta": delta,
                                                     "logprobs": None, "finish_reason": finish}]}

            first = chunk()
            first["choices"][0]["delta"] = {"role": "assistant", "content": ""}
            usage = (lambda done: {"id": cid, "object": "chat.completion.chunk",
                                   "created": created, "model": model, "choices": [],
                                   "usage": _usage(done)}) if include_usage else None
            return self._stream(h, req, first, chunk, usage)
        got = self._collect(h, req)
        if got is None:
            return
        text, done = got
        self._send_json(h, 200, {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "logprobs": None, "finish_reason": done["finish_reason"]}],
            "usage": _usage(done)})

    def _prompt_ids(self, body: Dict, key: str = "prompt") -> List[int]:
        p = body.get(key)
        if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
            p = p[0]
        if isinstance(p, str):
            return self.engine.encode(p, special=True)
        if isinstance(p, list) and p and all(isinstance(t, int) and not isinstance(t, bool)
                                             and t >= 0 for t in p):
            return list(p)
        raise _HTTPError(400, f"{key} must be a string or a list of token ids")

    def _completions(self, h) -> None:
        body = self._body(h)
        self._common_checks(body)
        settings = self._sampling(body)
        stops = self._stops(body)
        requested = self._max_tokens(body)
        prompt_ids = self._prompt_ids(body)
        req = self._submit(prompt_ids, settings, self._budget(prompt_ids, requested), stops, body=body,
                           route="completions", job=self._job_tag(h))
        cid, created, model = f"cmpl-{uuid.uuid4().hex}", int(time.time()), self.engine.model_name

        def chunk(text=None, finish=None):
            return {"id": cid, "object": "text_completion", "created": created, "model": model,
                    "choices": [{"index": 0, "text": text or "", "logprobs": None,
                                 "finish_reason": finish}]}

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            usage = (lambda done: {"id": cid, "object": "text_completion", "created": created,
                                   "model": model, "choices": [], "usage": _usage(done)}) \
                if include_usage else None
            return self._stream(h, req, None, chunk, usage)
        got = self._collect(h, req)
        if got is None:
            return
        text, done = got
        out = chunk(text=text, finish=done["finish_reason"])
        out["usage"] = _usage(done)
        self._send_json(h, 200, out)

    def _debug_generate(self, h) -> None:
        body = self._body(h)
        settings = self._sampling(body, greedy_default=True)
        key = "prompt_ids" if "prompt_ids" in body else "prompt"
        prompt_ids = self._prompt_ids(body, key)
        requested = self._max_tokens(body)
        return_logits = bool(body.get("return_logits"))
        max_tokens = self._budget(prompt_ids, requested)
        if return_logits and max_tokens > MAX_LOGIT_TOKENS:
            raise _HTTPError(400, f"return_logits caps max_tokens at {MAX_LOGIT_TOKENS}")
        req = self._submit(prompt_ids, settings, max_tokens, self._stops(body),
                           ignore_eos=bool(body.get("ignore_eos")), return_logits=return_logits,
                           body=body, diag=bool(body.get("diag")), route="generate")
        got = self._collect(h, req)
        if got is None:
            return
        text, done = got
        out = {"id": f"bbgen-{uuid.uuid4().hex}", "object": "bigbrain.generation",
               "model": self.engine.model_name, "prompt_ids": prompt_ids, "tokens": req.tokens,
               "text": text, "finish_reason": done["finish_reason"], "usage": _usage(done),
               "settings": settings.to_dict()}
        if req.info:
            # batched mode: what speculation did for this request, its timing and (diag) the
            # top-1 minus top-2 logit margin of every token
            out["info"] = req.info
        if return_logits:
            arr = np.stack(req.logits).astype("<f4") if req.logits else np.zeros((0, 0), "<f4")
            out["logits"] = {"dtype": "<f4", "shape": list(arr.shape),
                             "b64": base64.b64encode(arr.tobytes()).decode("ascii")}
        self._send_json(h, 200, out)


class _Handler(BaseHTTPRequestHandler):
    server_version = "BigBrain/2"
    protocol_version = "HTTP/1.0"

    def do_GET(self):  # noqa: N802
        self.server.app.route(self, "GET")

    def do_POST(self):  # noqa: N802
        self.server.app.route(self, "POST")

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)


def make_http_server(app: BigBrainApp, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    httpd.app = app
    return httpd


def serve_in_thread(app: BigBrainApp, host: str = "127.0.0.1", port: int = 0):
    """(httpd, bound_port, thread) -- tests and embedding."""
    httpd = make_http_server(app, host, port)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    return httpd, httpd.server_address[1], th


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def make_batch_engine(w, args) -> Optional[PipelineEngine]:
    """The batched engine for node 0, or None for the single-sequence path (one slot, no
    speculation). Loads the draft model when --spec-draft model."""
    if not w.batched:
        return None
    mode = {"none": "none", "ngram": "ngram", "model": "draft"}[args.spec_draft]
    spec = SpecConfig(mode=mode, k=args.spec_k, ngram_min=args.ngram_min, ngram_max=args.ngram_max)
    drafters = {}
    if args.spec_draft == "model" or args.spec_model:
        if not args.spec_model:
            raise SystemExit("--spec-draft model needs --spec-model DIR (the draft model folder)")
        from .draft import load_draft_model
        from .spec import DraftModelDrafter
        backend = load_draft_model(args.spec_model, args.model_dir, args.draft_device or args.device,
                                   w.n_slots, w.max_num_tokens, mem_fraction=args.mem_fraction,
                                   max_draft=w.max_draft)
        drafters["draft"] = DraftModelDrafter(backend, prefill_chunk=args.prefill_chunk)
    cfg = EngineConfig(spec=spec, inflight=args.inflight, mb_size=args.mb_size,
                       prefill_chunk=args.prefill_chunk, prefill_window=args.prefill_window,
                       hello_timeout_s=(args.hello_timeout or None))
    return PipelineEngine(w, cfg, drafters)


def _start_batch_engine(batch: PipelineEngine) -> None:
    try:
        batch.start()
    except EngineDead as exc:
        log.error("the batched pipeline could not start: %s", exc)
        batch.dead = str(exc)
    except Exception:  # noqa: BLE001
        log.exception("the batched pipeline could not start")
        batch.dead = "the engine crashed while starting"


def build_arg_parser() -> argparse.ArgumentParser:
    a = add_node_args(argparse.ArgumentParser(
        description="BigBrain node 0 with an OpenAI-compatible HTTP endpoint"))
    a.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    a.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    a.add_argument("--model-name", default=None,
                   help="the id /v1/models reports (default: the model folder's name)")
    a.add_argument("--api-key-file", default=None, help="file holding the bearer token")
    a.add_argument("--api-token", default=None,
                   help="the bearer token itself (what the cluster controller passes; visible in ps -- "
                        "prefer --api-key-file when you launch by hand)")
    a.add_argument("--api-key-env", default="BIGBRAIN_API_KEY",
                   help="env var holding the bearer token (when no --api-key-file)")
    a.add_argument("--insecure-no-auth", action="store_true",
                   help="serve without auth (loopback tests only)")
    a.add_argument("--queue-size", type=int, default=8,
                   help="requests that may WAIT behind the running one; more -> 503")
    a.add_argument("--retry-after", type=int, default=5, help="Retry-After seconds on a 503")
    a.add_argument("--default-max-tokens", type=int, default=0,
                   help="max_tokens when a request gives none (0 = up to the context end)")
    a.add_argument("--inflight", type=int, default=0,
                   help="batched mode: decode steps in flight (0 = the number of nodes)")
    a.add_argument("--mb-size", type=int, default=0,
                   help="batched mode: most sequences per decode step (0 = spread the active "
                        "sequences over the in-flight window)")
    a.add_argument("--prefill-window", type=int, default=2,
                   help="batched mode: prompt chunks in flight")
    a.add_argument("--hello-timeout", type=float, default=0.0,
                   help="batched mode: give up the nodes' handshake after this many seconds "
                        "(0 = wait for as long as they take to load)")
    a.add_argument("--token-timeout", type=float, default=600.0,
                   help="seconds to wait for one token from the last node (prefill included)")
    # The controller passes these to every node; the server ignores them.
    a.add_argument("--prompt", default=None, help=argparse.SUPPRESS)
    a.add_argument("--max-new", type=int, default=None, help=argparse.SUPPRESS)
    return a


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s bigbrain.server[%(process)d] %(levelname)s %(message)s")
    if args.rank != 0:
        raise SystemExit("server.py runs on node 0 (--rank 0); other nodes run tools.bigbrain.worker")
    api_key = resolve_api_key(args.api_key_file, args.api_key_env, args.insecure_no_auth,
                              token=args.api_token)
    if api_key is None:
        log.warning("AUTH DISABLED (--insecure-no-auth): anyone who can reach the port can use it")
    w = worker_from_args(args)
    app = httpd = batch = None
    try:
        w.start_recv()
        batch = make_batch_engine(w, args)
        engine = WorkerEngine(w, args.model_dir, args.model_name, args.token_timeout, batch=batch)
        if batch is not None:
            batch.eos_ids = set(engine.eos_ids)
            # the handshake waits for every node to finish loading: do it off the main thread so
            # /health (503 "starting") and the HTTP port are up meanwhile
            threading.Thread(target=_start_batch_engine, args=(batch,), name="bigbrain-handshake",
                             daemon=True).start()
        app = BigBrainApp(engine, api_key=api_key, queue_size=args.queue_size,
                          retry_after=args.retry_after, default_max_tokens=args.default_max_tokens,
                          generation_defaults=load_generation_defaults(args.model_dir))
        httpd = make_http_server(app, args.host, args.http_port)
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(
            target=httpd.shutdown, daemon=True).start())
        log.info("serving %s on http://%s:%d/v1 (%d node(s), context %d, template %s, "
                 "queue %d)", engine.model_name, args.host, args.http_port, args.total,
                 engine.max_context, engine.template.source, args.queue_size)
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if httpd is not None:
            httpd.server_close()
        if batch is not None:
            batch.close()           # first: it fails the sequences the scheduler threads wait on
        if app is not None:
            app.close()
        try:
            w.send_terminator()  # nodes 1..T-1 exit cleanly
        except Exception:  # noqa: BLE001
            pass
        w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
