# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/bench.py -- measure a running BigBrain server: the speed grid and the parity proof.

Stdlib only; runs anywhere that can reach node 0's HTTP port (directly, or through an ssh -L).
It talks to `/v1/bigbrain/generate` (greedy, ignore_eos, fixed length) and
`/v1/bigbrain/metrics`.

    grid     for each (concurrency, speculation) cell: send every prompt once, `concurrency` at
             a time, greedy; record aggregate and per-sequence decode tok/s, prefill tok/s,
             first-token latency, the speculation acceptance rate, per-node busy / hop-wait /
             send %, VRAM per node -- and compare every sequence's tokens with the PLAIN run
             (concurrency 1, no speculation: the first cell, or --baseline FILE from a
             single-sequence server). A divergence reports where it happened and the top-1
             minus top-2 logit margin there (a near-tie is numerics, not a bug).
    compare  two result files -> the same parity verdict.

    # the speed grid: slots {1,2,4,8} x speculation {none, ngram, draft}
    python -m bigbrain.bench grid --url http://127.0.0.1:16980 --key-file api_key \\
        --concurrency 1 2 4 8 --spec none ngram draft --max-tokens 256 --prompts-file prompts.json \\
        --out grid.json

A cell the server cannot run (no draft model loaded, more concurrency than slots is fine -- the
rest queue) is recorded as skipped with the reason.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_REVIEW_PROMPT = ("Review the following source file. List every real bug you find, each with the "
                         "line it is on and why it is wrong.\n\n")


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

class Client:
    def __init__(self, base: str, key: str = "", timeout: float = 3600.0):
        self.base, self.key, self.timeout = base.rstrip("/"), key, timeout

    def call(self, method: str, path: str, body: Optional[dict] = None) -> Tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path, method=method, data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}

    def post(self, path, body):
        return self.call("POST", path, body)

    def get(self, path):
        return self.call("GET", path)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------

def synthetic_prompts(n: int, chars: int) -> List[str]:
    """Deterministic code-like prompts (repetitive enough for prompt lookup to find matches)."""
    out = []
    for i in range(n):
        lines, size, j = [], 0, 0
        while size < chars:
            ln = f"def handler_{i}_{j % 7}(request, ctx):\n    return ctx.lookup('{i}-{j % 5}') + len(request)\n"
            lines.append(ln)
            size += len(ln)
            j += 1
        out.append(DEFAULT_REVIEW_PROMPT + "".join(lines)[:chars])
    return out


def review_prompts(root: str, patterns: Sequence[str], chars: int, limit: int) -> List[str]:
    """One 'review this file' prompt per source file under `root` matching the globs."""
    files: List[str] = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(root, pat), recursive=True))
    out = []
    for f in dict.fromkeys(files):
        if os.path.isfile(f):
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    text = fh.read(chars)
            except OSError:
                continue
            out.append(DEFAULT_REVIEW_PROMPT + f"File: {os.path.relpath(f, root)}\n\n" + text)
        if len(out) >= limit:
            break
    return out


def load_prompts(a) -> List[str]:
    if a.prompts_file:
        with open(a.prompts_file, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise SystemExit("--prompts-file must be a JSON list of strings")
        return data[:a.limit]
    if a.review_dir:
        ps = review_prompts(a.review_dir, a.glob, a.prompt_chars, a.limit)
        if not ps:
            raise SystemExit(f"no files under {a.review_dir} match {a.glob}")
        return ps
    return synthetic_prompts(a.limit, a.prompt_chars)


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def _pct(xs: Sequence[float], p: float) -> Optional[float]:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


def run_cell(client: Client, prompts: Sequence[str], concurrency: int, spec: str, spec_k: int,
             max_tokens: int, diag: bool = False, log: Callable[[str], None] = lambda s: None) -> dict:
    """Send every prompt once, `concurrency` at a time. Returns the cell record."""
    cell = {"concurrency": concurrency, "spec": spec, "spec_k": spec_k if spec != "none" else 0,
            "max_tokens": max_tokens, "requests": len(prompts)}
    client.post("/v1/bigbrain/metrics", {"reset": True})
    results: List[Optional[dict]] = [None] * len(prompts)
    errors: List[Tuple[int, int, str]] = []
    nxt = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= len(prompts):
                return
            # always say which speculation: the server's own default must not leak into a cell
            body = {"prompt": prompts[i], "max_tokens": max_tokens, "ignore_eos": True, "diag": diag,
                    "spec": spec}
            if spec != "none":
                body["spec_k"] = spec_k
            t0 = time.monotonic()
            code, obj = client.post("/v1/bigbrain/generate", body)
            wall = time.monotonic() - t0
            if code != 200:
                with lock:
                    errors.append((i, code, (obj.get("error") or {}).get("message", str(obj))[:300]))
                continue
            results[i] = {"i": i, "tokens": obj["tokens"], "prompt_tokens": len(obj.get("prompt_ids") or []),
                          "wall_s": wall, "info": obj.get("info") or {}}
            log(f"  [{concurrency}x {spec}] prompt {i}: {len(obj['tokens'])} tokens in {wall:.1f}s")

    t_start = time.monotonic()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t_start
    if errors and not any(results):
        cell["skipped"] = errors[0][2]
        cell["http_status"] = errors[0][1]
        return cell
    done = [r for r in results if r]
    toks = sum(len(r["tokens"]) for r in done)
    cell.update({
        "completed": len(done), "failed": len(errors), "errors": errors[:5],
        "tokens": toks, "wall_s": round(wall, 3),
        "agg_tok_s": round(toks / wall, 3) if wall > 0 else None,
        "per_seq_tok_s": (round(sum(r["info"].get("decode_tok_s") or 0 for r in done) / len(done), 3)
                          if done else None),
        "request_wall_s": {"mean": round(sum(r["wall_s"] for r in done) / len(done), 3) if done else None,
                           "max": round(max((r["wall_s"] for r in done), default=0), 3)},
        "outputs": {str(r["i"]): r["tokens"] for r in done},
        "margins": {str(r["i"]): r["info"].get("margins") for r in done if r["info"].get("margins")},
    })
    first = [r["info"]["first_token_s"] for r in done if r["info"].get("first_token_s") is not None]
    cell["first_token_s"] = {"mean": round(sum(first) / len(first), 4) if first else None,
                             "p50": _pct(first, 0.5), "max": max(first) if first else None}
    # per-request speculation
    prop = sum(r["info"].get("proposed", 0) for r in done)
    acc = sum(r["info"].get("accepted", 0) for r in done)
    cell["spec_stats"] = {"proposed": prop, "accepted": acc,
                          "acceptance_rate": round(acc / prop, 4) if prop else None}
    code, m = client.get("/v1/bigbrain/metrics")
    if code == 200:
        eng = m.get("engine") or {}
        cell["engine"] = {k: eng.get(k) for k in (
            "decode_tok_s", "prefill_tok_s", "prefill_tokens", "decode_tokens", "per_sequence_tok_s",
            "first_token_s", "steps", "step_latency_s", "inflight", "rows_per_decode_step",
            "driver_idle_pct", "active_s")}
        cell["engine"]["spec"] = eng.get("spec")
        cell["nodes"] = [{
            "rank": n.get("rank"), "busy_pct": n.get("busy_pct"), "hop_wait_pct": n.get("wait_pct"),
            "hop_wait_s": n.get("wait_s"), "send_pct": n.get("send_pct"), "steps": n.get("steps"),
            "rollbacks": n.get("rollbacks"), "rolled_tokens": n.get("rolled_tokens"),
            "vram_allocated_mib": (n.get("vram") or {}).get("allocated_mib"),
            "vram_peak_mib": (n.get("vram") or {}).get("peak_allocated_mib"),
            "vram_reserved_mib": (n.get("vram") or {}).get("reserved_mib")} for n in m.get("nodes") or []]
    return cell


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------

def compare_outputs(base: Dict[str, List[int]], other: Dict[str, List[int]],
                    base_margins: Optional[Dict[str, List[float]]] = None) -> dict:
    """Per prompt: identical, or the first divergent position with the plain run's top-1/top-2
    margin there. Prompts missing from either side are not compared."""
    rows, same = [], 0
    keys = [k for k in base if k in other]
    for k in sorted(keys, key=lambda x: int(x)):
        a, b = base[k], other[k]
        n = min(len(a), len(b))
        div = next((i for i in range(n) if a[i] != b[i]), None)
        if div is None and len(a) != len(b):
            div = n
        if div is None:
            same += 1
            continue
        m = (base_margins or {}).get(k)
        rows.append({"prompt": int(k), "first_divergence": div,
                     "plain_token": a[div] if div < len(a) else None,
                     "other_token": b[div] if div < len(b) else None,
                     "plain_margin": (m[div] if m and div < len(m) else None)})
    return {"compared": len(keys), "identical": same, "all_identical": bool(keys) and same == len(keys),
            "diverged": rows}


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------

def run_grid(client: Client, prompts: Sequence[str], concurrencies: Sequence[int], specs: Sequence[str],
             spec_k: int, max_tokens: int, baseline: Optional[dict] = None,
             log: Callable[[str], None] = lambda s: None) -> dict:
    cells: List[dict] = []
    plain: Optional[dict] = None
    base_out = (baseline or {}).get("outputs")
    base_margins = (baseline or {}).get("margins")
    order = [(c, s) for s in specs for c in concurrencies]
    # the plain cell first: it is the reference for every other cell
    if (1, "none") not in order:
        order.insert(0, (1, "none"))
    else:
        order.remove((1, "none"))
        order.insert(0, (1, "none"))
    for c, s in order:
        log(f"cell: concurrency {c}, speculation {s}")
        cell = run_cell(client, prompts, c, s, spec_k, max_tokens, diag=(c == 1 and s == "none"), log=log)
        if (c, s) == (1, "none") and "outputs" in cell:
            plain = cell
            if base_out is None:
                base_out, base_margins = cell["outputs"], cell.get("margins")
        if "outputs" in cell and base_out is not None:
            cell["parity"] = compare_outputs(base_out, cell["outputs"], base_margins)
        cells.append(cell)
    return {"cells": cells, "baseline": "file" if baseline else "this run's concurrency-1 / no-speculation cell"}


def format_table(grid: dict) -> str:
    head = ("conc spec   agg tok/s  /seq tok/s  1st-tok s  prefill tok/s  accept  tok/row  "
            "node busy% (hop-wait%)            parity")
    lines = [head, "-" * len(head)]
    for c in grid["cells"]:
        if "skipped" in c:
            lines.append(f"{c['concurrency']:>4} {c['spec']:<6} skipped: {c['skipped']}")
            continue
        eng, ss = c.get("engine") or {}, c.get("spec_stats") or {}
        nodes = " ".join(f"{n['busy_pct']:.0f}({n['hop_wait_pct']:.0f})" for n in c.get("nodes") or []
                         if n.get("busy_pct") is not None)
        par = c.get("parity")
        verdict = "-" if not par else ("IDENTICAL" if par["all_identical"]
                                       else f"{par['identical']}/{par['compared']} identical")
        spec = (eng.get("spec") or {})
        lines.append(f"{c['concurrency']:>4} {c['spec']:<6} {c['agg_tok_s'] or 0:>9.2f} {c['per_seq_tok_s'] or 0:>11.2f} "
                     f"{(c['first_token_s']['mean'] or 0):>10.2f} {eng.get('prefill_tok_s') or 0:>14.1f} "
                     f"{('%.2f' % ss['acceptance_rate']) if ss.get('acceptance_rate') is not None else '  -  ':>7} "
                     f"{('%.2f' % spec['tokens_per_row']) if spec.get('tokens_per_row') else '  -  ':>8}  "
                     f"{nodes:<34} {verdict}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _client(a) -> Client:
    key = a.key
    if a.key_file:
        with open(a.key_file, encoding="utf-8") as f:
            key = f.read().strip()
    return Client(a.url, key or os.environ.get("BIGBRAIN_API_KEY", ""))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Measure a running BigBrain server")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid", help="the concurrency x speculation grid, with parity")
    g.add_argument("--url", default="http://127.0.0.1:16980")
    g.add_argument("--key", default="")
    g.add_argument("--key-file", default=None)
    g.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    g.add_argument("--spec", nargs="+", default=["none", "ngram", "draft"], choices=("none", "ngram", "draft"))
    g.add_argument("--spec-k", type=int, default=4)
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--prompts-file", default=None, help="JSON list of prompt strings")
    g.add_argument("--review-dir", default=None, help="one 'review this file' prompt per source file here")
    g.add_argument("--glob", nargs="+", default=["**/*.py"])
    g.add_argument("--prompt-chars", type=int, default=6000)
    g.add_argument("--limit", type=int, default=8, help="how many prompts (each is sent once per cell)")
    g.add_argument("--baseline", default=None,
                   help="a result file from the PLAIN single-sequence server: parity is checked against it")
    g.add_argument("--out", default=None)
    c = sub.add_parser("compare", help="parity between two result files' concurrency-1 / no-speculation outputs")
    c.add_argument("plain")
    c.add_argument("other")
    a = p.parse_args(argv)

    if a.cmd == "compare":
        def load(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            cells = d["cells"]
            pick = next((x for x in cells if x.get("concurrency") == 1 and x.get("spec") == "none"), cells[0])
            return pick
        pa, pb = load(a.plain), load(a.other)
        res = compare_outputs(pa["outputs"], pb["outputs"], pa.get("margins"))
        print(json.dumps(res, indent=1))
        return 0 if res["all_identical"] else 1

    client = _client(a)
    code, h = client.get("/health")
    if code != 200:
        print(f"server not healthy: {code} {h}", file=sys.stderr)
        return 2
    prompts = load_prompts(a)
    baseline = None
    if a.baseline:
        with open(a.baseline, encoding="utf-8") as f:
            d = json.load(f)
        baseline = next((x for x in d["cells"] if x.get("concurrency") == 1 and x.get("spec") == "none"), None)
    grid = run_grid(client, prompts, a.concurrency, a.spec, a.spec_k, a.max_tokens, baseline,
                    log=lambda s: print(s, file=sys.stderr, flush=True))
    grid["meta"] = {"url": a.url, "health": h, "prompts": len(prompts), "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "args": {k: v for k, v in vars(a).items() if k not in ("key",)}}
    print(format_table(grid))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(grid, f, indent=1)
        print(f"wrote {a.out}")
    bad = [c for c in grid["cells"] if c.get("parity") and not c["parity"]["all_identical"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
