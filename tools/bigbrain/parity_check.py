# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""tools/bigbrain/parity_check.py -- does the pipeline compute what the
monolithic model computes? Greedy, N tokens: token-for-token match plus the max
absolute logit difference, for dense AND hybrid (recurrent) models.

Subcommands (run with the venv python that has exllamav3 from the directory that holds
tools/bigbrain/, e.g. the controller's stage root /tmp/bigbrain):

  reference  Load the WHOLE model on one GPU and generate N greedy tokens with
             exllamav3's own API -- Model.prefill() over prompt[:-1] (chunked
             like the pipeline) then one Model.forward() per token, with ONE
             recurrent-state list for the whole run (exactly how exllamav3's
             generator drives a job). Writes a dump (.npz: prompt ids, tokens,
             every step's logits over the real vocab).
  run        On THIS machine: the reference (unless --reference FILE is
             given), then the pipeline as separate worker processes over
             127.0.0.1 (--node DEVICE:START:END per node, or --devices with an
             even split), node 0 writing its dump (--dump; the logits come back
             from the last node), then compare. Processes run one phase at a
             time, so the reference and the pipeline never share VRAM.
  compare    Two dumps -> verdict (exit 0 PASS, 1 FAIL).
  http       A running tools/bigbrain/server.py (any topology, cross-machine):
             POST the reference dump's prompt ids to /v1/bigbrain/generate
             (greedy, ignore_eos, return_logits), then compare.

Comparison rules: prompts must be identical; tokens are compared position by
position; logits are compared on every step up to AND including the first
divergence (step k's logits depend only on tokens < k, so those steps saw the
same history); a divergence reports both rows' top-2 so a near-tie (tiny
margin) is told apart from a real bug. PASS = every token identical and
max |diff| <= --atol (default 1e-2; the 2026-10-12 dense acceptance measured
0.0). allclose(atol=1e-3, rtol=1e-3) is reported alongside.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .worker import encode_prompt, make_cache, read_dump, write_dump

DEFAULT_PROMPT = (
    "Write a short, factual paragraph about how a transistor works, then list three "
    "everyday devices that depend on transistors and explain in one sentence each why.")
PKG_ROOT = Path(__file__).resolve().parent.parent.parent  # the dir holding tools/


def _torch():
    import torch
    return torch


# --------------------------------------------------------------------------
# reference: the monolithic model through exllamav3's own API
# --------------------------------------------------------------------------

def reference_run(model, cache, prompt_ids: Sequence[int], n: int, *, max_num_tokens: int,
                  prefill_chunk: int = 2048, vocab_size: Optional[int] = None
                  ) -> Tuple[List[int], np.ndarray]:
    """Greedy N tokens through Model.prefill / Model.forward (model.py:368-441)
    with one persistent recurrent-state list (generator.py:1043-1048 +
    job.py:1540). Returns (tokens, logits rows (N, vocab) fp32)."""
    torch = _torch()
    prompt = [int(t) for t in prompt_ids]
    S = len(prompt)
    if S == 0 or S + n > max_num_tokens:
        raise ValueError(f"prompt {S} + n {n} must fit max_num_tokens {max_num_tokens}")
    recurrent = bool(getattr(model, "caps", {}).get("recurrent_states"))
    states = None

    def params(past: int) -> dict:
        p = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, max_num_tokens),
             "past_len": past}
        if states is not None:
            p["recurrent_states"] = states
        return p

    toks, rows = [], []
    # EVERYTHING that touches the cache runs under inference_mode: the loader
    # allocated its pools as inference tensors, and get_new_state() clears the
    # slot in place (gated_delta_net.py:152 -> :345) -- outside inference_mode
    # that raises (the live failure of 2026-10-11 23:20).
    with torch.inference_mode():
        try:
            if recurrent:
                states = [cache.get_new_state()]
            ids = torch.tensor([prompt], dtype=torch.long)
            for start in range(0, S - 1, prefill_chunk):
                end = min(start + prefill_chunk, S - 1)
                model.prefill(ids[:, start:end], params(start))
            pos, tok = S - 1, prompt[-1]
            for _ in range(n):
                logits = model.forward(torch.tensor([[tok]], dtype=torch.long), params(pos))
                row = logits[0, -1]
                if vocab_size:
                    row = row[:vocab_size]
                row = row.float()
                tok = int(torch.argmax(row))
                toks.append(tok)
                rows.append(row.cpu().numpy().copy())
                pos += 1
        finally:
            for s in states or ():
                cache.release_state(s)
    return toks, np.stack(rows) if rows else np.zeros((0, 0), np.float32)


def cmd_reference(a) -> int:
    torch = _torch()
    from exllamav3 import Cache, Config, Model, Tokenizer
    import exllamav3
    cfg = Config.from_directory(a.model_dir)
    model = Model.from_config(cfg)
    cache = make_cache(Cache, model, a.max_num_tokens, max_batch_size=1)
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, torch.device(a.device))
    t0 = time.time()
    model.load(device=a.device)
    tokenizer = Tokenizer.from_config(cfg)
    prompt_ids = _prompt_ids(a, tokenizer)
    t1 = time.time()
    toks, rows = reference_run(model, cache, prompt_ids, a.n, max_num_tokens=a.max_num_tokens,
                               prefill_chunk=a.prefill_chunk, vocab_size=cfg.vocab_size)
    t2 = time.time()
    write_dump(a.out, prompt_ids, toks, rows, meta={
        "source": "reference", "model_dir": a.model_dir, "device": a.device,
        "exllamav3": getattr(exllamav3, "__version__", "?"), "prefill_chunk": a.prefill_chunk,
        "recurrent": bool(model.caps.get("recurrent_states")), "load_s": round(t1 - t0, 1),
        "generate_s": round(t2 - t1, 2)})
    print(f"reference: {len(prompt_ids)} prompt tokens, {len(toks)} generated in "
          f"{t2 - t1:.2f}s -> {a.out}")
    print(f"text: {tokenizer.decode(torch.tensor(toks))!r}")
    return 0


def _prompt_ids(a, tokenizer) -> List[int]:
    if getattr(a, "prompt_ids", None):
        return [int(t) for t in json.loads(Path(a.prompt_ids).read_text())]
    text = Path(a.prompt_file).read_text(encoding="utf-8") if getattr(a, "prompt_file", None) \
        else a.prompt
    return encode_prompt(tokenizer, text)  # the worker's one-shot encoding, exactly


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------

def _top2(row: np.ndarray) -> List[Tuple[int, float]]:
    idx = np.argsort(-row, kind="stable")[:2]
    return [(int(i), float(row[i])) for i in idx]


def compare_dumps(ref: Dict, cand: Dict, atol: float = 1e-2) -> Dict:
    """The verdict as a dict (see the module docstring for the rules)."""
    out: Dict = {"atol": atol}
    if list(ref["prompt_ids"]) != list(cand["prompt_ids"]):
        out.update(passed=False, reason="the two runs used different prompts",
                   prompt_tokens=[len(ref["prompt_ids"]), len(cand["prompt_ids"])])
        return out
    rt, ct = list(ref["tokens"]), list(cand["tokens"])
    n = min(len(rt), len(ct))
    div = next((i for i in range(n) if rt[i] != ct[i]), None)
    out.update(prompt_tokens=len(ref["prompt_ids"]), ref_tokens=len(rt), cand_tokens=len(ct),
               matched=div if div is not None else n, first_divergence=div,
               tokens_identical=div is None and len(rt) == len(ct))
    rl, cl = np.asarray(ref["logits"]), np.asarray(cand["logits"])
    have = rl.ndim == 2 and cl.ndim == 2 and rl.size and cl.size
    if have and rl.shape[1] != cl.shape[1]:
        out.update(passed=False, reason=f"vocab widths differ: {rl.shape[1]} vs {cl.shape[1]}")
        return out
    if have:
        steps = min(rl.shape[0], cl.shape[0], n if div is None else div + 1)
        diffs = np.abs(rl[:steps].astype(np.float64) - cl[:steps].astype(np.float64))
        per_step = diffs.max(axis=1) if steps else np.zeros(0)
        worst = int(per_step.argmax()) if steps else None
        out.update(logit_steps=steps,
                   max_abs_diff=float(per_step.max()) if steps else 0.0,
                   worst_step=worst,
                   allclose_1e3=bool(np.allclose(rl[:steps], cl[:steps], atol=1e-3, rtol=1e-3)),
                   per_step_max=[float(x) for x in per_step])
        if div is not None and div < min(rl.shape[0], cl.shape[0]):
            out["divergence_top2"] = {"reference": _top2(rl[div]), "candidate": _top2(cl[div])}
    else:
        out.update(logit_steps=0, max_abs_diff=None,
                   note="no logits in one of the dumps: tokens compared only")
    ok = out["tokens_identical"] and (out["max_abs_diff"] is None or out["max_abs_diff"] <= atol)
    out["passed"] = bool(ok)
    if not ok:
        if not out["tokens_identical"]:
            out["reason"] = (f"tokens diverge at step {div}" if div is not None
                             else f"lengths differ: {len(rt)} vs {len(ct)}")
        else:
            out["reason"] = f"max |logit diff| {out['max_abs_diff']:.3e} > atol {atol}"
    return out


def format_verdict(v: Dict, ref_name: str = "reference", cand_name: str = "candidate") -> str:
    lines = [f"reference: {ref_name}", f"candidate: {cand_name}"]
    if "matched" not in v:
        return "\n".join(lines + [f"VERDICT: FAIL ({v.get('reason')})"])
    lines.append(f"prompt: {v['prompt_tokens']} tokens (identical)")
    lines.append(f"tokens: {v['matched']}/{v['ref_tokens']} identical"
                 + ("" if v["first_divergence"] is None else
                    f"; first divergence at step {v['first_divergence']}"))
    if v.get("max_abs_diff") is not None:
        lines.append(f"logits: {v['logit_steps']} step(s) compared; max |diff| = "
                     f"{v['max_abs_diff']:.3e} (step {v['worst_step']}); "
                     f"allclose(atol=1e-3, rtol=1e-3) = {v['allclose_1e3']}")
    elif v.get("note"):
        lines.append(f"logits: {v['note']}")
    if v.get("divergence_top2"):
        t = v["divergence_top2"]
        lines.append(f"at the divergence: reference top-2 {t['reference']}, "
                     f"candidate top-2 {t['candidate']}")
    lines.append("VERDICT: PASS" if v["passed"] else f"VERDICT: FAIL ({v.get('reason')})")
    return "\n".join(lines)


def cmd_compare(a) -> int:
    v = compare_dumps(read_dump(a.reference), read_dump(a.candidate), atol=a.atol)
    print(json.dumps(v, indent=2) if a.json else format_verdict(v, a.reference, a.candidate))
    return 0 if v["passed"] else 1


# --------------------------------------------------------------------------
# run: reference + a same-machine pipeline + compare
# --------------------------------------------------------------------------

def num_layers_of(model_dir: str) -> int:
    cfg = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    for c in (cfg, cfg.get("text_config") or {}):
        if c.get("num_hidden_layers"):
            return int(c["num_hidden_layers"])
    raise KeyError("num_hidden_layers not found (top-level or text_config)")


def parse_nodes(node_specs: Optional[Sequence[str]], devices: Optional[str],
                total_layers: Optional[int]) -> List[Tuple[str, int, int]]:
    """--node DEVICE:START:END (repeat, rank order) or --devices a,b,c (even
    split, remainder to the early nodes). Validates contiguity and coverage."""
    nodes: List[Tuple[str, int, int]] = []
    if node_specs:
        for spec in node_specs:
            try:
                dev, start, end = spec.rsplit(":", 2)
                nodes.append((dev, int(start), int(end)))
            except ValueError:
                raise ValueError(f"--node {spec!r}: expected DEVICE:START:END, e.g. cuda:1:0:32")
    elif devices:
        devs = [d.strip() for d in devices.split(",") if d.strip()]
        if not total_layers:
            raise ValueError("--devices needs the model's layer count")
        base, rem = divmod(total_layers, len(devs))
        prev = 0
        for i, d in enumerate(devs):
            cnt = base + (1 if i < rem else 0)
            nodes.append((d, prev, prev + cnt))
            prev += cnt
    else:
        raise ValueError("give --node DEVICE:START:END (repeat) or --devices a,b")
    prev = 0
    for i, (_, s, e) in enumerate(nodes):
        if s != prev or e <= s:
            raise ValueError(f"node {i}: slice [{s},{e}) does not continue from {prev}")
        prev = e
    if total_layers and prev != total_layers:
        raise ValueError(f"the slices end at {prev}, the model has {total_layers} layers")
    return nodes


def build_pipeline_commands(python: str, model_dir: str, nodes: Sequence[Tuple[str, int, int]],
                            *, prompt_args: List[str], n: int, dump: str, base_port: int,
                            max_num_tokens: int, prefill_chunk: int,
                            mem_fraction: Optional[float] = None,
                            extra: Sequence[str] = ()) -> List[List[str]]:
    """The worker argv per rank for a pipeline on THIS machine (127.0.0.1):
    token port = base_port, rank r>0 listens on base_port + r. Greedy,
    --ignore-eos, node 0 writes the dump."""
    T = len(nodes)
    cmds = []
    for r, (dev, s, e) in enumerate(nodes):
        argv = [python, "-m", "tools.bigbrain.worker", "--model-dir", model_dir,
                "--device", dev, "--slice", str(s), str(e), "--rank", str(r), "--total", str(T),
                "--max-num-tokens", str(max_num_tokens), "--prefill-chunk", str(prefill_chunk),
                "--max-new", str(n), "--temperature", "0", "--ignore-eos", *prompt_args]
        if mem_fraction:
            argv += ["--mem-fraction", str(mem_fraction)]
        if r > 0:
            argv += ["--in-port", str(base_port + r)]
        if r < T - 1:
            argv += ["--next-host", "127.0.0.1", "--next-port", str(base_port + r + 1)]
            if r == 0:
                argv += ["--token-port", str(base_port)]
        elif T > 1:
            argv += ["--token-host", "127.0.0.1", "--token-port", str(base_port)]
        if r == 0:
            argv += ["--dump", dump]
        cmds.append(argv + list(extra))
    return cmds


def _prompt_args(a) -> List[str]:
    if a.prompt_ids:
        raise SystemExit("run: --prompt-ids is for http/reference; give --prompt or --prompt-file")
    if a.prompt_file:
        text = Path(a.prompt_file).read_text(encoding="utf-8")
    else:
        text = a.prompt
    return ["--prompt", text]


def _launch(argvs: List[List[str]], workdir: Path, label: str, timeout_s: float) -> int:
    """Start every argv (reverse rank order: listeners first), wait, return the
    worst exit code. Logs: <workdir>/<label>_<rank>.log."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    procs = {}
    for r in reversed(range(len(argvs))):
        log = open(workdir / f"{label}_{r}.log", "w")
        print(f"  start {label} {r}: {shlex.join(argvs[r])}")
        procs[r] = (subprocess.Popen(argvs[r], cwd=str(PKG_ROOT), stdout=log,
                                     stderr=subprocess.STDOUT, env=env), log)
    deadline = time.time() + timeout_s
    worst = 0
    try:
        while procs:
            for r, (p, log) in list(procs.items()):
                rc = p.poll()
                if rc is not None:
                    log.close()
                    worst = max(worst, rc if rc >= 0 else 128 - rc)
                    print(f"  {label} {r} exited rc={rc}")
                    del procs[r]
            if procs and time.time() > deadline:
                raise TimeoutError(f"{label} did not finish within {timeout_s:.0f}s")
            time.sleep(0.5)
    finally:
        for p, log in procs.values():
            p.kill()
            log.close()
    return worst


def cmd_run(a) -> int:
    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    total = num_layers_of(a.model_dir) if (a.devices or a.check_coverage) else None
    nodes = parse_nodes(a.node, a.devices, total)
    prompt_args = _prompt_args(a)
    ref = a.reference or str(workdir / "reference.npz")
    if not a.reference:
        argv = [a.python, "-m", "tools.bigbrain.parity_check", "reference", "--model-dir",
                a.model_dir, "--device", a.ref_device, "--n", str(a.n), "--out", ref,
                "--max-num-tokens", str(a.max_num_tokens), "--prefill-chunk",
                str(a.prefill_chunk), *prompt_args]
        if a.mem_fraction:
            argv += ["--mem-fraction", str(a.mem_fraction)]
        print("phase 1: monolithic reference")
        if _launch([argv], workdir, "reference", a.timeout) != 0:
            print(f"reference FAILED: see {workdir / 'reference_0.log'}")
            return 2
    pipe = str(workdir / "pipeline.npz")
    print(f"phase 2: pipeline over {len(nodes)} node(s): "
          + ", ".join(f"{d}[{s},{e})" for d, s, e in nodes))
    cmds = build_pipeline_commands(a.python, a.model_dir, nodes, prompt_args=prompt_args,
                                   n=a.n, dump=pipe, base_port=a.base_port,
                                   max_num_tokens=a.max_num_tokens,
                                   prefill_chunk=a.prefill_chunk, mem_fraction=a.mem_fraction,
                                   extra=a.worker_arg or ())
    if _launch(cmds, workdir, "node", a.timeout) != 0:
        print(f"pipeline FAILED: see {workdir}/node_*.log")
        return 2
    print("phase 3: compare")
    v = compare_dumps(read_dump(ref), read_dump(pipe), atol=a.atol)
    print(format_verdict(v, ref, pipe))
    (workdir / "verdict.json").write_text(json.dumps(v, indent=2))
    return 0 if v["passed"] else 1


# --------------------------------------------------------------------------
# http: a running server.py against a reference dump
# --------------------------------------------------------------------------

def fetch_http_dump(url: str, api_key: Optional[str], prompt_ids: Sequence[int], n: int,
                    timeout_s: float = 3600.0) -> Dict:
    body = json.dumps({"prompt_ids": list(prompt_ids), "max_tokens": n, "temperature": 0,
                       "ignore_eos": True, "return_logits": True}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/bigbrain/generate", data=body,
                                 method="POST", headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"server answered {e.code}: {e.read().decode('utf-8', 'replace')}")
    lg = out["logits"]
    rows = np.frombuffer(base64.b64decode(lg["b64"]), dtype=lg["dtype"]).reshape(lg["shape"])
    return {"prompt_ids": out["prompt_ids"], "tokens": out["tokens"],
            "logits": rows.astype(np.float32), "meta": {"source": "http", "url": url,
                                                        "model": out.get("model")}}


def cmd_http(a) -> int:
    from .server import resolve_api_key
    key = resolve_api_key(a.api_key_file, a.api_key_env, insecure=True)
    ref = read_dump(a.reference)
    cand = fetch_http_dump(a.url, key, ref["prompt_ids"], len(ref["tokens"]), a.timeout)
    if a.out:
        write_dump(a.out, cand["prompt_ids"], cand["tokens"], cand["logits"], cand["meta"])
    v = compare_dumps(ref, cand, atol=a.atol)
    print(json.dumps(v, indent=2) if a.json else format_verdict(v, a.reference, a.url))
    return 0 if v["passed"] else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _add_prompt(p):
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file", default=None, help="read the prompt text from a file")
    p.add_argument("--prompt-ids", default=None,
                   help="reference only: a JSON list of token ids (skips tokenization)")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BigBrain pipeline-vs-monolithic parity check")
    sp = ap.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("reference", help="the whole model on one GPU -> a dump")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--n", type=int, default=64)
    r.add_argument("--out", required=True)
    r.add_argument("--max-num-tokens", type=int, default=8192)
    r.add_argument("--prefill-chunk", type=int, default=2048)
    r.add_argument("--mem-fraction", type=float, default=None)
    _add_prompt(r)

    c = sp.add_parser("compare", help="two dumps -> verdict")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--atol", type=float, default=1e-2)
    c.add_argument("--json", action="store_true")

    u = sp.add_parser("run", help="reference + same-machine pipeline + compare")
    u.add_argument("--model-dir", required=True)
    u.add_argument("--python", default=sys.executable,
                   help="the venv python for the child processes")
    u.add_argument("--ref-device", default="cuda:0")
    u.add_argument("--reference", default=None, help="reuse an existing reference dump")
    u.add_argument("--node", action="append", metavar="DEVICE:START:END",
                   help="one pipeline node, in rank order (repeat)")
    u.add_argument("--devices", default=None, help="comma list: split the layers evenly")
    u.add_argument("--check-coverage", action="store_true",
                   help="with --node: require the slices to cover every layer")
    u.add_argument("--n", type=int, default=64)
    u.add_argument("--workdir", default="./bigbrain_parity")
    u.add_argument("--base-port", type=int, default=17900)
    u.add_argument("--max-num-tokens", type=int, default=8192)
    u.add_argument("--prefill-chunk", type=int, default=2048)
    u.add_argument("--mem-fraction", type=float, default=None)
    u.add_argument("--atol", type=float, default=1e-2)
    u.add_argument("--timeout", type=float, default=3600.0, help="per phase, seconds")
    u.add_argument("--worker-arg", action="append",
                   help="an extra flag passed to every worker (repeat), e.g. --rebase-layers")
    _add_prompt(u)

    h = sp.add_parser("http", help="a running server.py vs a reference dump")
    h.add_argument("--url", required=True, help="e.g. http://10.0.0.11:16890")
    h.add_argument("--reference", required=True)
    h.add_argument("--api-key-file", default=None)
    h.add_argument("--api-key-env", default="BIGBRAIN_API_KEY")
    h.add_argument("--out", default=None, help="also save the server's run as a dump")
    h.add_argument("--atol", type=float, default=1e-2)
    h.add_argument("--timeout", type=float, default=3600.0)
    h.add_argument("--json", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_arg_parser().parse_args(argv)
    if getattr(a, "worker_arg", None):
        a.worker_arg = [x for arg in a.worker_arg for x in shlex.split(arg)]
    return {"reference": cmd_reference, "compare": cmd_compare, "run": cmd_run,
            "http": cmd_http}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
