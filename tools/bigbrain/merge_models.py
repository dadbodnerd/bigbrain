#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""tools/bigbrain/merge_models.py -- add measured models to model_registry.json and mark what is no longer on disk, WITHOUT touching
anything else in the file (nodes, GPU uuids and indices, notes, other models).

    python -m tools.bigbrain.merge_models --registry PATH --update update.json            # dry run: prints what would change
    python -m tools.bigbrain.merge_models --registry PATH --update update.json --write    # backs the file up (.bak-<stamp>), then replaces it

`update.json` (written by a read-only measuring pass over ssh) has:
    add          {model name: full registry entry}      new models, or a known one whose entry is missing facts
    path_fixes   {model name: {node name: new folder}}  a folder that moved (a symlink target, a library copy)
    missing      [{model, host, path, reason}]          folders that no longer exist: the model stays listed, `missing_on` names the nodes
    size_differences, notes                              carried into the report only

Rules: an existing entry is never replaced, only completed (a key it lacks is filled; `paths` gains hosts, never loses one);
`missing_on` is recomputed from `missing` plus the folders that are present again; nothing is deleted; `nodes` is untouched, so GPU
indices pinned by UUID stay exactly as they were. The result is validated (every model keeps a dict entry, the node block is
byte-identical) before anything is written. Exit 0 = ok / written, 2 = refused.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple


def merge(reg: Dict[str, Any], upd: Dict[str, Any], now: str = "") -> Tuple[Dict[str, Any], List[str]]:
    """(the new registry, report lines). Pure."""
    new = copy.deepcopy(reg)
    models = new.setdefault("models", {})
    report: List[str] = []
    for name, entry in (upd.get("add") or {}).items():
        if not isinstance(entry, dict):
            continue
        if name not in models:
            models[name] = copy.deepcopy(entry)
            report.append(f"added {name} ({entry.get('size_gib')} GiB, {entry.get('bits')} bpw, on {', '.join((entry.get('paths') or {}).keys()) or 'no host'})")
            continue
        have = models[name]
        for k, v in entry.items():
            if k == "paths":
                for host, path in (v or {}).items():
                    if host not in (have.get("paths") or {}):
                        have.setdefault("paths", {})[host] = path
                        report.append(f"{name}: now also on {host}")
            elif have.get(k) is None and v is not None:
                have[k] = v
                report.append(f"{name}: filled {k}")
    for name, fixes in (upd.get("path_fixes") or {}).items():
        if name in models:
            for host, path in (fixes or {}).items():
                if (models[name].get("paths") or {}).get(host) != path:
                    models[name].setdefault("paths", {})[host] = path
                    report.append(f"{name}: path on {host} fixed to {path}")
    gone: Dict[str, set] = {}
    for m in upd.get("missing") or []:
        if isinstance(m, dict) and m.get("model") in models and m.get("host"):
            gone.setdefault(m["model"], set()).add(m["host"])
    if "missing" in upd:                                              # an audit ran: a model it did not list is present everywhere
        for name, info in models.items():
            want = sorted(gone.get(name, set()))
            if sorted(info.get("missing_on") or []) != want:
                if want:
                    info["missing_on"] = want
                    report.append(f"{name}: missing on disk on {', '.join(want)}")
                else:
                    info.pop("missing_on", None)
                    report.append(f"{name}: present again")
    if report:
        new["_models_refreshed"] = {"at": now or time.strftime("%Y-%m-%dT%H:%M:%S"), "by": "tools/bigbrain/merge_models.py",
                                    "notes": [str(n) for n in (upd.get("notes") or [])][:10]}
    return new, report


def validate(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    bad: List[str] = []
    if json.dumps(old.get("nodes"), sort_keys=True) != json.dumps(new.get("nodes"), sort_keys=True):
        bad.append("the node block changed (it must not)")
    for name in old.get("models") or {}:
        if not isinstance((new.get("models") or {}).get(name), dict):
            bad.append(f"model {name} was dropped")
    for k in old:
        if k not in new:
            bad.append(f"top-level key {k} was dropped")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--update", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    with open(a.registry, encoding="utf-8-sig") as fh:
        reg = json.load(fh)
    with open(a.update, encoding="utf-8-sig") as fh:
        upd = json.load(fh)
    new, report = merge(reg, upd)
    bad = validate(reg, new)
    for line in report:
        print("  " + line)
    if bad:
        print("REFUSED: " + "; ".join(bad))
        return 2
    if not report:
        print("no change")
        return 0
    if not a.write:
        print("(dry run: add --write)")
        return 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(a.registry, "rb") as fh:
        raw = fh.read()
    with open(f"{a.registry}.bak-{stamp}-models", "wb") as fh:
        fh.write(raw)
    crlf = b"\r\n" in raw
    text = json.dumps(new, indent=1) + "\n"
    tmp = f"{a.registry}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        fh.write((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    os.replace(tmp, a.registry)
    print(f"wrote {a.registry} (the old one is {a.registry}.bak-{stamp}-models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
