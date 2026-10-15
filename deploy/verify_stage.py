#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""deploy/verify_stage.py -- READ-ONLY: is what a host holds exactly what a commit says?

    python deploy/verify_stage.py --commit v0.2.1 --host alpha --stage "~/.cache/bigbrain/stage"
    python deploy/verify_stage.py --commit v0.2.1 --host alpha --release ~/bigbrain/releases/<short>
    python deploy/verify_stage.py --commit v0.2.1 --stage ./stage_dir            # no --host: a local directory

--stage   a node's stage dir: the files the controller stages (controller.PACKAGE_FILES: the 8 files of the `bigbrain` package
          a node runs, e.g. bigbrain/pipe.py). Each is compared with `git show <commit>:src/<path>`.
--release a release dir made by deploy/release.py: every file the commit tracks, plus BUILD = the full commit hash.

Per file: IDENTICAL (same bytes), IDENTICAL-LF (same after CRLF -> LF: what a stage dir reads like when it was staged from a CRLF
Windows checkout), DIFFERENT, or MISSING. Extra files are not an error in a stage dir (logs, api_key live there) and are not listed. It runs one `python3 -` over ssh that only reads and hashes.
Exit 0 = every file identical (--strict: byte-identical; default: LF-identical is enough); 1 = a difference; 3 = host unreachable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import release  # noqa: E402  (also makes `bigbrain` importable from this checkout's src/ when it is not pip-installed)

REPO = release.REPO

#: runs on the host (stdin of `python3 -`): reads and hashes, nothing else
PROBE = r'''
import hashlib, json, os, sys
root = os.path.expanduser(sys.argv[1])
out = {}
for rel in json.loads(sys.argv[2]):
    p = os.path.join(root, *rel.split("/"))
    try:
        raw = open(p, "rb").read()
    except OSError:
        out[rel] = None
        continue
    out[rel] = [hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()]
print(json.dumps(out))
'''


#: where the staged package lives in the repo: the controller stages `bigbrain/pipe.py`, the commit tracks `src/bigbrain/pipe.py`
SRC_PREFIX = "src/"


def package_files() -> list:
    """The files the controller stages to a node (read from the controller itself, so the two cannot disagree)."""
    from bigbrain.controller import PACKAGE_FILES  # noqa: PLC0415
    return list(PACKAGE_FILES)


def expected_hashes(commit: str, paths: list, prefix: str = "") -> dict:
    """{path: bytes of that file at `commit`, or None when the commit does not track it}. `prefix` is put in front of each path
    for the lookup only (--stage: SRC_PREFIX); the result is keyed by the path as given."""
    out = {}
    for rel in paths:
        p = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{prefix}{rel}"], capture_output=True)
        out[rel] = p.stdout if p.returncode == 0 else None
    return out


def tracked_files(commit: str) -> list:
    p = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", commit], capture_output=True, text=True, check=True)
    return [line for line in p.stdout.splitlines() if line]


def probe(host: str | None, root: str, paths: list) -> dict:
    if host is None:
        res = subprocess.run([sys.executable, "-B", "-c", PROBE, root, json.dumps(paths)], capture_output=True, text=True)
    else:
        target = release.site.ssh_target(host)
        prog = PROBE
        res = subprocess.run([*release.SSH, "--", target, "python3 - " + release.rpath(root) + " " + _q(json.dumps(paths))],
                             input=prog.encode(), capture_output=True)
        res.stdout = res.stdout.decode("utf-8", "replace") if isinstance(res.stdout, bytes) else res.stdout
        res.stderr = res.stderr.decode("utf-8", "replace") if isinstance(res.stderr, bytes) else res.stderr
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout).strip()[:300] or f"exit {res.returncode}")
    return json.loads(res.stdout)


def _q(s: str) -> str:
    import shlex
    return shlex.quote(s)


def compare(expected: dict, found: dict) -> dict:
    """{path: 'IDENTICAL' | 'IDENTICAL-LF' | 'DIFFERENT' | 'MISSING'}"""
    verdict = {}
    for rel, want in expected.items():
        got = found.get(rel)
        if got is None:
            verdict[rel] = "MISSING"
        elif want is None:
            verdict[rel] = "DIFFERENT"                                  # the commit does not track it, the host has it
        else:
            raw = hashlib.sha256(want).hexdigest()
            lf = hashlib.sha256(want.replace(b"\r\n", b"\n")).hexdigest()
            if got[0] == raw:
                verdict[rel] = "IDENTICAL"
            elif got[1] == lf:
                verdict[rel] = "IDENTICAL-LF"
            else:
                verdict[rel] = "DIFFERENT"
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commit", required=True)
    ap.add_argument("--host", default=None, help="a name from site.toml [hosts], or user@address; omit for a local directory")
    ap.add_argument("--stage", help="a node's stage dir")
    ap.add_argument("--release", help="a release dir made by release.py")
    ap.add_argument("--strict", action="store_true", help="require byte-identical files (CRLF copies fail)")
    a = ap.parse_args(argv)
    if bool(a.stage) == bool(a.release):
        ap.error("give exactly one of --stage and --release")
    full = release.resolve_commit(a.commit)
    root = a.stage or a.release
    paths = package_files() if a.stage else tracked_files(full)
    expected = expected_hashes(full, paths, SRC_PREFIX if a.stage else "")
    if a.release:
        expected["BUILD"] = (full + "\n").encode("ascii")
    try:
        found = probe(a.host, root, list(expected))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"{a.host or 'local'}: could not read {root}: {exc}", file=sys.stderr)
        return 3
    verdict = compare(expected, found)
    width = max(len(p) for p in verdict)
    for rel, v in verdict.items():
        print(f"  {v:<13} {rel:<{width}}")
    ok_words = {"IDENTICAL"} if a.strict else {"IDENTICAL", "IDENTICAL-LF"}
    bad = [p for p, v in verdict.items() if v not in ok_words]
    print(f"{a.commit} = {full[:12]}: " + (f"{len(verdict)} files, all identical" + ("" if a.strict else " (CRLF copies count)")
                                           if not bad else f"{len(bad)} of {len(verdict)} differ: {', '.join(bad[:6])}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
