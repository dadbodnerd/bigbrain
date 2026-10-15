#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/verify.py -- READ-ONLY. For every host in a hosts file (patches/hosts.example.toml is the template): the upstream
version of each TabbyAPI tree and exllamav3 venv, and the --check state of every patch, compared with what that file says the
host should carry (the upstream pins are PINS.toml). A difference is REPORTED, never fixed: this script runs `--check` /
`git rev-parse` / a dist-info listing and nothing else.

    python patches/verify.py --hosts patches/hosts.toml --all          # every host in the file (ssh for the Linux ones)
    python patches/verify.py --hosts patches/hosts.toml --host alpha --host beta
    python patches/verify.py --hosts patches/hosts.toml --all --json   # machine-readable

The hosts file is --hosts FILE, else $BIGBRAIN_PATCH_HOSTS, else site.toml's [patches] hosts.

How a Linux host is checked without copying anything to it: the patch script and the shared kit (_patchkit.py) are sent
on stdin to the host's `python3 -` (the scripts need only the standard library). `.patch` files go to `git apply
--check [-R] -` on stdin. A host with `local = true` is checked in-process (run verify.py on that machine).

exit 0 = every host matches its entry; 1 = at least one difference; 3 = a host could not be reached (nothing was checked there).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import patchfile  # noqa: E402
from tools.bigbrain import site  # noqa: E402  (a host's `ssh` may be a name from site.toml [hosts])

ENV_HOSTS = "BIGBRAIN_PATCH_HOSTS"

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
STATE_WORDS = {0: "applied", 1: "not applied", 2: "drifted"}


def load_pins(path=None, hosts_path=None) -> dict:
    """PINS.toml (upstream pins + patches) with a `hosts` table from the hosts file: `hosts_path`, else $BIGBRAIN_PATCH_HOSTS,
    else site.toml's [patches] hosts. With none of them `hosts` is empty."""
    with open(path or HERE / "PINS.toml", "rb") as fh:
        pins = tomllib.load(fh)
    hp = hosts_path or os.environ.get(ENV_HOSTS) or site.patch_hosts_path()
    if hp:
        with open(hp, "rb") as fh:
            pins["hosts"] = tomllib.load(fh).get("hosts", {})
    pins.setdefault("hosts", {})
    return pins


def rpath(p: str) -> str:
    """A path for the remote shell: a leading ~/ becomes "$HOME"/ (quoted, so it expands), the rest is quoted."""
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    return shlex.quote(p)


def bootstrap_program(script: Path, args: list) -> str:
    """One self-contained python program (for `python3 -` on stdin) that runs a patch script with the shared kit, so nothing
    is copied to the host: the kit is registered as the module `_patchkit`, then the script runs as __main__ with `args`."""
    kit = (HERE / "_patchkit.py").read_text(encoding="utf-8")
    src = script.read_text(encoding="utf-8")
    fake_file = "/bigbrain/patches/" + script.parent.name + "/" + script.name
    return ("import sys, types\n"
            "_m = types.ModuleType('_patchkit'); sys.modules['_patchkit'] = _m\n"
            f"exec(compile({kit!r}, '_patchkit.py', 'exec'), _m.__dict__)\n"
            f"sys.argv = {[script.name, *args]!r}\n"
            f"exec(compile({src!r}, {script.name!r}, 'exec'), {{'__name__': '__main__', '__file__': {fake_file!r}}})\n")


class Runner:
    """Runs read-only commands on a host: in-process for `local`, over ssh otherwise."""

    def __init__(self, name: str, ssh: str | None):
        self.name, self.ssh = name, ssh

    @property
    def local(self) -> bool:
        return not self.ssh

    def shell(self, command: str, stdin: bytes | None = None, timeout: int = 90):
        p = subprocess.run([*SSH, self.ssh, command], input=stdin, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")

    def reachable(self) -> bool:
        if self.local:
            return True
        try:
            return self.shell("true", timeout=20)[0] == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    # -- upstream versions -------------------------------------------------
    def git_head(self, tree: str) -> str:
        if self.local:
            p = subprocess.run(["git", "-C", os.path.expanduser(tree), "rev-parse", "HEAD"], capture_output=True, text=True)
            return p.stdout.strip() if p.returncode == 0 else "?"
        rc, out, _ = self.shell(f"git -C {rpath(tree)} rev-parse HEAD")
        return out.strip() if rc == 0 else "?"

    def exllamav3_version(self, venv: str) -> str:
        prog = ("import glob, os\n"
                f"v = os.path.expanduser({venv!r})\n"
                "hits = []\n"
                "for sub in ('lib/python3*/site-packages', 'Lib/site-packages'):\n"
                "    hits += glob.glob(os.path.join(v, *sub.split('/'), 'exllamav3-*.dist-info'))\n"
                "print(os.path.basename(hits[0])[len('exllamav3-'):-len('.dist-info')] if hits else '?')\n")
        if self.local:
            p = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
            return p.stdout.strip() or "?"
        rc, out, _ = self.shell("python3 -", stdin=prog.encode())
        return out.strip() if rc == 0 and out.strip() else "?"

    # -- one patch ---------------------------------------------------------
    def script_state(self, script: Path, target: str, flags=()) -> tuple[int, str]:
        """(--check exit code, its output) for a patch script."""
        args = ["--target", target, "--check", *flags]
        if self.local:
            p = subprocess.run([sys.executable, "-B", str(script), *args], capture_output=True, text=True)
            return p.returncode, p.stdout.strip()
        rc, out, err = self.shell("python3 -", stdin=bootstrap_program(script, args).encode())
        return rc, (out.strip() or err.strip())

    def patchfile_state(self, patch: Path, tree: str) -> tuple[int, str]:
        if self.local:
            run = patchfile.local_runner(str(patch), os.path.expanduser(tree))
        else:
            data = patch.read_bytes()

            def run(extra):
                rc, _, err = self.shell(f"git -C {rpath(tree)} apply {' '.join(extra)} -", stdin=data)
                return rc, err
        state, why = patchfile.classify(run)
        code = {"applied": 0, "not applied": 1, "drifted": 2}[state]
        return code, f"{patch.name}: {state}" + (f" -- {why}" if why else "")


def patch_file(pins: dict, name: str) -> tuple[Path, str]:
    spec = pins["patches"][name]
    return HERE / spec["path"], spec["kind"]


def check_target(runner: Runner, pins: dict, tree: str, kind: str, entry: dict, report: list) -> list:
    """Compare one tree/venv with its pins. Returns the list of differences (strings)."""
    diffs = []
    label = f"{runner.name}  {kind}  {tree}"
    if kind == "tabbyapi":
        found = runner.git_head(tree)
        want = pins["upstream"]["tabbyapi"]["commit"]
        ok = found == want
        report.append({"host": runner.name, "target": tree, "what": "upstream commit", "found": found[:12], "expected": want[:12], "ok": ok})
        if not ok:
            diffs.append(f"{label}: TabbyAPI is at {found[:12]}, PINS.toml says {want[:12]}")
    else:
        found = runner.exllamav3_version(tree)
        want = entry.get("version")
        ok = want is None or found == want
        report.append({"host": runner.name, "target": tree, "what": "exllamav3 version", "found": found, "expected": want, "ok": ok})
        if not ok:
            diffs.append(f"{label}: exllamav3 is {found}, the hosts file says {want}")
    for want_state, names in (("applied", entry.get("applied", [])), ("not applied", entry.get("absent", []))):
        for name in names:
            path, pkind = patch_file(pins, name)
            if pkind == "script":
                rc, text = runner.script_state(path, tree)
            else:
                rc, text = runner.patchfile_state(path, tree)
            got = STATE_WORDS.get(rc, f"exit {rc}")
            ok = got == want_state
            shown = got
            if rc == 1 and "built in, never patch" in text:
                shown = "not applied (built in)"          # a version that carries the feature itself: not drift, just not ours to patch
            report.append({"host": runner.name, "target": tree, "what": f"patch {name}", "found": shown, "expected": want_state, "ok": ok})
            if not ok:
                diffs.append(f"{label}: patch {name} is {got}, the hosts file says {want_state}"
                             + (f"  [{text.splitlines()[-1]}]" if text else ""))
    return diffs


def verify(pins: dict, hosts: list[str]) -> tuple[list, list, list]:
    report, diffs, unreachable = [], [], []
    for name in hosts:
        h = pins["hosts"][name]
        runner = Runner(name, None if h.get("local") else site.ssh_target(h["ssh"]))
        if not runner.reachable():
            unreachable.append(name)
            continue
        for entry in h.get("tabbyapi", []):
            diffs += check_target(runner, pins, entry["dir"], "tabbyapi", entry, report)
        for entry in h.get("exllamav3", []):
            diffs += check_target(runner, pins, entry["venv"], "exllamav3", entry, report)
    return report, diffs, unreachable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="every host in the hosts file")
    ap.add_argument("--host", action="append", default=[], help="one host (repeatable)")
    ap.add_argument("--hosts", default=None, help=f"the hosts file (default: ${ENV_HOSTS}, then site.toml [patches] hosts)")
    ap.add_argument("--pins", default=None, help="PINS.toml (default: beside this script)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    pins = load_pins(a.pins, a.hosts)
    if not pins["hosts"]:
        ap.error("no hosts to check: pass --hosts FILE (patches/hosts.example.toml is the template), set "
                 f"${ENV_HOSTS}, or set [patches] hosts in site.toml")
    hosts = list(pins["hosts"]) if a.all or not a.host else a.host
    unknown = [h for h in hosts if h not in pins["hosts"]]
    if unknown:
        ap.error(f"not in the hosts file: {', '.join(unknown)} (known: {', '.join(pins['hosts'])})")
    report, diffs, unreachable = verify(pins, hosts)
    if a.json:
        print(json.dumps({"report": report, "differences": diffs, "unreachable": unreachable}, indent=1))
    else:
        last = None
        for r in report:
            head = (r["host"], r["target"])
            if head != last:
                print(f"\n{r['host']}: {r['target']}")
                last = head
            mark = "ok  " if r["ok"] else "DIFF"
            print(f"  {mark} {r['what']:<26} {str(r['found']):<24} (pinned: {r['expected']})")
        for u in unreachable:
            print(f"\n{u}: NOT REACHED -- nothing was checked there")
        print()
        if diffs:
            print(f"{len(diffs)} difference(s) from the hosts file (reported, nothing was changed):")
            for d in diffs:
                print("  - " + d)
        elif not unreachable:
            print("every host matches the hosts file")
    if unreachable:
        return 3
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
