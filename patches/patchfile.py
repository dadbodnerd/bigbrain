# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/patchfile.py -- the same --check / --apply / --revert command line as the patch scripts, for a plain unified-diff
`.patch` file (patches/tabbyapi/auth_redact.patch, windows_launcher.patch). It only drives `git apply`:

    python patches/patchfile.py patches/tabbyapi/auth_redact.patch --target ~/tabbyAPI            # --check
    python patches/patchfile.py ...patch --target DIR --apply         # git apply
    python patches/patchfile.py ...patch --target DIR --revert        # git apply -R (refuses unless the reverse applies cleanly,
                                                                       # which is exactly "the touched lines are as the apply left them")
    exit codes   --check: 0 applied (git apply -R --check passes), 1 not applied (git apply --check passes), 2 neither: the file
                 drifted from the version the patch was made against.
                 --apply: 0 done or already; 2 refused.   --revert: 0 done; 1 not applied; 2 refused.

--target is the TabbyAPI checkout (a git work tree or a plain copy; git apply does not need a repository). The working copy may
use CRLF (a Windows checkout with core.autocrlf): git apply converts as it applies. No backup file is made, because the reverse
patch IS the undo. verify.py reuses classify() with an ssh runner.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

APPLIED, NOT_APPLIED, DRIFTED = 0, 1, 2


def classify(run):
    """`run(extra_git_apply_args) -> (returncode, stderr text)` applies the patch with those args. Returns (state, why)."""
    rc, _ = run(["-R", "--check"])
    if rc == 0:
        return "applied", ""
    rc, err = run(["--check"])
    if rc == 0:
        return "not applied", ""
    return "drifted", err.strip().splitlines()[0] if err.strip() else "git apply refused it both ways"


def local_runner(patch_path, target):
    patch_path = os.path.abspath(patch_path)

    def run(extra):
        p = subprocess.run(["git", "-C", target, "apply", *extra, patch_path], capture_output=True, text=True)
        return p.returncode, p.stderr
    return run


def main(argv=None, out=print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("patch", help="the .patch file")
    ap.add_argument("--target", required=True, help="the checkout the patch applies to")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="(default) report; change nothing")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--revert", action="store_true")
    a = ap.parse_args(argv)
    name = os.path.basename(a.patch)
    target = os.path.expanduser(a.target)
    if not os.path.isfile(a.patch):
        out(f"{name}: no such patch file")
        return DRIFTED
    if not os.path.isdir(target):
        out(f"{name}: {target} is not a directory")
        return DRIFTED
    run = local_runner(a.patch, target)
    state, why = classify(run)
    if a.apply:
        if state == "applied":
            out(f"{name}: already applied; nothing to do")
            return APPLIED
        if state == "drifted":
            out(f"{name}: REFUSING: {why}")
            return DRIFTED
        rc, err = run([])
        out(f"{name}: " + ("applied. Takes effect when the program next starts." if rc == 0 else f"git apply failed: {err.strip()}"))
        return APPLIED if rc == 0 else DRIFTED
    if a.revert:
        if state == "not applied":
            out(f"{name}: not applied; nothing to revert")
            return NOT_APPLIED
        if state == "drifted":
            out(f"{name}: REFUSING: the reverse patch does not apply cleanly ({why}); the file changed since the apply")
            return DRIFTED
        rc, err = run(["-R"])
        out(f"{name}: " + ("reverted. Takes effect when the program next starts." if rc == 0 else f"git apply -R failed: {err.strip()}"))
        return APPLIED if rc == 0 else DRIFTED
    out(f"{name}: {'applied' if state == 'applied' else state.upper() if state == 'drifted' else 'not applied'}"
        + (f" -- {why}" if why else ""))
    return {"applied": APPLIED, "not applied": NOT_APPLIED, "drifted": DRIFTED}[state]


if __name__ == "__main__":
    sys.exit(main())
