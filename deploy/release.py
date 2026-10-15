#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""deploy/release.py -- put one commit of this repo on a host as an immutable release dir.

    python deploy/release.py --commit v0.2.1 --host alpha --host beta      # names from site.toml [hosts], or user@address
    python deploy/release.py --commit v0.2.1 --to-dir /tmp/releases           # a local directory instead of ssh (tests, dry runs)
    python deploy/release.py --commit v0.2.1 --host alpha --dry-run          # say what would happen; touch nothing

What a release is: `git archive <commit>` extracted into <dest>/<first 12 hex of the commit>/ on the host, with a file
BUILD holding the FULL commit hash, then made read-only (`chmod -R a-w`), then a symlink <dest>/current pointing at it.
The dir is built as <dest>/.<short>.partial and renamed into place last, so a half-copied release can never be mistaken for a
finished one. An existing release dir is never touched again: a second run says EXISTS and leaves it as it is. The default
<dest> is ~/bigbrain/releases. The release is the code only; data (the model registry, api keys, node.env) never goes in one.

Nothing is installed or started on the host and no process is touched: it only writes the new directory. The tests exercise it
on a local directory.
Exit 0 = every target installed or already there; 1 = a target failed.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import os
import shlex
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if importlib.util.find_spec("bigbrain") is None:       # a bare checkout, nothing pip-installed: use this checkout's src/
    sys.path.insert(0, str(REPO / "src"))
from bigbrain import site  # noqa: E402  (host names come from site.toml [hosts]; `user@address` is accepted as it is)

DEFAULT_DEST = "~/bigbrain/releases"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def resolve_commit(rev: str, repo: Path = REPO) -> str:
    p = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", rev + "^{commit}"], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"{rev!r} is not a commit of {repo}: {p.stderr.strip()}")
    return p.stdout.strip()


def archive_bytes(commit: str, repo: Path = REPO) -> bytes:
    p = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit], capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"git archive {commit} failed: {p.stderr.decode(errors='replace').strip()}")
    return p.stdout


def short_of(full: str) -> str:
    return full[:12]


def rpath(p: str) -> str:
    """Quote a path for the remote shell; a leading ~/ becomes "$HOME"/ so it still expands."""
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    return shlex.quote(p)


def remote_script(dest: str, full: str) -> str:
    """The shell program that reads the tar on stdin and builds the release. Prints INSTALLED or EXISTS."""
    short = short_of(full)
    d, partial = rpath(f"{dest}/{short}"), rpath(f"{dest}/.{short}.partial")
    root = rpath(dest)
    return (f"set -eu; mkdir -p {root}; "
            f"if [ -e {d} ]; then echo EXISTS; cat >/dev/null; exit 0; fi; "
            f"chmod -R u+w {partial} 2>/dev/null || true; rm -rf {partial}; mkdir {partial}; "
            f"tar -x -C {partial}; printf '%s\\n' {shlex.quote(full)} > {partial}/BUILD; "
            f"chmod -R a-w {partial}; mv {partial} {d}; ln -sfn {shlex.quote(short)} {root}/current; echo INSTALLED")


def install_remote(host: str, dest: str, full: str, archive: bytes) -> str:
    target = site.ssh_target(host)
    p = subprocess.run([*SSH, "--", target, remote_script(dest, full)], input=archive, capture_output=True)
    out = p.stdout.decode(errors="replace").strip()
    if p.returncode != 0 or out.split()[-1:] not in (["INSTALLED"], ["EXISTS"]):
        raise RuntimeError(f"{target}: exit {p.returncode}: {(p.stderr.decode(errors='replace') or out).strip()[:300]}")
    return out.split()[-1]


def install_local(dest_root: str, full: str, archive: bytes) -> str:
    """The same release, into a local directory. Used by the tests; the ssh path above is the one a real deploy uses."""
    root = Path(os.path.expanduser(dest_root))
    root.mkdir(parents=True, exist_ok=True)
    final, partial = root / short_of(full), root / f".{short_of(full)}.partial"
    if final.exists():
        return "EXISTS"
    if partial.exists():
        _make_writable(partial)
        _rmtree(partial)
    partial.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        try:
            tf.extractall(partial, filter="data")
        except TypeError:                              # python < 3.11.4 has no extraction filters: check every member ourselves
            extract_checked(tf, partial)
    (partial / "BUILD").write_text(full + "\n", encoding="ascii", newline="\n")
    for p in sorted(partial.rglob("*"), key=lambda q: len(q.parts), reverse=True) + [partial]:
        mode = p.stat().st_mode
        os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    partial.rename(final)
    link = root / "current"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(short_of(full), link)
    except OSError:                                   # Windows without the symlink privilege
        (root / "current.txt").write_text(short_of(full) + "\n", encoding="ascii")
    return "INSTALLED"


def extract_checked(tf: tarfile.TarFile, dest) -> None:
    """Extract `tf` into `dest` taking only plain files and directories whose paths stay inside `dest`; anything else (an absolute
    path, a `..`, a link, a device) raises ValueError before a byte is written. What filter="data" does on Python >= 3.11.4, for
    the Pythons that lack it (security review S10)."""
    root = Path(dest).resolve()
    members = []
    for m in tf.getmembers():
        name = m.name.replace("\\", "/")
        target = (root / name).resolve()
        if name.startswith("/") or os.path.isabs(m.name) or ".." in name.split("/") or not (target == root or root in target.parents):
            raise ValueError(f"refusing archive member {m.name!r}: it would land outside {dest}")
        if not (m.isfile() or m.isdir()):
            raise ValueError(f"refusing archive member {m.name!r}: only plain files and directories are extracted")
        members.append(m)
    tf.extractall(root, members=members)


def _make_writable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        try:
            os.chmod(p, p.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commit", required=True, help="a commit, tag or branch of this repo (v0.2.1)")
    ap.add_argument("--host", action="append", default=[],
                    help="a host: a name from site.toml [hosts], or user@address; repeatable")
    ap.add_argument("--to-dir", default=None, help="install into this local directory instead of over ssh")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"the releases directory on the host (default {DEFAULT_DEST})")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.host and not a.to_dir:
        ap.error("name at least one --host, or --to-dir")
    full = resolve_commit(a.commit)
    archive = archive_bytes(full)
    print(f"{a.commit} = {full}  ({len(archive):,} bytes of tar)")
    failed = 0
    targets = [("local:" + a.to_dir, lambda: install_local(a.to_dir, full, archive))] if a.to_dir else []
    targets += [(h, (lambda h=h: install_remote(h, a.dest, full, archive))) for h in a.host]
    for name, run in targets:
        if a.dry_run:
            print(f"[dry-run] {name}: would extract into {a.dest if name != 'local:' + str(a.to_dir) else a.to_dir}/{short_of(full)} "
                  f"(read-only, BUILD = {full}) and point `current` at it")
            continue
        try:
            print(f"{name}: {run()}  {short_of(full)}")
        except Exception as exc:  # noqa: BLE001 -- report each target, keep going
            failed += 1
            print(f"{name}: FAILED -- {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
