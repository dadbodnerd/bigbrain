# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/_patchkit.py -- the shared mechanics of every BigBrain patch script.

A patch script declares a `Patch`: a name and a list of `Edit`s. One Edit is one file under --target, a MARKER (text that
is in the file once the edit has been applied) and a function that turns the file's LF text into the patched text, raising
ValueError when an anchor does not match exactly once. The script then calls `main(PATCH)`. The command line is the same
for every patch:

    <script> --target DIR [--check | --apply | --revert] [--diff] [patch-specific flags]

    --check   (the default) say what state each file is in; change nothing.
              exit 0 = applied, 1 = not applied (or only part of it), 2 = anchors drifted: upstream changed, re-read it
    --apply   exit 0 = applied now or already; 2 = refused (anchors drifted, a prerequisite patch is missing, or a
              backup of an unknown file is in the way). Backs up first, keeps each file's line endings, compiles .py
              output before anything is written, all-or-nothing.
    --revert  puts back the backup THIS patch made. exit 0 = reverted; 1 = nothing to revert; 2 = refused: the file
              changed since the apply (another patch went on top? revert that one first), there is no state record
              (patched by an earlier copy of this script that kept only a .bak-/.orig- backup: --adopt it first, or restore
              that backup by hand), or the backup is damaged.
    --adopt   for a host an EARLIER copy of the script patched: finds the old .bak-*/.orig-* backup that reproduces the current file when this
              patch is applied to it, copies it to <file>.bb-orig-NAME and writes the state record, so --revert works from
              then on. exit 0 = adopted (or already recorded); 2 = refused: a file is not patched, or no old backup
              reproduces it (also when another patch sits on top: revert that one with the kit, adopt, then re-apply it).
              It writes only the copy and the record; the old backup stays.

A patch may carry a `gate(target)`: it answers None (go ahead) or ("builtin" | "unsupported", message). --apply refuses on either
answer. --check on "builtin" says the message and exits 1 (it exits 2 if our patch is on a version that has the feature
built in, which is a mistake to undo); on "unsupported" --check reports the files as usual and adds the message.
--apply is all-or-nothing: if writing fails part way, every file already written is put back and the backups and the state record
this run made are removed.

State: for patch NAME, `<file>.bb-orig-NAME` is the file before the apply, and `<target>/.bb-patch-NAME.json` records the
sha256 of every file before and after, which is how --revert knows nothing changed in between.

The scripts only need the standard library, so they also run on a GPU node from stdin (see verify.py) without being copied.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import sys
import time

APPLIED, NOT_APPLIED, DRIFTED = 0, 1, 2          # --check exit codes
REFUSED = 2                                        # --apply / --revert refusal
NOTHING_TO_REVERT = 1

STATE_NAME = ".bb-patch-{name}.json"
BACKUP_SUFFIX = ".bb-orig-{name}"


class Edit:
    """One file of a patch. `apply(text)` maps the file's LF text to the patched text or raises ValueError;
    `needs(text)` raises ValueError when a prerequisite patch is missing; `opt` names the patch-specific flag that
    selects this edit (None = always selected)."""

    def __init__(self, rel, marker, apply, needs=None, opt=None):
        self.rel, self.marker, self.apply, self.needs, self.opt = rel, marker, apply, needs, opt


class Patch:
    def __init__(self, name, summary, edits, options=None, resolve=None, gate=None):
        self.gate = gate or (lambda target: None)     # target -> None | ("builtin" | "unsupported", message)
        self.name = name
        self.summary = summary
        self.edits = list(edits)
        self.options = dict(options or {})            # {"stdio-limit": "help text"} -> --stdio-limit selects Edit(opt=...)
        self.resolve = resolve or (lambda p: p)       # --target DIR -> the directory the edits' paths are under

    @property
    def state_name(self):
        return STATE_NAME.format(name=self.name)

    @property
    def backup_suffix(self):
        return BACKUP_SUFFIX.format(name=self.name)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_file(path):
    """(raw bytes, LF text, has CRLF)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return raw, raw.decode("utf-8").replace("\r\n", "\n"), b"\r\n" in raw


def encode(text: str, crlf: bool) -> bytes:
    return (text.replace("\n", "\r\n") if crlf else text).encode("utf-8")


def write_atomic(path, data: bytes, like=None):
    """Replace `path` with `data` (temp file + os.replace, mode copied from `like` or the old file)."""
    tmp = f"{path}.bb-tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        shutil.copymode(like or path, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _path(target, rel):
    return os.path.join(target, *rel.split("/"))


# ── inspecting one edit ───────────────────────────────────────────────────────────────────────────────────────────

class Inspection:
    def __init__(self, edit, state, why="", raw=b"", text="", crlf=False, new=None):
        self.edit, self.state, self.why, self.raw, self.text, self.crlf, self.new = edit, state, why, raw, text, crlf, new


def inspect_edit(edit: Edit, target: str) -> Inspection:
    """state is 'applied', 'not applied' (and `new` holds the patched LF text), 'blocked' (not applied, and a prerequisite
    patch is missing: `why` says which) or 'drifted' (an anchor no longer matches, and `why` says what)."""
    path = _path(target, edit.rel)
    try:
        raw, text, crlf = read_file(path)
    except OSError as exc:
        return Inspection(edit, "drifted", f"cannot read {path}: {exc.strerror or exc}")
    except UnicodeDecodeError as exc:
        return Inspection(edit, "drifted", f"{path} is not UTF-8: {exc}", raw=b"")
    if edit.marker in text:
        return Inspection(edit, "applied", raw=raw, text=text, crlf=crlf)
    if edit.needs is not None:
        try:
            edit.needs(text)
        except ValueError as exc:                      # a prerequisite patch is missing: not applied, and cannot be yet
            return Inspection(edit, "blocked", str(exc), raw=raw, text=text, crlf=crlf)
    try:
        new = edit.apply(text)
        if edit.rel.endswith(".py"):
            compile(new, path, "exec")                 # a syntax error refuses before anything is written
    except (ValueError, SyntaxError) as exc:
        return Inspection(edit, "drifted", str(exc), raw=raw, text=text, crlf=crlf)
    return Inspection(edit, "not applied", raw=raw, text=text, crlf=crlf, new=new)


def selected_edits(patch: Patch, flags) -> list:
    return [e for e in patch.edits if e.opt is None or (e.opt in flags)]


def _label(i: Inspection) -> str:
    return {"applied": "PATCHED", "not applied": "NOT patched", "blocked": "NOT patched", "drifted": "DRIFTED"}[i.state]


# ── state records ─────────────────────────────────────────────────────────────────────────────────────────────────

def _state_path(patch, target):
    return os.path.join(target, patch.state_name)


def load_state(patch, target):
    try:
        with open(_state_path(patch, target), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) and isinstance(data.get("files"), dict) else None
    except (OSError, ValueError):
        return None


def _save_state(patch, target, state):
    path = _state_path(patch, target)
    if not state["files"]:
        if os.path.exists(path):
            os.unlink(path)
        return
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
        fh.write("\n")


# ── the three actions ─────────────────────────────────────────────────────────────────────────────────────────────

def do_check(patch, target, flags, show_diff=False, out=print) -> int:
    infos = [inspect_edit(e, target) for e in selected_edits(patch, flags)]
    gate = patch.gate(target)
    if gate and gate[0] == "builtin":
        if any(i.state == "applied" for i in infos):
            out(f"{patch.name}: DRIFTED: {gate[1]} -- and our patch is applied anyway; revert it")
            return DRIFTED
        out(f"{patch.name}: {gate[1]}")
        return NOT_APPLIED
    for i in infos:
        line = f"{i.edit.rel}: {_label(i)}"
        if i.state == "drifted":
            line += f" -- {i.why}"
        elif i.state == "blocked":
            line += f" (cannot be applied yet: {i.why})"
        out(line)
        if show_diff and i.state == "not applied":
            for chunk in difflib.unified_diff(i.text.splitlines(True), i.new.splitlines(True),
                                              f"{i.edit.rel} (now)", f"{i.edit.rel} (patched)", n=2):
                out(chunk.rstrip("\n"))
    states = {i.state for i in infos}
    if "drifted" in states:
        verdict, rc = "DRIFTED: an anchor no longer matches exactly once (or a file is unreadable); re-read upstream", DRIFTED
    elif states == {"applied"}:
        verdict, rc = "applied", APPLIED
    elif "applied" in states:
        verdict, rc = "PARTIAL: some files are patched and some are not (--apply finishes the rest)", NOT_APPLIED
    else:
        verdict, rc = "not applied", NOT_APPLIED
    out(f"{patch.name}: {verdict}")
    if gate:
        out(f"{patch.name}: note: {gate[1]}")
    return rc


def _restore_state_file(patch, target, previous):
    path = _state_path(patch, target)
    if previous is None:
        if os.path.exists(path):
            os.unlink(path)
    else:
        with open(path, "wb") as fh:
            fh.write(previous)


def do_apply(patch, target, flags, out=print) -> int:
    gate = patch.gate(target)
    if gate:
        out(f"{patch.name}: REFUSING: {gate[1]}")
        out(f"{patch.name}: refused, nothing was written")
        return REFUSED
    infos = [inspect_edit(e, target) for e in selected_edits(patch, flags)]
    drifted = [i for i in infos if i.state in ("drifted", "blocked")]
    if drifted:
        for i in drifted:
            out(f"{i.edit.rel}: REFUSING: {i.why}")
        out(f"{patch.name}: refused, nothing was written")
        return REFUSED
    todo = [i for i in infos if i.state == "not applied"]
    if not todo:
        out(f"{patch.name}: already applied; nothing to do")
        return APPLIED
    # a backup of that name is only acceptable as the leftover of an apply that died before it wrote the file
    for i in todo:
        bak = _path(target, i.edit.rel) + patch.backup_suffix
        if os.path.exists(bak) and sha256(open(bak, "rb").read()) != sha256(i.raw):
            out(f"{i.edit.rel}: REFUSING: {bak} exists and is not this file's current content; look at it first")
            return REFUSED
    state_file = _state_path(patch, target)
    previous_state = open(state_file, "rb").read() if os.path.exists(state_file) else None
    state = load_state(patch, target) or {"patch": patch.name, "files": {}}
    prepared, made_backups, written = [], [], []
    try:
        for i in todo:
            path = _path(target, i.edit.rel)
            data = encode(i.new, i.crlf)
            bak = path + patch.backup_suffix
            existed = os.path.exists(bak)
            shutil.copy2(path, bak)
            if not existed:
                made_backups.append(bak)
            state["files"][i.edit.rel] = {"before_sha256": sha256(i.raw), "after_sha256": sha256(data),
                                          "backup": os.path.basename(bak), "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            prepared.append((path, data))
            out(f"backup -> {bak}")
        _save_state(patch, target, state)              # before the files: a crash leaves a record, never a surprise
        for i, (path, data) in zip(todo, prepared):
            write_atomic(path, data)
            written.append((path, i.raw))
    except BaseException as exc:                       # all or nothing: put back whatever was written
        problems = []
        for path, raw in reversed(written):
            try:
                write_atomic(path, raw)
            except Exception as exc2:                  # noqa: BLE001
                problems.append(f"{path}: {exc2}")
        for bak in made_backups:
            try:
                os.unlink(bak)
            except OSError:
                pass
        try:
            _restore_state_file(patch, target, previous_state)
        except OSError as exc2:
            problems.append(f"{state_file}: {exc2}")
        if problems:
            out(f"{patch.name}: COULD NOT PUT BACK: {'; '.join(problems)}. Restore from the .bb-orig-{patch.name} backups by hand.")
        if not isinstance(exc, Exception):
            raise
        out(f"{patch.name}: REFUSING: writing failed ({exc}); every file already written was put back, nothing changed")
        return REFUSED
    for i in todo:
        out(f"{i.edit.rel}: patched ({'CRLF' if i.crlf else 'LF'} kept). Takes effect when the program next starts.")
    return APPLIED


def _legacy_backups(path):
    """Old-script backups of `path`, newest first: <file>.bak-*, <file>.orig-* (never this kit's own .bb-orig-*)."""
    import glob
    found = [p for pat in (path + ".bak-*", path + ".orig-*", path + ".bak") for p in glob.glob(pat)
             if ".bb-orig-" not in os.path.basename(p) and os.path.isfile(p)]
    return sorted(set(found), key=lambda p: (os.path.getmtime(p), p), reverse=True)


def do_adopt(patch, target, flags, out=print) -> int:
    state_file = _state_path(patch, target)
    previous_state = open(state_file, "rb").read() if os.path.exists(state_file) else None
    state = load_state(patch, target) or {"patch": patch.name, "files": {}}
    plan, problems = [], []
    for e in selected_edits(patch, flags):
        if e.rel in state["files"]:
            continue
        i = inspect_edit(e, target)
        if i.state != "applied":
            why = "drifted: " + i.why if i.state == "drifted" else "nothing to adopt; use --apply"
            problems.append(f"{e.rel}: not patched ({why})")
            continue
        path = _path(target, e.rel)
        bak = path + patch.backup_suffix
        chosen = None
        for cand in _legacy_backups(path):
            try:
                raw, text, _ = read_file(cand)
            except (OSError, UnicodeDecodeError):
                continue
            if e.marker in text:
                continue                                     # a backup that already has this patch is not the "before"
            try:
                if e.needs is not None:
                    e.needs(text)
                if e.apply(text) == i.text:
                    chosen = (cand, raw)
                    break
            except ValueError:
                continue
        if chosen is None:
            problems.append(f"{e.rel}: no old backup (<file>.bak-*, <file>.orig-*) reproduces the file when this patch is applied to "
                            f"it. If another patch sits on top of it, revert that one first. Restore by hand otherwise.")
            continue
        if os.path.exists(bak) and sha256(open(bak, "rb").read()) != sha256(chosen[1]):
            problems.append(f"{e.rel}: {bak} exists and is a different file; look at it first")
            continue
        plan.append((e.rel, path, bak, chosen[0], chosen[1], i.raw))
    if problems:
        for p in problems:
            out(f"REFUSING: {p}")
        out(f"{patch.name}: refused, nothing was written")
        return REFUSED
    if not plan:
        out(f"{patch.name}: already recorded; nothing to adopt")
        return APPLIED
    made = []
    try:
        for rel, path, bak, cand, raw_before, raw_now in plan:
            existed = os.path.exists(bak)
            shutil.copy2(cand, bak)
            if not existed:
                made.append(bak)
            state["files"][rel] = {"before_sha256": sha256(raw_before), "after_sha256": sha256(raw_now),
                                   "backup": os.path.basename(bak), "adopted_from": os.path.basename(cand),
                                   "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save_state(patch, target, state)
    except BaseException as exc:
        for bak in made:
            try:
                os.unlink(bak)
            except OSError:
                pass
        try:
            _restore_state_file(patch, target, previous_state)
        except OSError:
            pass
        if not isinstance(exc, Exception):
            raise
        out(f"{patch.name}: REFUSING: adopting failed ({exc}); nothing changed")
        return REFUSED
    for rel, path, bak, cand, raw_before, raw_now in plan:
        out(f"{rel}: adopted -- {os.path.basename(cand)} reproduces it under this patch; --revert now restores that file")
    return APPLIED


def do_revert(patch, target, out=print) -> int:
    state = load_state(patch, target)
    if state is None:
        edits = [inspect_edit(e, target) for e in patch.edits]
        if all(i.state != "applied" for i in edits):
            out(f"{patch.name}: not applied; nothing to revert")
            return NOTHING_TO_REVERT
        out(f"{patch.name}: REFUSING: the files are patched but there is no {patch.state_name} record here (applied by an "
            f"earlier copy of this script?). Restore its .bak-* / .orig-* backup by hand.")
        return REFUSED
    problems, plan = [], []
    for rel, rec in state["files"].items():
        path = _path(target, rel)
        bak = os.path.join(os.path.dirname(path), rec.get("backup") or "")
        try:
            now = sha256(open(path, "rb").read())
        except OSError as exc:
            problems.append(f"{rel}: cannot read it ({exc.strerror or exc})")
            continue
        if now != rec["after_sha256"]:
            problems.append(f"{rel}: changed since the apply (its sha256 is not the one recorded). If another patch went on "
                            f"top of this one, revert that one first.")
            continue
        try:
            data = open(bak, "rb").read()
        except OSError:
            problems.append(f"{rel}: the backup {bak} is gone")
            continue
        if sha256(data) != rec["before_sha256"]:
            problems.append(f"{rel}: the backup {bak} is not the file this patch backed up (sha256 differs)")
            continue
        plan.append((rel, path, bak, data))
    if problems:
        for p in problems:
            out(f"REFUSING: {p}")
        out(f"{patch.name}: refused, nothing was changed")
        return REFUSED
    for rel, path, bak, data in plan:
        write_atomic(path, data, like=bak)
        os.unlink(bak)
        del state["files"][rel]
        out(f"{rel}: restored from {os.path.basename(bak)}")
    _save_state(patch, target, state)
    out(f"{patch.name}: reverted. Takes effect when the program next starts.")
    return APPLIED


# ── the command line ──────────────────────────────────────────────────────────────────────────────────────────────

def build_parser(patch: Patch, prog=None):
    ap = argparse.ArgumentParser(prog=prog, description=patch.summary,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="exit codes: --check 0 applied / 1 not applied / 2 anchors drifted; "
                                        "--apply 0 done / 2 refused; --revert 0 done / 1 nothing to revert / 2 refused; --adopt 0 done / 2 refused")
    ap.add_argument("--target", required=True, help="the directory the patch edits (see the script's docstring)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="(default) report the state of each file; change nothing")
    mode.add_argument("--apply", action="store_true", help="apply the patch (backs up first)")
    mode.add_argument("--revert", action="store_true", help="restore the backup this patch made")
    mode.add_argument("--adopt", action="store_true",
                      help="record a host the OLD patch script patched, so --revert works (writes only a copy of the old backup and a record)")
    ap.add_argument("--diff", action="store_true", help="with --check: also show what --apply would change")
    for flag, helptext in patch.options.items():
        ap.add_argument(f"--{flag}", action="store_true", help=helptext)
    return ap


def main(patch: Patch, argv=None, out=print) -> int:
    ap = build_parser(patch)
    a = ap.parse_args(argv)
    flags = {flag for flag in patch.options if getattr(a, flag.replace("-", "_"), False)}
    target = patch.resolve(os.path.expanduser(a.target))
    if not os.path.isdir(target):
        out(f"{patch.name}: {target} is not a directory")
        return DRIFTED if not (a.apply or a.revert or a.adopt) else REFUSED
    if a.apply:
        return do_apply(patch, target, flags, out)
    if a.adopt:
        return do_adopt(patch, target, flags, out)
    if a.revert:
        return do_revert(patch, target, out)
    return do_check(patch, target, flags, show_diff=a.diff, out=out)
