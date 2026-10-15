# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""deploy/: release.py builds an immutable release dir from a commit, verify_stage.py says whether a stage dir or a release dir is
that commit, run_node.sh starts a node from its stage dir, and the unit never restarts one. Everything here runs on local
directories; the ssh paths are the same code with `ssh host` in front and are not exercised here."""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "deploy"))

import release  # noqa: E402  (deploy/ is repo-only tooling; release.py makes src/ importable only when `bigbrain` is not installed)
import verify_stage  # noqa: E402

HAS_GIT_REPO = (ROOT / ".git").exists() and shutil.which("git") is not None
needs_repo = pytest.mark.skipif(not HAS_GIT_REPO, reason="needs this repo's .git (a release dir has none)")


def _is_readonly_posix(p: Path) -> bool:
    return not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


@needs_repo
def test_release_installs_the_commit_read_only_with_the_full_hash_in_build(tmp_path):
    full = release.resolve_commit("HEAD")
    assert release.main(["--commit", "HEAD", "--to-dir", str(tmp_path)]) == 0
    rel = tmp_path / release.short_of(full)
    assert (rel / "BUILD").read_text(encoding="ascii") == full + "\n"
    assert (rel / "src" / "bigbrain" / "worker.py").is_file() and (rel / "pyproject.toml").is_file()
    assert not (rel / "tools").exists()                                        # the pre-0.2 layout is gone from a release too
    assert _is_readonly_posix(rel / "BUILD") and _is_readonly_posix(rel / "src" / "bigbrain" / "pipe.py")
    assert not list(tmp_path.glob(".*.partial")), "a finished release leaves no .partial dir"
    link = tmp_path / "current"
    assert (link.is_symlink() and os.readlink(link) == release.short_of(full)) or (tmp_path / "current.txt").exists()
    # it is byte-for-byte the commit: every tracked file hashes the same
    assert verify_stage.main(["--commit", full, "--release", str(rel)]) == 0
    _writable(tmp_path)


@needs_repo
def test_a_second_release_of_the_same_commit_changes_nothing(tmp_path, capsys):
    release.main(["--commit", "HEAD", "--to-dir", str(tmp_path)])
    capsys.readouterr()
    assert release.main(["--commit", "HEAD", "--to-dir", str(tmp_path)]) == 0
    assert "EXISTS" in capsys.readouterr().out
    _writable(tmp_path)


@needs_repo
def test_a_dry_run_touches_nothing(tmp_path, capsys):
    assert release.main(["--commit", "HEAD", "--host", "bigbrain@10.0.0.11", "--dry-run"]) == 0
    assert "would extract" in capsys.readouterr().out
    assert release.main(["--commit", "HEAD", "--to-dir", str(tmp_path / "x"), "--dry-run"]) == 0
    assert not (tmp_path / "x").exists()


def test_release_refuses_an_unknown_commit_and_needs_a_target():
    with pytest.raises(SystemExit):
        release.resolve_commit("no-such-tag-xyz")
    with pytest.raises(SystemExit):
        release.main(["--commit", "HEAD"])


def test_the_remote_script_is_atomic_idempotent_and_read_only():
    full = "ab" * 20
    s = release.remote_script("~/bigbrain/releases", full)
    short = full[:12]
    assert f'"$HOME"/bigbrain/releases/.{short}.partial' in s and "mv " in s and s.index("tar -x") < s.index("mv ")
    assert s.index("chmod -R a-w") < s.index("mv "), "read-only before it is renamed into place"
    assert "if [ -e " in s and "echo EXISTS" in s and "echo INSTALLED" in s
    assert f"'{full}'" in s or full in s
    assert "BUILD" in s and "ln -sfn" in s
    assert "sudo" not in s and "systemctl" not in s and "kill" not in s


# -- verify_stage ----------------------------------------------------------------------------------------------------
def _make_stage(tmp_path, crlf=False):
    stage = tmp_path / "stage"
    for rel in verify_stage.package_files():
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / "src" / rel).read_bytes()                               # staged as bigbrain/x.py, tracked as src/bigbrain/x.py
        dst.write_bytes(data.replace(b"\n", b"\r\n") if crlf and data else data)
    return stage


@needs_repo
def test_a_stage_dir_that_is_the_commit_verifies(tmp_path, capsys):
    stage = _make_stage(tmp_path)
    assert verify_stage.main(["--commit", "HEAD", "--stage", str(stage)]) == 0
    out = capsys.readouterr().out
    assert "IDENTICAL " in out and "all identical" in out and "DIFFERENT" not in out
    assert verify_stage.main(["--commit", "HEAD", "--stage", str(stage), "--strict"]) == 0


@needs_repo
def test_a_crlf_stage_dir_is_identical_after_lf_but_fails_strict(tmp_path, capsys):
    """How a stage dir reads when it was staged from a CRLF Windows checkout: the same code once LF-normalised."""
    stage = _make_stage(tmp_path, crlf=True)
    assert verify_stage.main(["--commit", "HEAD", "--stage", str(stage)]) == 0
    assert "IDENTICAL-LF" in capsys.readouterr().out
    assert verify_stage.main(["--commit", "HEAD", "--stage", str(stage), "--strict"]) == 1


@needs_repo
def test_a_changed_or_missing_file_is_reported(tmp_path, capsys):
    stage = _make_stage(tmp_path)
    (stage / "bigbrain" / "pipe.py").write_bytes(b"# edited\n")
    (stage / "bigbrain" / "draft.py").unlink()
    assert verify_stage.main(["--commit", "HEAD", "--stage", str(stage)]) == 1
    out = capsys.readouterr().out
    assert "DIFFERENT     bigbrain/pipe.py" in out and "MISSING       bigbrain/draft.py" in out


@needs_repo
def test_extra_files_in_a_stage_dir_are_not_an_error(tmp_path):
    stage = _make_stage(tmp_path)
    (stage / "node_0.log").write_text("log\n")
    (stage / "api_key").write_text("secret\n")
    (stage / "launcher.py").write_text("# harness\n")
    assert verify_stage.main(["--commit", "HEAD", "--stage", str(stage)]) == 0


def test_compare_words():
    import hashlib
    body = b"a\nb\n"
    h = lambda b: hashlib.sha256(b).hexdigest()           # noqa: E731
    expected = {"same": body, "crlf": body, "diff": body, "gone": body}
    found = {"same": [h(body), h(body)], "crlf": [h(b"a\r\nb\r\n"), h(body)], "diff": [h(b"x"), h(b"x")], "gone": None}
    assert verify_stage.compare(expected, found) == {"same": "IDENTICAL", "crlf": "IDENTICAL-LF", "diff": "DIFFERENT", "gone": "MISSING"}


def test_the_stage_file_list_comes_from_the_controller():
    files = verify_stage.package_files()
    assert len(files) == 8 and all(f.startswith("bigbrain/") for f in files)
    assert {Path(f).name for f in files} == {"__init__.py", "pipe.py", "worker.py", "server.py", "engine.py", "paged.py",
                                             "spec.py", "draft.py"}


# -- run_node.sh and the unit ------------------------------------------------------------------------------------------
SH = shutil.which("sh")


@pytest.mark.skipif(SH is None, reason="no sh on this machine")
def test_run_node_sh_parses_and_checks_its_inputs(tmp_path):
    script = ROOT / "deploy" / "run_node.sh"
    assert subprocess.run([SH, "-n", str(script)], capture_output=True).returncode == 0
    no_args = subprocess.run([SH, str(script), str(tmp_path)], capture_output=True, text=True)
    assert no_args.returncode == 66 and "node.args is missing" in no_args.stderr
    (tmp_path / "node.args").write_text("--rank 0\n")
    no_py = subprocess.run([SH, str(script), str(tmp_path)], capture_output=True, text=True, env={**os.environ, "BB_PYTHON": ""})
    assert no_py.returncode != 0 and "BB_PYTHON" in no_py.stderr
    # v0.1.1: no `server` (or any other) mode -- one argument, the stage dir
    for extra in ("server", "worker", "evil"):
        too_many = subprocess.run([SH, str(script), str(tmp_path), extra], capture_output=True, text=True)
        assert too_many.returncode == 64 and "usage: run_node.sh <stage-dir>" in too_many.stderr, extra


def test_run_node_sh_has_no_server_mode_and_always_starts_the_worker_module():
    text = (ROOT / "deploy" / "run_node.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "bigbrain.worker" in code and "bigbrain.server" not in code and "worker|server" not in text
    assert "mod=" not in code and "$mod" not in code
    unit = (ROOT / "deploy" / "bigbrain-node@.service").read_text(encoding="utf-8")
    assert "run_node.sh /%I worker" not in unit and "run_node.sh /%I server" not in unit


@pytest.mark.skipif(SH is None, reason="no sh on this machine")
def test_run_node_sh_runs_the_module_from_the_stage_dir_with_its_args(tmp_path):
    """cwd = the stage dir (the controller's teardown matches on it) and the argv is node.args."""
    stage = tmp_path / "stage"
    (stage / "bigbrain").mkdir(parents=True)
    (stage / "bigbrain" / "__init__.py").write_text("")
    (stage / "bigbrain" / "worker.py").write_text(
        "import os, sys\nprint('cwd=' + os.getcwd())\nprint('argv=' + ' '.join(sys.argv[1:]))\n")
    (stage / "node.args").write_text("--rank 3 --total 4\n")
    (stage / "node.env").write_text(f'BB_PYTHON="{sys.executable.replace(chr(92), "/")}"\n')
    p = subprocess.run([SH, str(ROOT / "deploy" / "run_node.sh"), str(stage)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "argv=--rank 3 --total 4" in p.stdout
    cwd_line = [line for line in p.stdout.splitlines() if line.startswith("cwd=")][0][4:]
    assert Path(cwd_line).resolve() == stage.resolve()


def test_the_unit_never_restarts_and_installs_nowhere():
    text = (ROOT / "deploy" / "bigbrain-node@.service").read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "Restart=no" in body and "KillMode=control-group" in body and "TimeoutStopSec=20" in body
    assert "[Install]" not in body and "WantedBy" not in body
    assert "Restart=always" not in body and "Restart=on-failure" not in body
    assert "WorkingDirectory=/%I" in body and "run_node.sh /%I" in body


def _writable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        try:
            os.chmod(p, p.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass
