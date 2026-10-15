# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/patchfile.py: --check / --apply / --revert for a plain .patch file, driven through real `git apply` on a scratch tree.
Skipped where git is not installed."""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "patches"))

import patchfile  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

PATCH = """diff --git a/common/auth.py b/common/auth.py
index 1111111..2222222 100644
--- a/common/auth.py
+++ b/common/auth.py
@@ -1,6 +1,6 @@
 def load():
     logger.info(
-        f"Your API key is: {key}\\n"
+        f"Your API key is: <redacted>\\n"
         "If these keys get compromised"
     )
     return 1
"""
ORIGINAL = 'def load():\n    logger.info(\n        f"Your API key is: {key}\\n"\n        "If these keys get compromised"\n    )\n    return 1\n'


@pytest.fixture
def tree(tmp_path, monkeypatch):
    # git apply honours core.autocrlf (a Windows git is configured with it on): pin the config so the result is the same everywhere
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    t = tmp_path / "tabby"
    (t / "common").mkdir(parents=True)
    (t / "common" / "auth.py").write_bytes(ORIGINAL.encode())
    pf = tmp_path / "auth.patch"
    pf.write_bytes(PATCH.encode())
    return t, pf


def run(pf, t, *flags):
    lines = []
    rc = patchfile.main([str(pf), "--target", str(t), *flags], out=lines.append)
    return rc, "\n".join(lines)


def test_the_round_trip(tree):
    t, pf = tree
    assert run(pf, t)[0] == 1                                                    # --check: not applied
    assert run(pf, t, "--apply")[0] == 0
    assert b"<redacted>" in (t / "common" / "auth.py").read_bytes()
    assert run(pf, t)[0] == 0
    rc, out = run(pf, t, "--apply")
    assert rc == 0 and "already applied" in out
    assert run(pf, t, "--revert")[0] == 0
    assert (t / "common" / "auth.py").read_bytes() == ORIGINAL.encode()
    assert run(pf, t, "--revert")[0] == 1                                        # nothing to revert


def test_a_drifted_file_is_exit_2_and_untouched(tree):
    t, pf = tree
    changed = ORIGINAL.replace("Your API key is", "Your key is").encode()
    (t / "common" / "auth.py").write_bytes(changed)
    assert run(pf, t)[0] == 2
    rc, out = run(pf, t, "--apply")
    assert rc == 2 and "REFUSING" in out and (t / "common" / "auth.py").read_bytes() == changed


def test_revert_refuses_when_the_patched_lines_were_edited_since(tree):
    t, pf = tree
    run(pf, t, "--apply")
    edited = (t / "common" / "auth.py").read_bytes().replace(b"<redacted>", b"<hidden>")
    (t / "common" / "auth.py").write_bytes(edited)
    rc, out = run(pf, t, "--revert")
    assert rc == 2 and "changed since the apply" in out
    assert (t / "common" / "auth.py").read_bytes() == edited


def test_crlf_working_copies_are_patched_as_git_does(tree):
    """A Windows checkout (core.autocrlf) holds CRLF; the patch is LF. git apply handles it only inside a repo that has the
    setting, so this test builds one."""
    t, pf = tree
    subprocess.run(["git", "init", "-q"], cwd=t, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=t, check=True)
    (t / "common" / "auth.py").write_bytes(ORIGINAL.replace("\n", "\r\n").encode())
    assert run(pf, t)[0] in (1, 2)          # not applied (1); a git without the conversion would say drifted (2): either way, no crash
    if run(pf, t)[0] == 1:
        assert run(pf, t, "--apply")[0] == 0
        assert run(pf, t)[0] == 0


def test_the_shipped_patch_files_are_lf_and_start_with_a_git_header():
    for name in ("auth_redact.patch", "windows_launcher.patch"):
        raw = open(os.path.join(ROOT, "patches", "tabbyapi", name), "rb").read()
        assert raw.startswith(b"diff --git a/"), name
        assert b"\r" not in raw, name
    auth = open(os.path.join(ROOT, "patches", "tabbyapi", "auth_redact.patch"), "rb").read()
    assert b"common/auth.py" in auth and b"main.py" not in auth
    launcher = open(os.path.join(ROOT, "patches", "tabbyapi", "windows_launcher.patch"), "rb").read()
    assert b"main.py" in launcher and b"start.bat" in launcher and b"common/auth.py" not in launcher
