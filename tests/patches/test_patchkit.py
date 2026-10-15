# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""patches/_patchkit.py: the one command line and the one set of exit codes every patch script shares, shown with a toy patch
so the rules are tested apart from any real upstream file.

    --check   0 applied, 1 not applied (or partly), 2 anchors drifted
    --apply   0 done or already, 2 refused (nothing written)
    --revert  0 done, 1 nothing to revert, 2 refused (nothing changed)
"""
import hashlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "patches"))

import _patchkit as pk  # noqa: E402

MARK = "TOYMARK"


def _add_line(text):
    if text.count("ANCHOR\n") != 1:
        raise ValueError(f"anchor matched {text.count('ANCHOR' + chr(10))} times, expected 1")
    return text.replace("ANCHOR\n", f"ANCHOR\nx = 1  # {MARK}\n", 1)


def _second(text):
    if text.count("SECOND\n") != 1:
        raise ValueError("second anchor missing")
    return text.replace("SECOND\n", f"SECOND\ny = 2  # {MARK}\n", 1)


def toy(two=False, name="toy"):
    edits = [pk.Edit("a.py", MARK, _add_line)]
    if two:
        edits.append(pk.Edit("sub/b.py", MARK, _second))
    return pk.Patch(name, "a toy patch", edits)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.py").write_text("ANCHOR\n", encoding="utf-8", newline="\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("SECOND\n", encoding="utf-8", newline="\n")
    return tmp_path


def run(patch, tree, *flags):
    lines = []
    rc = pk.main(patch, ["--target", str(tree), *flags], out=lines.append)
    return rc, "\n".join(lines)


def test_check_is_the_default_and_changes_nothing(tree):
    before = sha(tree / "a.py")
    rc, out = run(toy(), tree)
    assert rc == pk.NOT_APPLIED == 1 and "NOT patched" in out and "not applied" in out
    assert sha(tree / "a.py") == before and sorted(os.listdir(tree)) == ["a.py", "sub"]


def test_apply_check_revert_round_trip(tree):
    original = (tree / "a.py").read_bytes()
    assert run(toy(), tree, "--apply")[0] == 0
    assert (tree / "a.py").read_bytes() == b"ANCHOR\nx = 1  # TOYMARK\n"
    assert run(toy(), tree, "--check")[0] == 0
    assert (tree / "a.py.bb-orig-toy").read_bytes() == original            # the patch's own backup
    assert (tree / ".bb-patch-toy.json").exists()
    assert run(toy(), tree, "--revert")[0] == 0
    assert (tree / "a.py").read_bytes() == original
    assert not (tree / "a.py.bb-orig-toy").exists() and not (tree / ".bb-patch-toy.json").exists()
    assert run(toy(), tree, "--check")[0] == 1


def test_apply_twice_is_a_no_op(tree):
    assert run(toy(), tree, "--apply")[0] == 0
    once = sha(tree / "a.py")
    rc, out = run(toy(), tree, "--apply")
    assert rc == 0 and "already applied" in out and sha(tree / "a.py") == once


def test_apply_is_all_or_nothing_when_one_file_drifted(tree):
    (tree / "sub" / "b.py").write_text("nothing to anchor on\n", encoding="utf-8")
    before = (sha(tree / "a.py"), sha(tree / "sub" / "b.py"))
    rc, out = run(toy(two=True), tree, "--apply")
    assert rc == 2 and "REFUSING" in out
    assert (sha(tree / "a.py"), sha(tree / "sub" / "b.py")) == before
    assert not (tree / ".bb-patch-toy.json").exists() and not (tree / "a.py.bb-orig-toy").exists()


def test_check_says_drifted_with_exit_2(tree):
    (tree / "a.py").write_text("ANCHOR\nANCHOR\n", encoding="utf-8")                  # matches twice
    rc, out = run(toy(), tree)
    assert rc == 2 and "DRIFTED" in out and "2 times" in out


def test_a_missing_file_is_drift_not_a_crash(tree):
    os.unlink(tree / "sub" / "b.py")
    assert run(toy(two=True), tree)[0] == 2
    assert run(toy(two=True), tree, "--apply")[0] == 2


def test_partly_applied_is_not_applied_and_apply_finishes_it(tree):
    assert run(toy(), tree, "--apply")[0] == 0                                            # only a.py patched
    rc, out = run(toy(two=True), tree)
    assert rc == 1 and "PARTIAL" in out
    assert run(toy(two=True), tree, "--apply")[0] == 0
    assert run(toy(two=True), tree)[0] == 0
    assert run(toy(two=True), tree, "--revert")[0] == 0
    assert (tree / "a.py").read_text() == "ANCHOR\n" and (tree / "sub" / "b.py").read_text() == "SECOND\n"


def test_revert_refuses_when_the_file_changed_after_the_apply(tree):
    run(toy(), tree, "--apply")
    edited = (tree / "a.py").read_bytes() + b"z = 3\n"
    (tree / "a.py").write_bytes(edited)
    rc, out = run(toy(), tree, "--revert")
    assert rc == 2 and "changed since the apply" in out
    assert (tree / "a.py").read_bytes() == edited and (tree / "a.py.bb-orig-toy").exists()


def test_revert_refuses_when_the_backup_is_not_what_was_backed_up(tree):
    run(toy(), tree, "--apply")
    (tree / "a.py.bb-orig-toy").write_text("tampered\n", encoding="utf-8")
    patched = (tree / "a.py").read_bytes()
    rc, out = run(toy(), tree, "--revert")
    assert rc == 2 and "not the file this patch backed up" in out
    assert (tree / "a.py").read_bytes() == patched


def test_revert_of_a_file_patched_by_the_old_scripts_is_refused_with_a_hint(tree):
    (tree / "a.py").write_text("ANCHOR\nx = 1  # TOYMARK\n", encoding="utf-8")            # marker present, no state record
    rc, out = run(toy(), tree, "--revert")
    assert rc == 2 and "no .bb-patch-toy.json record" in out and "by hand" in out
    assert run(toy(), tree, "--check")[0] == 0                                             # but it still READS as applied


def test_revert_with_nothing_applied_is_exit_1(tree):
    rc, out = run(toy(), tree, "--revert")
    assert rc == 1 and "nothing to revert" in out


def test_a_leftover_backup_from_a_crashed_apply_is_reused_but_a_foreign_one_is_not(tree):
    (tree / "a.py.bb-orig-toy").write_bytes((tree / "a.py").read_bytes())               # same content: a crash leftover
    assert run(toy(), tree, "--apply")[0] == 0
    run(toy(), tree, "--revert")
    (tree / "a.py.bb-orig-toy").write_text("somebody else's backup\n", encoding="utf-8")
    rc, out = run(toy(), tree, "--apply")
    assert rc == 2 and "REFUSING" in out and (tree / "a.py").read_text() == "ANCHOR\n"


def test_crlf_is_kept_and_so_is_the_mode(tree):
    p = tree / "a.py"
    p.write_bytes(b"ANCHOR\r\n")
    os.chmod(p, 0o640)
    assert run(toy(), tree, "--apply")[0] == 0
    assert p.read_bytes() == b"ANCHOR\r\nx = 1  # TOYMARK\r\n"
    if os.name == "posix":
        assert (os.stat(p).st_mode & 0o777) == 0o640
    assert run(toy(), tree, "--revert")[0] == 0
    assert p.read_bytes() == b"ANCHOR\r\n"


def test_patched_python_that_does_not_compile_is_refused_before_anything_is_written(tree):
    def broken(text):
        return text + "def (:\n"
    patch = pk.Patch("broken", "x", [pk.Edit("a.py", "BROKENMARK", broken)])
    rc, out = run(patch, tree, "--apply")
    assert rc == 2 and (tree / "a.py").read_text() == "ANCHOR\n"
    assert not (tree / "a.py.bb-orig-broken").exists()


def test_a_file_that_is_not_utf8_is_drift_not_a_crash(tree):
    (tree / "a.py").write_bytes(b"\xff\xfeANCHOR\n")
    assert run(toy(), tree)[0] == 2


def test_an_optional_edit_is_only_selected_by_its_flag(tree):
    patch = pk.Patch("opt", "x", [pk.Edit("a.py", MARK, _add_line),
                                  pk.Edit("sub/b.py", MARK, _second, opt="second")],
                     options={"second": "also patch sub/b.py"})
    assert run(patch, tree, "--apply")[0] == 0
    assert (tree / "sub" / "b.py").read_text() == "SECOND\n"
    assert run(patch, tree, "--check")[0] == 0                                              # b.py is not part of the verdict
    assert run(patch, tree, "--check", "--second")[0] == 1
    assert run(patch, tree, "--apply", "--second")[0] == 0
    assert run(patch, tree, "--check", "--second")[0] == 0
    assert run(patch, tree, "--revert")[0] == 0                                             # one revert undoes everything it made
    assert (tree / "sub" / "b.py").read_text() == "SECOND\n" and (tree / "a.py").read_text() == "ANCHOR\n"


def test_diff_shows_the_change_and_still_writes_nothing(tree):
    rc, out = run(toy(), tree, "--check", "--diff")
    assert rc == 1 and "+x = 1  # TOYMARK" in out
    assert sorted(os.listdir(tree)) == ["a.py", "sub"]


def test_two_patches_on_one_file_must_be_reverted_in_reverse_order(tree):
    second = pk.Patch("second", "x", [pk.Edit("a.py", "SECONDMARK",
                                              lambda t: t.replace("ANCHOR\n", "ANCHOR\nq = 4  # SECONDMARK\n", 1))])
    run(toy(), tree, "--apply")
    run(second, tree, "--apply")
    assert run(toy(), tree, "--revert")[0] == 2                    # `second` went on top: its sha256 is what is on disk now
    assert run(second, tree, "--revert")[0] == 0
    assert run(toy(), tree, "--revert")[0] == 0
    assert (tree / "a.py").read_text() == "ANCHOR\n"


def test_target_that_is_not_a_directory_is_an_error_exit(tmp_path):
    assert run(toy(), tmp_path / "missing")[0] == 2
    assert run(toy(), tmp_path / "missing", "--apply")[0] == 2


def test_modes_are_mutually_exclusive(tree):
    with pytest.raises(SystemExit) as e:
        pk.main(toy(), ["--target", str(tree), "--apply", "--revert"], out=lambda s: None)
    assert e.value.code == 2


# -- --apply is all-or-nothing, also when a write fails part way -----------------------------------------------------------------
def _snapshot(tree):
    return {str(p.relative_to(tree)): p.read_bytes() for p in sorted(tree.rglob("*")) if p.is_file()}


def test_a_write_that_fails_part_way_puts_every_written_file_back(tree, monkeypatch):
    real = pk.write_atomic

    def flaky(path, data, like=None):
        if str(path).endswith("b.py") and b"TOYMARK" in data:
            raise OSError("disk full (test)")
        return real(path, data, like)

    before = _snapshot(tree)
    monkeypatch.setattr(pk, "write_atomic", flaky)
    rc, out = run(toy(two=True), tree, "--apply")
    assert rc == 2 and "put back" in out and "disk full (test)" in out
    assert _snapshot(tree) == before, "no file, backup or state record differs from before the apply"


def test_a_failed_second_apply_keeps_the_first_ones_record(tree, monkeypatch):
    assert run(toy(), tree, "--apply")[0] == 0                                 # a.py patched, recorded
    after_first = _snapshot(tree)
    real = pk.write_atomic

    def flaky(path, data, like=None):
        if str(path).endswith("b.py") and b"TOYMARK" in data:
            raise OSError("no space (test)")
        return real(path, data, like)
    monkeypatch.setattr(pk, "write_atomic", flaky)
    assert run(toy(two=True), tree, "--apply")[0] == 2
    assert _snapshot(tree) == after_first
    monkeypatch.setattr(pk, "write_atomic", real)
    assert run(toy(), tree, "--revert")[0] == 0                                  # and the first record still reverts it


def test_ctrl_c_during_the_writes_also_rolls_back_and_propagates(tree, monkeypatch):
    real = pk.write_atomic

    def interrupted(path, data, like=None):
        if str(path).endswith("b.py") and b"TOYMARK" in data:
            raise KeyboardInterrupt
        return real(path, data, like)

    before = _snapshot(tree)
    monkeypatch.setattr(pk, "write_atomic", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(toy(two=True), tree, "--apply")
    assert _snapshot(tree) == before


def test_a_failing_backup_leaves_nothing_behind(tree, monkeypatch):
    calls = {"n": 0}
    real = pk.shutil.copy2

    def bad_copy(src, dst, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("cannot copy (test)")
        return real(src, dst, **kw)

    before = _snapshot(tree)
    monkeypatch.setattr(pk.shutil, "copy2", bad_copy)
    rc, out = run(toy(two=True), tree, "--apply")
    assert rc == 2 and "writing failed" in out
    assert _snapshot(tree) == before


# -- --adopt: record a host the old script patched ---------------------------------------------------------------------------------
def _old_script_applied(tree, backup_name="a.py.bak-20260101-000000", text="ANCHOR\n"):
    """What an earlier copy of the script left: the patched file and a timestamped backup of the original."""
    (tree / backup_name).write_text(text, encoding="utf-8", newline="\n")
    (tree / "a.py").write_text(_add_line(text), encoding="utf-8", newline="\n")


def test_adopt_makes_an_old_patched_file_revertable(tree):
    _old_script_applied(tree)
    patched = (tree / "a.py").read_bytes()
    assert run(toy(), tree, "--revert")[0] == 2                                  # before: refused, no record
    rc, out = run(toy(), tree, "--adopt")
    assert rc == 0 and "adopted" in out and "a.py.bak-20260101-000000" in out
    assert (tree / "a.py").read_bytes() == patched, "adopt never touches the patched file"
    assert (tree / "a.py.bak-20260101-000000").exists(), "the old backup stays"
    assert (tree / "a.py.bb-orig-toy").read_bytes() == b"ANCHOR\n" and (tree / ".bb-patch-toy.json").exists()
    assert run(toy(), tree, "--revert")[0] == 0
    assert (tree / "a.py").read_bytes() == b"ANCHOR\n"


def test_adopt_picks_the_backup_that_reproduces_the_file_not_just_the_newest(tree):
    (tree / "a.py.bak-20250101").write_text("ANCHOR\n", encoding="utf-8")        # older, the right one
    (tree / "a.py").write_text(_add_line("ANCHOR\n"), encoding="utf-8")
    newer = tree / "a.py.bak-20260202"
    newer.write_text("ANCHOR\n# an unrelated later edit\n", encoding="utf-8")     # newer, but apply() of it is not the file
    os.utime(newer, (4102444800, 4102444800))
    rc, out = run(toy(), tree, "--adopt")
    assert rc == 0 and "a.py.bak-20250101" in out and "a.py.bak-20260202" not in out


def test_adopt_refuses_when_no_old_backup_reproduces_the_file(tree):
    (tree / "a.py").write_text("ANCHOR\nx = 1  # TOYMARK\n# plus a hand edit\n", encoding="utf-8")
    (tree / "a.py.bak-1").write_text("ANCHOR\n", encoding="utf-8")
    before = _snapshot(tree)
    rc, out = run(toy(), tree, "--adopt")
    assert rc == 2 and "no old backup" in out
    assert _snapshot(tree) == before


def test_adopt_refuses_a_file_that_is_not_patched_and_one_with_no_backup(tree):
    rc, out = run(toy(), tree, "--adopt")
    assert rc == 2 and "not patched" in out
    (tree / "a.py").write_text("ANCHOR\nx = 1  # TOYMARK\n", encoding="utf-8")
    rc, out = run(toy(), tree, "--adopt")
    assert rc == 2 and "no old backup" in out


def test_adopt_twice_and_after_an_apply_is_a_no_op(tree):
    _old_script_applied(tree)
    assert run(toy(), tree, "--adopt")[0] == 0
    before = _snapshot(tree)
    rc, out = run(toy(), tree, "--adopt")
    assert rc == 0 and "already recorded" in out and _snapshot(tree) == before


def test_adopt_also_finds_orig_style_backups(tree):
    (tree / "a.py.orig-dflash2tp").write_text("ANCHOR\n", encoding="utf-8")
    (tree / "a.py").write_text(_add_line("ANCHOR\n"), encoding="utf-8")
    rc, out = run(toy(), tree, "--adopt")
    assert rc == 0 and "a.py.orig-dflash2tp" in out


def test_adopt_keeps_a_crlf_backup_byte_for_byte(tree):
    (tree / "a.py.bak-1").write_bytes(b"ANCHOR\r\n")
    (tree / "a.py").write_bytes(b"ANCHOR\r\nx = 1  # TOYMARK\r\n")
    assert run(toy(), tree, "--adopt")[0] == 0
    assert run(toy(), tree, "--revert")[0] == 0
    assert (tree / "a.py").read_bytes() == b"ANCHOR\r\n"


def test_adopt_is_all_or_nothing_across_files(tree):
    _old_script_applied(tree)
    (tree / "sub" / "b.py").write_text("SECOND\ny = 2  # TOYMARK\n", encoding="utf-8")      # patched, but no backup of it
    before = _snapshot(tree)
    rc, out = run(toy(two=True), tree, "--adopt")
    assert rc == 2 and _snapshot(tree) == before


def test_adopt_and_apply_are_mutually_exclusive(tree):
    with pytest.raises(SystemExit) as e:
        pk.main(toy(), ["--target", str(tree), "--apply", "--adopt"], out=lambda s: None)
    assert e.value.code == 2
