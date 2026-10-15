# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Rules about the repository itself: the one top-level package is src/bigbrain, no secret-bearing files, LF everywhere,
and every file the controller stages to a node is really in the tree."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules", "build", "dist", ".eggs"}

#: files the repo must never hold (.gitignore + this test)
FORBIDDEN_NAMES = {"api_key", "endpoint.env", "node.env", "api_tokens.yml", "config.yml", ".env"}
CRLF_OK = {".bat", ".cmd", ".ps1", ".patch", ".diff"}
GENERATED_BY_SDIST = {"PKG-INFO", "setup.cfg"}        # written by setuptools into an sdist, with the build machine's line endings

#: patches/ is repo-only (not in the wheel, left out of the sdist: other licences), so what is said about it is checked only where it exists
needs_patches = pytest.mark.skipif(not (ROOT / "patches").is_dir(), reason="patches/ is not in an sdist (repo-only tooling)")


def _files():
    for p in ROOT.rglob("*"):
        parts = p.relative_to(ROOT).parts
        if p.is_file() and not (SKIP_DIRS & set(parts)) and not any(x.endswith(".egg-info") for x in parts):
            yield p


def test_the_only_top_level_package_is_bigbrain():
    """v0.2.0: a src layout with ONE importable top-level name, `bigbrain`. The old `tools` namespace package (which would have
    collided with any other project's `tools`) is gone, and nothing in the repo may bring it back or add a second package."""
    assert not (ROOT / "tools").exists()
    assert (ROOT / "src" / "bigbrain" / "__init__.py").is_file()
    assert not (ROOT / "src" / "__init__.py").exists()
    assert [p.name for p in (ROOT / "src").iterdir() if p.is_dir() and p.name != "__pycache__"
            and not p.name.endswith(".egg-info")] == ["bigbrain"]


def test_no_secret_bearing_file_is_in_the_tree():
    bad = [str(p.relative_to(ROOT)) for p in _files()
           if p.name in FORBIDDEN_NAMES or p.name.endswith(".env") or (p.name == "config.json" and p.parent == ROOT)]
    assert not bad, bad


# split over two lines so this file does not itself look like it holds a key header
_PRIVATE_KEY_PEM = (r"-----BEGIN [A-Z ]*"
                    r"PRIVATE" r" KEY-----")


def test_no_key_shaped_string_is_in_a_text_file():
    pats = [re.compile(r"\bsk-[A-Za-z0-9]{20,}"), re.compile(r"\bghp_[A-Za-z0-9]{30,}"),
            re.compile(r"\bhf_[A-Za-z0-9]{30,}"), re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            re.compile(_PRIVATE_KEY_PEM),
            re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{24,}")]
    hits = []
    for p in _files():
        if p.suffix in {".png", ".safetensors", ".pyc"}:
            continue
        text = p.read_bytes().decode("utf-8", "replace")
        for rx in pats:
            if rx.search(text):
                hits.append((str(p.relative_to(ROOT)), rx.pattern))
    assert not hits, hits


def test_every_text_file_is_lf_except_windows_scripts_and_patches():
    crlf = [str(p.relative_to(ROOT)) for p in _files()
            if p.suffix not in CRLF_OK and p.name not in GENERATED_BY_SDIST and b"\r\n" in p.read_bytes()]
    assert not crlf, crlf


def test_every_file_the_controller_stages_is_in_the_repo():
    from bigbrain.controller import PACKAGE_FILES
    missing = [f for f in PACKAGE_FILES if not (ROOT / "src" / f).exists()]       # staged as bigbrain/x.py, tracked as src/bigbrain/x.py
    assert not missing, missing
    assert all(f.startswith("bigbrain/") for f in PACKAGE_FILES) and len(PACKAGE_FILES) == 8


def test_the_licence_stance_is_in_the_tree():
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert lic.startswith("MIT License") and "Copyright (c) 2026 dadbodnerd" in lic


@needs_patches
def test_the_patch_licences_are_in_the_tree():
    assert (ROOT / "patches" / "tabbyapi" / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("GNU AFFERO GENERAL PUBLIC LICENSE")
    notice = (ROOT / "patches" / "NOTICE.md").read_text(encoding="utf-8")
    assert "Copyright (c) 2025 Turboderp" in notice and "AGPL-3.0" in notice and "MIT License" in notice


def test_no_upstream_source_is_vendored():
    """Only scripts and pins: no TabbyAPI or exllamav3 file, and none of their top-level directories."""
    for d in ("exllamav3", "common", "backends", "endpoints", "venv"):
        assert not (ROOT / d).exists(), d
    for p in _files():
        if p.suffix == ".py" and "patches" not in p.relative_to(ROOT).parts and "tests" not in p.relative_to(ROOT).parts:
            text = p.read_text(encoding="utf-8")
            assert "class ExllamaV3Container" not in text, p


# -- licence headers (v0.1.1) ----------------------------------------------------------------------------------------------------
@needs_patches
def test_every_tabbyapi_patch_file_carries_the_agpl_spdx_tag():
    d = ROOT / "patches" / "tabbyapi"
    seen = 0
    for p in sorted(d.iterdir()):
        if p.name in ("LICENSE",) or p.name.endswith(".license"):
            continue
        seen += 1
        if p.suffix == ".py":
            assert p.read_text(encoding="utf-8").splitlines()[0] == "# SPDX-License-Identifier: AGPL-3.0-or-later", p.name
        elif p.suffix == ".patch":
            side = d / (p.name + ".license")                 # the diff itself stays byte-identical to what the hosts carry
            assert side.is_file() and "SPDX-License-Identifier: AGPL-3.0-or-later" in side.read_text(encoding="utf-8"), p.name
            assert p.read_bytes().startswith(b"diff --git a/"), p.name
        else:
            raise AssertionError(f"unexpected file in patches/tabbyapi: {p.name}")
    assert seen == 5                                           # failed_load close_stc lora auth_redact.patch windows_launcher.patch


@needs_patches
def test_every_exllamav3_patch_file_carries_the_mit_tag_and_the_turboderp_notice():
    files = sorted((ROOT / "patches" / "exllamav3").glob("*.py"))
    assert files
    for p in files:
        head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:5])
        assert head.startswith("# SPDX-License-Identifier: MIT"), p.name
        assert "Copyright (c) 2025 Turboderp" in head and "patches/NOTICE.md" in head, p.name


@needs_patches
def test_an_spdx_comment_does_not_displace_the_module_docstring():
    import ast
    for p in list((ROOT / "patches" / "tabbyapi").glob("*.py")) + list((ROOT / "patches" / "exllamav3").glob("*.py")):
        assert ast.get_docstring(ast.parse(p.read_text(encoding="utf-8"))), p.name


def test_every_first_party_source_file_carries_the_mit_tag():
    """.py / .sh / .service files outside the two upstream-derived subtrees are BigBrain's own: MIT, the author's copyright."""
    bad = []
    for p in _files():
        rel = p.relative_to(ROOT).as_posix()
        if p.suffix not in {".py", ".sh", ".service"} or rel.startswith(("patches/tabbyapi/", "patches/exllamav3/")):
            continue
        head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:4])
        if "SPDX-License-Identifier: MIT" not in head or "Copyright (c) 2026 dadbodnerd" not in head:
            bad.append(rel)
    assert not bad, bad


def test_no_site_specific_value_of_the_authors_network_is_in_the_tree():
    """The examples use 10.0.0.x, the user "bigbrain" and generic paths. These patterns are assembled from pieces so this file
    does not contain what it forbids."""
    forbidden = [re.compile("192" + r"\.168\."), re.compile("/home/" + "dadbod"), re.compile("/srv/" + "aio"),
                 re.compile("AIO" + "rchestrator"), re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-"),
                 re.compile("T" + "340"), re.compile("nob" + "ara", re.I), re.compile("portable-" + "desktop")]
    hits = []
    for p in _files():
        if p.suffix in {".png", ".safetensors", ".pyc"}:
            continue
        text = p.read_bytes().decode("utf-8", "replace")
        for rx in forbidden:
            if rx.search(text):
                hits.append((p.relative_to(ROOT).as_posix(), rx.pattern))
    assert not hits, hits
