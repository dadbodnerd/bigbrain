# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""The pip package: pyproject.toml says what the code really does. Every console script resolves to a callable that answers
--help without a GPU, the version is written once, and the declared dependencies are the imports the package makes (torch and
exllamav3 only lazily, and only in the optional `gpu` extra)."""
import ast
import importlib
import re
import sys
import tomllib
from pathlib import Path

import pytest

import bigbrain

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PKG_DIR = Path(bigbrain.__file__).resolve().parent               # the imported package: the installed one when there is one

#: the user-facing commands (README.md / INSTALL.md list the same names)
EXPECTED_SCRIPTS = {
    "bigbrain-controller": "bigbrain.controller:main",
    "bigbrain-registry": "bigbrain.build_registry:main",
    "bigbrain-merge-models": "bigbrain.merge_models:main",
    "bigbrain-node": "bigbrain.worker:main",
    "bigbrain-server": "bigbrain.server:main",
    "bigbrain-bench": "bigbrain.bench:main",
    "bigbrain-parity": "bigbrain.parity_check:main",
}


def _dist_names(requirements):
    return {re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0].lower().replace("_", "-") for r in requirements}


def test_the_version_is_written_once_and_looks_like_a_release():
    project = PYPROJECT["project"]
    assert "version" in project["dynamic"] and "version" not in project
    assert PYPROJECT["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "bigbrain.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+", bigbrain.__version__), bigbrain.__version__
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"v{bigbrain.__version__}" in readme, "README.md names the release it describes"


def test_the_console_scripts_are_the_documented_ones():
    assert PYPROJECT["project"]["scripts"] == EXPECTED_SCRIPTS
    for text in ("README.md", "INSTALL.md"):
        doc = (ROOT / text).read_text(encoding="utf-8")
        for name in EXPECTED_SCRIPTS:
            assert name in doc, f"{text} does not mention {name}"


@pytest.mark.parametrize("script,target", sorted(EXPECTED_SCRIPTS.items()))
def test_every_console_script_resolves_and_answers_help(script, target, capsys):
    module, _, attr = target.partition(":")
    main = getattr(importlib.import_module(module), attr)
    assert callable(main)
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code in (0, None), script
    out = capsys.readouterr().out
    assert "usage:" in out and "--help" in out, script


def _module_level_nodes(tree):
    """Imports that run when the module is imported: the top level, and inside top-level try/if/with blocks (not def/class bodies)."""
    todo = list(tree.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.Try, ast.If, ast.With, ast.ExceptHandler)):
            todo.extend(ast.iter_child_nodes(node))


def _imports(path):
    """({third-party top-level names imported at module level}, {... imported anywhere, e.g. lazily inside a function})."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    eager_nodes = {id(n) for n in _module_level_nodes(tree)}
    module_level, anywhere = set(), set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        for n in names:
            if n not in sys.stdlib_module_names and n not in {"bigbrain", "__future__"}:
                anywhere.add(n)
                if id(node) in eager_nodes:
                    module_level.add(n)
    return module_level, anywhere


def test_the_declared_dependencies_are_the_imports_the_package_makes():
    eager, lazy = set(), set()
    for path in sorted(PKG_DIR.glob("*.py")):
        e, a = _imports(path)
        eager |= e
        lazy |= a
    base = _dist_names(PYPROJECT["project"]["dependencies"])
    gpu = _dist_names(PYPROJECT["project"]["optional-dependencies"]["gpu"])
    # what the package needs the moment it is imported: only numpy. Everything else is imported inside a function.
    assert eager == {"numpy"}, eager
    # lazily: jinja2 (the chat template), torch and exllamav3 (the model), nothing else
    assert lazy == {"numpy", "jinja2", "torch", "exllamav3"}, lazy
    assert base == {"numpy", "jinja2"}, "torch / exllamav3 belong to the gpu extra, not to every install"
    assert gpu == {"torch", "exllamav3"}
    assert {"pytest", "build"} <= _dist_names(PYPROJECT["project"]["optional-dependencies"]["dev"])


def test_the_exllamav3_pin_is_the_one_the_docs_and_patches_name():
    spec = [r for r in PYPROJECT["project"]["optional-dependencies"]["gpu"] if r.startswith("exllamav3")][0]
    assert spec.replace(" ", "") == "exllamav3>=1.6.0,<1.7"
    assert ">=1.6.0,<1.7" in (ROOT / "INSTALL.md").read_text(encoding="utf-8")


def test_python_version_floor_matches_what_the_code_needs():
    assert PYPROJECT["project"]["requires-python"] == ">=3.11"
    assert "tomllib" in (PKG_DIR / "site.py").read_text(encoding="utf-8")        # why 3.11
