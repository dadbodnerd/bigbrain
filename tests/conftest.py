# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Every test starts with no site file: a developer's own site.toml, $BIGBRAIN_SITE or $BIGBRAIN_REGISTRY must not change a result.

Which `bigbrain` the tests import: the INSTALLED one when there is one (`pip install -e ".[dev]"`, or a wheel -- the tests then
check what a user gets), else the checkout's own src/ so that a bare `python -m pytest` in a fresh clone still works."""
import importlib.util
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("bigbrain") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def _hermetic_site(monkeypatch):
    from bigbrain import site
    for var in ("BIGBRAIN_SITE", "BIGBRAIN_REGISTRY", "BIGBRAIN_PATCH_HOSTS", "BIGBRAIN_SERVERS_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(site, "SITE_FILE", "site.toml.not-read-by-the-tests")      # ./site.toml is not looked at either
    # nor is the developer's own ~/.config/bigbrain/site.toml
    monkeypatch.setattr(site, "user_site_file", lambda: Path(__file__).resolve().parent / "no-user-site" / "site.toml")
