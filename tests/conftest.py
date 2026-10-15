# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""Every test starts with no site file: a developer's own site.toml, $BIGBRAIN_SITE or $BIGBRAIN_REGISTRY must not change a result."""
import pytest


@pytest.fixture(autouse=True)
def _hermetic_site(monkeypatch):
    from tools.bigbrain import site
    for var in ("BIGBRAIN_SITE", "BIGBRAIN_REGISTRY", "BIGBRAIN_PATCH_HOSTS", "BIGBRAIN_SERVERS_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(site, "SITE_FILE", "site.toml.not-read-by-the-tests")      # ./site.toml is not looked at either
