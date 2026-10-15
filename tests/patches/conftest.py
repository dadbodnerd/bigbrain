# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""The patch kit (patches/) is repo-only tooling: it is not part of the installed package and is left out of the sdist (other
licences). Without that directory there is nothing to test here, so collect nothing instead of failing on the import."""
from pathlib import Path

PATCHES = Path(__file__).resolve().parent.parent.parent / "patches"

collect_ignore_glob = [] if PATCHES.is_dir() else ["test_*.py"]
