# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""BigBrain: cross-machine pipeline-parallel LLM inference on exllamav3."""

# The single source of the version: pyproject.toml reads it from here (tool.setuptools.dynamic). A plain string, not
# importlib.metadata, because the controller also copies this file to GPU nodes that have no installed distribution.
__version__ = "0.2.0"
