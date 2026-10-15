# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""What the decoupling promises: the registry is found in a defined order, a release dir that holds no data fails
loudly, and build_registry needs no file beside the repo (no servers config unless you pass one)."""
import json
from pathlib import Path

import pytest

from tools.bigbrain import build_registry as br
from tools.bigbrain import registry as reg

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bigbrain" / "model_registry_trimmed.json"
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "model_registry.example.json"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(reg.ENV_PATH, raising=False)
    monkeypatch.delenv(br.CONFIG_ENV, raising=False)


# -- the registry lookup order: $BIGBRAIN_REGISTRY, then site.toml [registry] path, then beside the module, then an error --
def test_the_env_var_wins(monkeypatch, tmp_path):
    other = tmp_path / "beside.json"
    other.write_text(json.dumps({"nodes": {"x": {}}, "models": {}}), encoding="utf-8")
    monkeypatch.setattr(reg, "REGISTRY_PATH", other)
    monkeypatch.setenv(reg.ENV_PATH, str(FIXTURE))
    assert reg.path() == FIXTURE
    assert set(reg.load()["nodes"]) == {"alpha", "beta", "gamma"}


def test_site_toml_names_the_registry_when_the_env_var_is_unset(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "reg.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    site_toml = tmp_path / "site.toml"
    site_toml.write_text('[registry]\npath = "data/reg.json"\n', encoding="utf-8")        # relative to site.toml
    monkeypatch.setenv("BIGBRAIN_SITE", str(site_toml))
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "beside.json")
    assert reg.path() == tmp_path / "data" / "reg.json"
    assert "Mistral-Small-4-119B-4.03bpw-exl3" in reg.load()["models"]
    monkeypatch.setenv(reg.ENV_PATH, str(FIXTURE))                          # the env var still wins
    assert reg.path() == FIXTURE


def test_the_file_beside_the_module_is_used_when_the_env_var_is_unset(monkeypatch, tmp_path):
    beside = tmp_path / "model_registry.json"
    beside.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(reg, "REGISTRY_PATH", beside)
    assert reg.path() == beside
    assert "Mistral-Small-4-119B-4.03bpw-exl3" in reg.load()["models"]


def test_nothing_anywhere_is_an_error_that_names_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "model_registry.json")     # does not exist
    with pytest.raises(reg.RegistryError) as e:
        reg.load()
    assert reg.ENV_PATH in str(e.value)
    assert "model_registry.json" in str(e.value)


def test_an_env_var_pointing_nowhere_names_that_path(monkeypatch, tmp_path):
    gone = tmp_path / "gone.json"
    monkeypatch.setenv(reg.ENV_PATH, str(gone))
    with pytest.raises(reg.RegistryError) as e:
        reg.load()
    assert str(gone) in str(e.value)


def test_the_package_dir_holds_no_registry_data():
    """A release dir never holds data (git archive of a commit): no model_registry.json is tracked beside registry.py."""
    here = Path(reg.__file__).resolve().parent
    tracked_by_gitignore = (here.parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "/tools/bigbrain/model_registry.json" in tracked_by_gitignore


# -- the sample and the trimmed fixture are real registries ------------------------------------------------------
def test_the_example_registry_has_three_fake_nodes_and_five_models():
    r = reg.load(EXAMPLE)
    assert len(r["models"]) == 5
    assert set(r["nodes"]) == {"alpha", "beta", "gamma"}


def test_the_trimmed_fixture_keeps_every_kind_of_model_the_tests_need():
    r = reg.load(FIXTURE)
    assert reg.is_hybrid(r["models"]["Qwen2.5-1.5B-Instruct-8bpw-exl3"]) is False
    assert reg.has_swa_state(r["models"]["gpt-oss-120b-4.0bpw-exl3"]) is True
    assert reg.model_path(r, "Gemma-2-9B-Instruct", "beta") is None          # marked missing_on: never placed on
    assert reg.model_path(r, "Mistral-Small-4-119B-4.03bpw-exl3", "alpha")


# -- build_registry needs nothing from outside this repository ---------------------------------------------------
def test_build_registry_has_no_default_config_and_says_what_to_pass(capsys):
    assert br.main([]) == 2
    err = capsys.readouterr().err
    assert "--config" in err and br.CONFIG_ENV in err and "--node" in err


def test_build_registry_without_a_config_measures_the_nodes_it_is_told_about(monkeypatch, capsys):
    seen = []

    def boom(host, root, user=None, **kw):
        seen.append((host, root))
        raise OSError("no route")
    monkeypatch.setattr(br, "measure", boom)
    assert br.main(["--node", "beta=10.0.0.12:/srv/models"]) == 2
    assert seen == [("10.0.0.12", "/srv/models")]
    assert "NOT measured" in capsys.readouterr().out


def test_build_registry_reads_the_servers_config_from_the_env_var(monkeypatch, tmp_path, capsys):
    cfgp = tmp_path / "config.json"
    cfgp.write_text(json.dumps({"tabby": {"servers": {}}}), encoding="utf-8")
    monkeypatch.setenv(br.CONFIG_ENV, str(cfgp))
    assert br.main([]) == 2                                   # no PCs in that config: nothing measured
    assert "no PC could be measured" in capsys.readouterr().out


def test_the_kv_rule_comes_from_footprint_py():
    cfg = {"model_type": "llama", "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
           "hidden_size": 4096}
    row = {"name": "m", "config": cfg, "safetensors_bytes": 1 << 30}
    assert br._model_entry(row, {})["kv_elements_per_token"] == 32 * 8 * 128 * 2
