# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/registry.py + build_registry.py: which model is on which PC.

Pure tests: a scratch registry in a temp dir, fake nvidia-smi / topo output,
fake safetensors headers. No ssh, no network, no cluster.

    C:/Python314/python.exe -m pytest tests/test_bigbrain_registry.py -q -p no:cacheprovider
"""
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigbrain import build_registry as br  # noqa: E402
from bigbrain import registry as reg  # noqa: E402


def scratch_registry():
    return {
        "version": 1,
        "nodes": {
            "alpha": {
                "host": "10.0.0.11", "aliases": ["ALPHA_PC"], "user": "bigbrain",
                "python": "/opt/venvs/alpha/bin/python",
                "model_root": "/data/models", "launch": True, "link_mb_s": 110,
                "gpus": [
                    {"index": 0, "name": "RTX 3090", "total_mib": 24576,
                     "card_key": "ALPHA_GPU0", "card_keys": ["ALPHA_GPU0", "ALPHA"]},
                    {"index": 1, "name": "RTX 3090", "total_mib": 24576,
                     "card_key": "ALPHA_GPU1", "card_keys": ["ALPHA_GPU1", "ALPHA"]},
                ],
                "nvlink": [[0, 1]],
            },
            "beta": {
                "host": "10.0.0.12", "aliases": ["BETA_PC"], "user": "bigbrain",
                "python": "/opt/venvs/beta/bin/python",
                "model_root": "/srv/models", "launch": True,
                "gpus": [
                    {"index": 0, "name": "RTX 5070", "total_mib": 12227, "card_key": "BETA_5070"},
                    {"index": 1, "name": "RTX 3090", "total_mib": 24576, "card_key": "BETA_3090"},
                    {"index": 2, "name": "RTX 3060", "total_mib": 12288, "card_key": "BETA_3060"},
                ],
            },
            "gamma": {
                "host": "10.0.0.13", "launch": False,
                "launch_blocker": "Windows, no ssh",
                "gpus": [{"index": 0, "name": "RTX 5070 Ti", "total_mib": 16303, "card_key": "GAMMA"}],
            },
        },
        "models": {
            "Qwen2.5-1.5B-Instruct-8bpw-exl3": {
                "arch": "qwen2", "size_gib": 1.821, "kv_elements_per_token": 14336, "layers": 28,
                "paths": {"alpha": "/data/models/Qwen2.5-1.5B-Instruct-8bpw-exl3",
                          "beta": "/srv/models/Qwen2.5-1.5B-Instruct-8bpw-exl3"},
            },
            "llama70b": {
                "arch": "llama", "size_gib": 34.581, "kv_elements_per_token": 163840,
                "paths": {"alpha": "/data/models/llama70b"},
            },
        },
    }


@pytest.fixture()
def regfile(tmp_path):
    p = tmp_path / "model_registry.json"
    p.write_text(json.dumps(scratch_registry()), encoding="utf-8")
    return p


# ── loading and lookups (the controller's API) ──────────────────────────

def test_load_reads_the_file_and_names_its_path(regfile):
    r = reg.load(regfile)
    assert set(r["models"]) == {"Qwen2.5-1.5B-Instruct-8bpw-exl3", "llama70b"}
    assert reg.REGISTRY_PATH.name == "model_registry.json"


def test_load_refuses_a_file_that_is_not_a_registry(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(reg.RegistryError):
        reg.load(p)
    with pytest.raises(reg.RegistryError):
        reg.load(tmp_path / "missing.json")


def test_hosts_resolve_by_name_ip_and_alias(regfile):
    r = reg.load(regfile)
    assert reg.node_name(r, "alpha") == "alpha"
    assert reg.node_name(r, "10.0.0.11") == "alpha"
    assert reg.node_name(r, "ALPHA_PC") == "alpha"
    assert reg.node_name(r, "nowhere") is None
    assert reg.node_meta(r, "10.0.0.12")["python"].endswith("/opt/venvs/beta/bin/python")
    assert reg.node_meta(r, "nowhere") == {}


def test_model_hosts_and_paths_are_per_host(regfile):
    r = reg.load(regfile)
    m = "Qwen2.5-1.5B-Instruct-8bpw-exl3"
    assert reg.model_hosts(r, m) == ["10.0.0.11", "10.0.0.12"]
    assert reg.model_host_names(r, m) == ["alpha", "beta"]
    assert reg.model_path(r, m, "10.0.0.12") == "/srv/models/Qwen2.5-1.5B-Instruct-8bpw-exl3"
    assert reg.model_path(r, m, "beta") == "/srv/models/Qwen2.5-1.5B-Instruct-8bpw-exl3"
    assert reg.model_path(r, "llama70b", "beta") is None
    assert reg.model_hosts(r, "nope") == []


def test_resolve_hosts_defaults_to_every_launchable_pc(regfile):
    r = reg.load(regfile)
    assert reg.resolve_hosts(r, None) == ["10.0.0.11", "10.0.0.12"]
    assert reg.resolve_hosts(r, "beta,alpha") == ["10.0.0.12", "10.0.0.11"]
    with pytest.raises(reg.RegistryError):
        reg.resolve_hosts(r, "beta,atlantis")


def test_all_cards_lists_every_gpu_of_the_hosts_in_order(regfile):
    r = reg.load(regfile)
    assert reg.all_cards(r, ["10.0.0.11", "10.0.0.12"]) == [
        ("10.0.0.11", 0), ("10.0.0.11", 1),
        ("10.0.0.12", 0), ("10.0.0.12", 1), ("10.0.0.12", 2)]


def test_gpu_list_flattens_with_host_and_card_keys(regfile):
    r = reg.load(regfile)
    gpus = reg.gpu_list(r)
    g0 = next(g for g in gpus if g["host_name"] == "alpha" and g["index"] == 0)
    assert g0["host"] == "10.0.0.11"
    assert g0["card_keys"] == ["ALPHA_GPU0", "ALPHA"]      # the pooled endpoint spans it
    assert g0["nvlink_peers"] == [1]
    r3060 = reg.find_gpu(r, "beta", 2)
    assert r3060["card_key"] == "BETA_3060" and r3060["card_keys"] == ["BETA_3060"]
    assert reg.find_gpu(r, "beta", 7) is None


def test_launch_blocker_names_the_windows_reason(regfile):
    r = reg.load(regfile)
    assert reg.launch_blocker(r, "alpha") is None
    assert "no ssh" in reg.launch_blocker(r, "10.0.0.13")
    assert "not in the registry" in reg.launch_blocker(r, "10.9.9.9")


def test_apply_config_facts_reads_text_config_and_layer_types():
    info = {"size_gib": 17.0, "kv_elements_per_token": 32768}
    cfg = {"model_type": "qwen3_5", "text_config": {
        "num_hidden_layers": 8, "hidden_size": 5120, "vocab_size": 248320,
        "num_key_value_heads": 4, "head_dim": 256, "tie_word_embeddings": False,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3
                       + ["full_attention"]}}
    out = reg.apply_config_facts(info, cfg)
    assert out["layers"] == 8 and out["hidden_size"] == 5120 and out["vocab_size"] == 248320
    assert out["layer_types"][3] == "full_attention"
    assert out["hybrid"] is True
    # the input is not mutated
    assert "layers" not in info


def test_save_is_atomic_and_round_trips(tmp_path):
    p = tmp_path / "r.json"
    reg.save(scratch_registry(), p)
    assert reg.load(p)["nodes"]["beta"]["host"] == "10.0.0.12"
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


# ── build_registry: the pure parts of the discovery ─────────────────────

def test_parse_gpus_and_nvlink_topology():
    gpus = br.parse_gpus("0, NVIDIA GeForce RTX 3090, 24576, 00000000:01:00.0\n"
                         "1, NVIDIA GeForce RTX 3090, 24576, 00000000:2E:00.0\n"
                         "garbage line\n")
    assert [(g["index"], g["total_mib"]) for g in gpus] == [(0, 24576), (1, 24576)]
    topo = ("\tGPU0\tGPU1\tCPU Affinity\n"
            "GPU0\t X \tNV4\t0-23\n"
            "GPU1\tNV4\t X \t0-23\n")
    assert br.parse_nvlink(topo) == [[0, 1]]
    assert br.parse_nvlink("\tGPU0\tGPU1\nGPU0\t X \tPHB\nGPU1\tPHB\t X \n") == []


def _safetensors_header(tensors):
    """A safetensors file's first bytes: u64 length + JSON header."""
    hdr, off = {}, 0
    for name, nbytes in tensors.items():
        hdr[name] = {"dtype": "U8", "shape": [nbytes], "data_offsets": [off, off + nbytes]}
        off += nbytes
    raw = json.dumps(hdr).encode()
    return struct.pack("<Q", len(raw)) + raw


def test_layer_bytes_groups_tensors_by_block_embed_and_head():
    header = json.loads(_safetensors_header({
        "model.embed_tokens.weight": 1000,
        "model.layers.0.self_attn.q_proj.weight": 100,
        "model.layers.0.mlp.down_proj.weight": 300,
        "model.layers.1.linear_attn.in_proj.weight": 250,
        "model.layers.1.mlp.down_proj.weight": 300,
        "model.norm.weight": 10,
        "lm_head.weight": 900,
        "visual.blocks.0.attn.qkv.weight": 777,
    })[8:])
    out = br.layer_bytes([header], num_layers=2)
    assert out["layer_bytes"] == [400, 550]
    assert out["embed_bytes"] == 1000
    assert out["head_bytes"] == 910           # lm_head + the final norm
    assert out["other_bytes"] == 777          # a vision tower is not on the pipeline


def test_layer_bytes_understands_multimodal_prefixes():
    header = json.loads(_safetensors_header({
        "model.language_model.layers.0.mlp.up_proj.weight": 5,
        "model.language_model.layers.1.mlp.up_proj.weight": 7,
        "model.language_model.embed_tokens.weight": 11,
    })[8:])
    out = br.layer_bytes([header], num_layers=2)
    assert out["layer_bytes"] == [5, 7] and out["embed_bytes"] == 11


def test_the_remote_script_runs_and_measures_a_model_folder(tmp_path):
    """The script build_registry ships to a node, run locally on a fake model
    root (PATH emptied: no nvidia-smi is invoked here). It must parse the
    folder's config and the safetensors HEADERS into per-layer bytes."""
    import subprocess
    d = tmp_path / "Tiny-exl3"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "num_hidden_layers": 2,
                                               "hidden_size": 8, "vocab_size": 16,
                                               "num_key_value_heads": 2, "head_dim": 4}), encoding="utf-8")
    hdr = _safetensors_header({"model.embed_tokens.weight": 64, "model.layers.0.mlp.w": 10,
                               "model.layers.1.mlp.w": 30, "lm_head.weight": 32})
    (d / "model.safetensors").write_bytes(hdr + b"\0" * 136)
    (tmp_path / "not-a-model").mkdir()
    env = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "COMSPEC", "TEMP", "TMP")}
    env["PATH"] = ""
    p = subprocess.run([sys.executable, "-I", "-", str(tmp_path)], input=br.REMOTE_SCRIPT,
                       capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert [m["name"] for m in out["models"]] == ["Tiny-exl3"]
    m = out["models"][0]
    assert m["layer_bytes"] == [10, 30] and m["embed_bytes"] == 64 and m["head_bytes"] == 32
    assert m["config"]["num_hidden_layers"] == 2 and m["safetensors_bytes"] == len(hdr) + 136


def test_card_keys_follow_the_mapping_rules(tmp_path, monkeypatch):
    """Two naming conventions come from site.toml [card_keys]; telemetry_index, pooled servers and the one-GPU rule need none."""
    site_toml = tmp_path / "site.toml"
    site_toml.write_text('[card_keys]\nindex_prefixes = ["ALPHA_GPU"]\nname_prefixes = ["BETA_"]\n', encoding="utf-8")
    monkeypatch.setenv("BIGBRAIN_SITE", str(site_toml))
    cfg = {"tabby": {"servers": {
        "ALPHA": {"url": "http://10.0.0.11:5000", "gpu_split": [17.0, 19.0]},
        "ALPHA_GPU0": {"url": "http://10.0.0.11:5002"},
        "ALPHA_GPU1": {"url": "http://10.0.0.11:5001"},
        "BETA_3090": {"url": "http://10.0.0.12:5001"},
        "BETA_3060": {"url": "http://10.0.0.12:5002"},
        "BETA_5070": {"url": "http://10.0.0.12:5000"},
        "PINNED": {"url": "http://10.0.0.12:5009", "telemetry_index": 2},
        "CLOUD": {"url": "https://api.example.com"},
    }}}
    alpha = [{"index": 0, "name": "NVIDIA GeForce RTX 3090"}, {"index": 1, "name": "NVIDIA GeForce RTX 3090"}]
    keys = br.card_keys_for_host(cfg, "10.0.0.11", alpha)
    assert keys[0]["card_key"] == "ALPHA_GPU0" and keys[0]["card_keys"] == ["ALPHA_GPU0", "ALPHA"]
    assert keys[1]["card_key"] == "ALPHA_GPU1" and keys[1]["card_keys"] == ["ALPHA_GPU1", "ALPHA"]
    beta = [{"index": 0, "name": "NVIDIA GeForce RTX 5070"}, {"index": 1, "name": "NVIDIA GeForce RTX 3090"},
           {"index": 2, "name": "NVIDIA GeForce RTX 3060"}]
    keys = br.card_keys_for_host(cfg, "10.0.0.12", beta)
    assert keys[0]["card_key"] == "BETA_5070"
    assert keys[1]["card_key"] == "BETA_3090"
    assert keys[2]["card_keys"] == ["BETA_3060", "PINNED"]


def test_build_merges_measurements_and_keeps_hand_fields():
    old = scratch_registry()
    old["nodes"]["beta"]["link_mb_s"] = 1150          # set by hand after the 10 GbE NIC
    measured = {
        "beta": {"host": "10.0.0.12", "model_root": "/srv/models",
                "gpus": [{"index": 0, "name": "RTX 5070", "total_mib": 12227, "card_key": "BETA_5070",
                          "card_keys": ["BETA_5070"]}],
                "nvlink": [],
                "models": [{"name": "NewModel-exl3", "config": {"num_hidden_layers": 4, "hidden_size": 64,
                                                                "vocab_size": 100,
                                                                "num_key_value_heads": 2, "head_dim": 8},
                            "safetensors_bytes": 4 * 2 ** 30,
                            "layer_bytes": [2 ** 28] * 4, "embed_bytes": 2 ** 20, "head_bytes": 2 ** 20,
                            "other_bytes": 0}]},
    }
    new = br.merge(old, measured, generated_at="2026-10-12T00:00:00Z")
    beta = new["nodes"]["beta"]
    assert beta["link_mb_s"] == 1150 and beta["python"].endswith("/opt/venvs/beta/bin/python")
    assert [g["index"] for g in beta["gpus"]] == [0]
    m = new["models"]["NewModel-exl3"]
    assert m["layers"] == 4 and m["paths"] == {"beta": "/srv/models/NewModel-exl3"}
    assert m["layer_mib"] == [256.0] * 4 and m["embed_mib"] == 1.0 and m["head_mib"] == 1.0
    assert m["size_gib"] == 4.0
    assert m["kv_elements_per_token"] == 4 * 2 * 8 * 2
    # a model the beta no longer has loses its beta path; one seen only on alpha keeps alpha
    assert "beta" not in new["models"]["Qwen2.5-1.5B-Instruct-8bpw-exl3"]["paths"]
    assert "alpha" in new["models"]["Qwen2.5-1.5B-Instruct-8bpw-exl3"]["paths"]


def test_the_example_registry_loads_and_every_path_names_its_model_folder():
    """examples/model_registry.example.json loads, every model path is an absolute folder named like the model and sits under
    its node's model_root, and the Windows PC cannot be launched on. (`paths`, not model_root, is what the controller places
    slices from, so a real registry may hold folders outside the root; the example keeps them inside.)"""
    r = reg.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "examples", "model_registry.example.json"))
    assert len(r["models"]) == 5
    for name, m in r["models"].items():
        for node, p in (m.get("paths") or {}).items():
            assert node in r["nodes"], (name, node)
            assert p.startswith("/") and p.rstrip("/").rsplit("/", 1)[-1] == name, (name, node, p)
            assert p.startswith(r["nodes"][node]["model_root"].rstrip("/") + "/"), (name, node, p)
    assert r["nodes"]["gamma"]["launch"] is False and "no ssh" in reg.launch_blocker(r, "gamma")
