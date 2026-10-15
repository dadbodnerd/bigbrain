#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""tools/bigbrain/build_registry.py -- measure every GPU PC and (re)write model_registry.json.

Read-only on the nodes: ONE `python3 -` over ssh per PC, which runs nvidia-smi
(GPU index, name, total MiB, PCI bus; `topo -m` for NVLink pairs) and, for every
model folder under the PC's model root, reads its config.json and the JSON
HEADER of each *.safetensors file (the first 8 + N bytes -- no weights are read,
nothing is loaded, nothing is restarted). From the headers it sums the bytes of
every tensor per transformer block, the embedding, and the head (lm_head + the
final norm), so the planner splits by MEASURED per-layer size rather than
assuming uniform layers. Vision towers, MTP draft heads and audio encoders are
counted as `other` (not on the text pipeline).

Which PCs, two ways:
  --node NAME=HOST:MODEL_ROOT   (repeatable) name the PCs yourself. The simple way, and what to use unless you already
                                keep a TabbyAPI server list.
  --config PATH                 a JSON file shaped like {"tabby": {"servers": {NAME: {"url": "http://HOST:PORT",
                                "model_dir": "/abs/path"}}}} (or set $BIGBRAIN_SERVERS_CONFIG): every non-loopback host
                                with a Linux model_dir is measured. The server names also fill each GPU's `card_key` /
                                `card_keys` (which server lives on which card): a server entry with `telemetry_index`
                                or `telemetry_name_contains` says so itself; a pooled server (`gpu_split` /
                                `tensor_parallel`, optional `gpu_indices`) is added to every GPU it spans; a host with
                                one GPU gets all its servers; site.toml's [card_keys] adds two naming conventions.
A PC found by address is named by site.toml's [hosts] (address -> name), else by its address.

Hand-set node fields survive a rebuild: python, user, aliases, launch,
launch_blocker, link_mb_s, loopback_mb_s.

    python -m tools.bigbrain.build_registry --node alpha=10.0.0.11:/data/models                    # dry run
    python -m tools.bigbrain.build_registry --node alpha=10.0.0.11:/data/models --registry PATH --write
    python -m tools.bigbrain.build_registry --config servers.json --write
(--registry defaults to $BIGBRAIN_REGISTRY, then site.toml's [registry] path, then the file beside registry.py.)

Run it where the ssh keys to the PCs are. Exit 0 = written / printed; 2 = no node could be measured.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

from . import registry as bbreg
from . import site

#: The optional TabbyAPI-servers config (see the module docstring). No default path.
CONFIG_ENV = "BIGBRAIN_SERVERS_CONFIG"
MIB = float(1024 ** 2)
GIB = float(1024 ** 3)

#: Fields a person sets on a node; a rebuild never overwrites them.
HAND_FIELDS = ("python", "user", "aliases", "launch", "launch_blocker", "link_mb_s", "loopback_mb_s", "notes")
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


# ── the pure parts (also shipped to the node, see REMOTE_SCRIPT) ─────────

def _tensor_slot(name):
    """("layer", n) | ("embed", None) | ("head", None) | ("other", None) for one
    tensor name. Stdlib `re` only: this function's SOURCE runs on the node too."""
    import re as _re
    low = name.lower()
    for marker in ("visual", "vision", "mm_projector", "multi_modal", "audio", "mtp."):
        if marker in low:
            return ("other", None)
    if "embed_tokens" in low or low.endswith("wte.weight"):
        return ("embed", None)
    if "lm_head" in low:
        return ("head", None)
    m = _re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if m:
        return ("layer", int(m.group(1)))
    if _re.search(r"(?:^|\.)norm\.[a-z_]+$", low) or low.endswith("final_layernorm.weight"):
        return ("head", None)
    return ("other", None)


def _layer_bytes_core(headers, num_layers):
    """Sum tensor bytes per block / embed / head / other from parsed
    safetensors headers. Stdlib only: this function's SOURCE runs on the node."""
    layers = [0] * int(num_layers or 0)
    embed = head = other = 0
    for header in headers or []:
        for tname, meta in (header or {}).items():
            if tname == "__metadata__" or not isinstance(meta, dict):
                continue
            off = meta.get("data_offsets") or [0, 0]
            try:
                nbytes = int(off[1]) - int(off[0])
            except (TypeError, ValueError, IndexError):
                continue
            kind, idx = _tensor_slot(tname)
            if kind == "layer" and idx is not None and idx < len(layers):
                layers[idx] += nbytes
            elif kind == "embed":
                embed += nbytes
            elif kind == "head":
                head += nbytes
            else:
                other += nbytes
    return {"layer_bytes": layers, "embed_bytes": embed, "head_bytes": head, "other_bytes": other}


def layer_bytes(headers: List[dict], num_layers: int) -> dict:
    return _layer_bytes_core(headers, num_layers)


def parse_gpus(text: str) -> List[dict]:
    """nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id rows."""
    out = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx, total = int(parts[0]), int(float(parts[2]))
        except ValueError:
            continue
        out.append({"index": idx, "name": parts[1], "total_mib": total,
                    "bus_id": parts[3] if len(parts) > 3 else ""})
    return out


def parse_nvlink(topo: str) -> List[List[int]]:
    """[[a, b], ...] for every GPU pair `nvidia-smi topo -m` connects with NV#."""
    lines = [ln for ln in (topo or "").splitlines() if ln.strip()]
    if not lines:
        return []
    cols = [t for t in lines[0].split() if re.fullmatch(r"GPU\d+", t)]
    pairs = set()
    for ln in lines[1:]:
        cells = ln.split()
        if not cells or not re.fullmatch(r"GPU\d+", cells[0]):
            continue
        a = int(cells[0][3:])
        for j, cell in enumerate(cells[1:1 + len(cols)]):
            if cell.upper().startswith("NV"):
                b = int(cols[j][3:])
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
    return [list(p) for p in sorted(pairs)]


def _host(url) -> Optional[str]:
    try:
        return urlparse(str(url or "")).hostname
    except ValueError:
        return None


def card_keys_for_host(cfg: dict, host: str, gpus: List[dict]) -> List[dict]:
    """[{"index", "card_key", "card_keys"}] aligned with `gpus`: which
    tabby.servers keys live on each GPU of `host`."""
    index_prefixes, name_prefixes = site.card_key_prefixes()
    servers = ((cfg or {}).get("tabby") or {}).get("servers") or {}
    present = [g["index"] for g in gpus]
    single: Dict[int, List[str]] = {i: [] for i in present}
    pooled: Dict[int, List[str]] = {i: [] for i in present}
    for key in sorted(servers):
        e = servers[key]
        if not isinstance(e, dict) or _host(e.get("url")) != host:
            continue
        if e.get("gpu_split") or e.get("tensor_parallel"):
            idxs = e.get("gpu_indices") if isinstance(e.get("gpu_indices"), list) else present
            for i in idxs:
                if i in pooled:
                    pooled[i].append(key)
            continue
        idx = None
        if isinstance(e.get("telemetry_index"), int):
            idx = e["telemetry_index"] if e["telemetry_index"] in single else None
        elif e.get("telemetry_name_contains"):
            needle = str(e["telemetry_name_contains"]).lower()
            idx = next((g["index"] for g in gpus if needle in str(g.get("name", "")).lower()), None)
        elif any(key.startswith(pfx) and key[len(pfx):].isdigit() for pfx in index_prefixes):
            pfx = next(pf for pf in index_prefixes if key.startswith(pf) and key[len(pf):].isdigit())
            n = int(key[len(pfx):])
            idx = n if n in single else None
        elif any(key.startswith(pfx) for pfx in name_prefixes):
            pfx = next(pf for pf in name_prefixes if key.startswith(pf))
            suffix = key[len(pfx):].lower()
            idx = next((g["index"] for g in gpus if suffix in str(g.get("name", "")).lower()), None)
        elif len(present) == 1:
            idx = present[0]
        if idx is not None:
            single[idx].append(key)
    out = []
    for g in gpus:
        keys = single[g["index"]] + [k for k in pooled[g["index"]] if k not in single[g["index"]]]
        out.append({"index": g["index"], "card_key": single[g["index"]][0] if single[g["index"]] else "",
                    "card_keys": keys})
    return out


def nodes_from_config(cfg: dict) -> List[dict]:
    """[{name, host, model_root}] -- one per distinct non-loopback host with a
    Linux model_dir in tabby.servers."""
    seen, out = {}, []
    for e in ((cfg or {}).get("tabby") or {}).get("servers", {}).values():
        if not isinstance(e, dict):
            continue
        host = _host(e.get("url"))
        root = e.get("model_dir")
        if not host or host in _LOOPBACK or str(e.get("url", "")).startswith("https"):
            continue
        if not isinstance(root, str) or not root.startswith("/") or host in seen:
            continue
        seen[host] = True
        out.append({"name": site.names_by_address().get(host, host), "host": host, "model_root": root.rstrip("/")})
    return out


def _model_entry(row: dict, prior: dict) -> dict:
    """One registry model entry from one measured row (shape facts + sizes)."""
    from .footprint import kv_shape  # noqa: PLC0415 -- the one rule for KV elements per token
    cfg = row.get("config") or {}
    out = copy.deepcopy(prior or {})
    out.pop("stale", None)
    full, _sliding, _window = kv_shape(cfg)
    layers = cfg.get("num_hidden_layers")
    facts = {"arch": cfg.get("model_type"), "bits": cfg.get("bits"), "layers": layers,
             "layer_types": cfg.get("layer_types"), "hidden_size": cfg.get("hidden_size"),
             "vocab_size": cfg.get("vocab_size"), "num_key_value_heads": cfg.get("num_key_value_heads"),
             "head_dim": cfg.get("head_dim"), "kv_elements_per_token": int(full)}
    out.update({k: v for k, v in facts.items() if v is not None})
    if row.get("safetensors_bytes"):
        out["size_gib"] = round(float(row["safetensors_bytes"]) / GIB, 3)
    lb = row.get("layer_bytes")
    if isinstance(lb, list) and lb and len(lb) == (layers or 0) and all(isinstance(x, (int, float)) for x in lb):
        out["layer_mib"] = [round(float(x) / MIB, 3) for x in lb]
        out["embed_mib"] = round(float(row.get("embed_bytes") or 0) / MIB, 3)
        out["head_mib"] = round(float(row.get("head_bytes") or 0) / MIB, 3)
        out["other_mib"] = round(float(row.get("other_bytes") or 0) / MIB, 3)
    out["hybrid"] = bbreg.is_hybrid({"arch": out.get("arch"), "layer_types": out.get("layer_types")})
    return out


def merge(existing: dict, measured: Dict[str, dict], generated_at: Optional[str] = None) -> dict:
    """The new registry: `existing` with every MEASURED node's GPUs and model
    paths replaced by what was found there. A model a measured node no longer
    has loses that node's path; a model left with no path at all is dropped (it
    is on no PC). Nodes not measured this run keep everything."""
    new = copy.deepcopy(existing or {})
    new.setdefault("version", bbreg.FILE_VERSION)
    new.setdefault("nodes", {})
    new.setdefault("models", {})
    new.pop("_seed", None)
    for name, m in measured.items():
        node = new["nodes"].setdefault(name, {})
        hand = {k: node[k] for k in HAND_FIELDS if k in node}
        node.update({"host": m["host"], "model_root": m["model_root"], "gpus": m.get("gpus") or [],
                     "nvlink": m.get("nvlink") or []})
        node.update(hand)
        node.setdefault("user", site.ssh_user())
        node.setdefault("launch", True)
        for entry in new["models"].values():
            if isinstance(entry, dict):
                (entry.get("paths") or {}).pop(name, None)
        root = m["model_root"].rstrip("/")
        for row in m.get("models") or []:
            if row.get("error") or not row.get("name"):
                continue
            prior = new["models"].get(row["name"]) or {}
            entry = _model_entry(row, prior)
            entry.setdefault("paths", {})
            entry["paths"] = dict(entry.get("paths") or {})
            entry["paths"][name] = f"{root}/{row['name']}"
            new["models"][row["name"]] = entry
    new["models"] = {k: v for k, v in sorted(new["models"].items(), key=lambda kv: kv[0].lower())
                     if isinstance(v, dict) and v.get("paths")}
    new["generated_at"] = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return new


# ── the remote half ──────────────────────────────────────────────────────

_REMOTE_MAIN = r'''
import json, os, struct, subprocess, sys
root = sys.argv[1]

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""

out = {"gpus": sh("nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader,nounits"),
       "topo": sh("nvidia-smi topo -m"), "models": []}
KEEP = ("model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
        "hidden_size", "vocab_size", "layer_types", "sliding_window", "sliding_window_pattern",
        "use_sliding_window", "tie_word_embeddings")
for name in sorted(os.listdir(root)):
    d = os.path.join(root, name)
    cfg_path = os.path.join(d, "config.json")
    if name.startswith(".") or not os.path.isdir(d) or not os.path.exists(cfg_path):
        continue
    try:
        cfg = json.load(open(cfg_path))
    except Exception as e:
        out["models"].append({"name": name, "error": str(e)[:200]})
        continue
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) and \
        cfg["text_config"].get("num_hidden_layers") else cfg
    keep = {k: tc.get(k) for k in KEEP}
    if keep.get("model_type") is None:
        keep["model_type"] = cfg.get("model_type")
    blocks = cfg.get("block_configs") or tc.get("block_configs")
    if isinstance(blocks, list):
        keep["block_attention"] = [(b or {}).get("attention") or {} for b in blocks]
    keep["bits"] = (cfg.get("quantization_config") or {}).get("bits")
    headers, size = [], 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".safetensors"):
            continue
        fp = os.path.join(d, f)
        try:
            size += os.path.getsize(fp)
            with open(fp, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if 0 < n < 256 * 1024 * 1024:
                    headers.append(json.loads(fh.read(n)))
        except Exception:
            continue
    row = {"name": name, "config": keep, "safetensors_bytes": size}
    row.update(_layer_bytes_core(headers, keep.get("num_hidden_layers") or 0))
    out["models"].append(row)
print(json.dumps(out))
'''

#: What runs on the node: the two pure functions above, verbatim, then the main.
REMOTE_SCRIPT = (inspect.getsource(_tensor_slot) + "\n\n" + inspect.getsource(_layer_bytes_core)
                 + "\n" + _REMOTE_MAIN)


def measure(host: str, model_root: str, user: Optional[str] = None, timeout: int = 300) -> dict:
    """One ssh round trip: the node's raw GPU/topology/model facts."""
    user = user or site.ssh_user()
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"{user}@{host}",
            f"python3 - {_shq(model_root)}"]
    p = subprocess.run(argv, input=REMOTE_SCRIPT, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{host}: ssh exited {p.returncode}: {(p.stderr or p.stdout).strip()[:300]}")
    return json.loads(p.stdout)


def _shq(s: str) -> str:
    import shlex  # noqa: PLC0415
    return shlex.quote(s)


def measured_node(cfg: dict, name: str, host: str, model_root: str, raw: dict) -> dict:
    gpus = parse_gpus(raw.get("gpus") or "")
    keys = {k["index"]: k for k in card_keys_for_host(cfg, host, gpus)}
    for g in gpus:
        g["card_key"] = keys.get(g["index"], {}).get("card_key", "")
        g["card_keys"] = keys.get(g["index"], {}).get("card_keys", [])
        g.pop("bus_id", None)
    return {"host": host, "model_root": model_root, "gpus": gpus,
            "nvlink": parse_nvlink(raw.get("topo") or ""), "models": raw.get("models") or []}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Measure the GPU PCs and rewrite model_registry.json")
    ap.add_argument("--write", action="store_true", help="rewrite the registry (default: print only)")
    ap.add_argument("--config", default=os.environ.get(CONFIG_ENV),
                    help=f"a JSON file with tabby.servers to read the PCs (and card keys) from (default: ${CONFIG_ENV})")
    ap.add_argument("--registry", default=None,
                    help=f"registry path (default: ${bbreg.ENV_PATH}, then site.toml [registry] path, then the file beside "
                         f"registry.py)")
    ap.add_argument("--node", action="append", default=[], metavar="NAME=HOST:ROOT",
                    help="measure this PC instead of the ones the config names (repeatable)")
    a = ap.parse_args(argv)
    if a.config:
        try:
            with open(a.config, encoding="utf-8-sig") as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            print(f"cannot read {a.config}: {e}", file=sys.stderr)
            return 2
    elif a.node:
        cfg = {}            # --node names the PCs; no tabby.servers means no card keys
    else:
        print(f"no PCs to measure: name them with --node NAME=HOST:ROOT, or pass --config PATH (or set {CONFIG_ENV}) "
              f"to a tabby.servers file", file=sys.stderr)
        return 2
    try:
        existing = bbreg.load(a.registry)
    except bbreg.RegistryError:
        existing = {"version": bbreg.FILE_VERSION, "nodes": {}, "models": {}}
    targets = []
    for spec in a.node:
        name, _, rest = spec.partition("=")
        host, _, root = rest.partition(":")
        targets.append({"name": name, "host": host, "model_root": root.rstrip("/")})
    if not targets:
        targets = nodes_from_config(cfg)
    measured = {}
    for t in targets:
        user = (existing.get("nodes", {}).get(t["name"]) or {}).get("user") or site.ssh_user()
        try:
            raw = measure(t["host"], t["model_root"], user=user)
        except Exception as e:  # noqa: BLE001 -- one PC down must not stop the others
            print(f"  {t['name']} ({t['host']}): NOT measured -- {e}")
            continue
        measured[t["name"]] = measured_node(cfg, t["name"], t["host"], t["model_root"], raw)
        node = measured[t["name"]]
        print(f"  {t['name']} ({t['host']}): {len(node['gpus'])} GPU(s), NVLink {node['nvlink'] or 'none'}, "
              f"{len(node['models'])} model folder(s)")
    if not measured:
        print("no PC could be measured -- registry unchanged")
        return 2
    new = merge(existing, measured)
    if a.write:
        p = bbreg.save(new, a.registry)
        print(f"wrote {p} ({len(new['models'])} models on {len(new['nodes'])} PCs)")
    else:
        print(json.dumps(new, indent=1)[:20000])
        print("(dry run -- pass --write to rewrite the registry)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
