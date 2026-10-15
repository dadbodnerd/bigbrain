# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/registry.py -- which model is on which PC, and which GPUs each PC has.

model_registry.json is the one place BigBrain looks up. WHERE it is read from, in this order:
  1. $BIGBRAIN_REGISTRY, when set (manual CLI runs can also pass --registry);
  2. site.toml's [registry] path (bigbrain/site.py), when set;
  3. the file beside this module, when there is one (a dev checkout may keep a copy there; it is git-ignored);
  4. otherwise RegistryError, naming the variable. A release dir (deploy/release.py) is read-only and never holds
     data, so a forgotten setting fails loudly instead of reading some stale copy.
A sample is examples/model_registry.example.json (fake names and addresses; CONFIGURATION.md describes every field).
What it holds:

  nodes    one per PC: its ssh host, user, the venv python that has exllamav3
           (it can differ per box), its model root, whether the controller can launch a worker there at all (`launch`;
           false for a PC with no remote-exec path, with `launch_blocker` saying why), its LAN speed (`link_mb_s`), its
           NVLink pairs, and every GPU: nvidia-smi index, name, total MiB, and the
           server names whose processes live on it (`card_key` is the single-card endpoint; `card_keys` adds a pooled
           endpoint that spans it -- reserving that card must also hold the pooled one).
  models   one per model folder: size, layer count, layer types (hybrid models),
           KV elements per token, per-layer MiB when measured, and the folder's
           path on every PC that has it (`paths`, keyed by node name).

bigbrain/build_registry.py measures all of it over ssh (read-only) and
rewrites the file. Everything here is pure: no ssh, no network.

Hosts are accepted three ways everywhere: the node name ("alpha"), its address
("10.0.0.11") or an alias (any string in the node's `aliases` list).
"""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from . import site

REGISTRY_PATH = Path(__file__).resolve().with_name("model_registry.json")
ENV_PATH = "BIGBRAIN_REGISTRY"
FILE_VERSION = 1

#: Architectures whose blocks are not all attention: GatedDeltaNet / linear
#: attention layers carry a fixed recurrent state instead of a per-token KV
#: cache (Qwen3.5/3.8 "qwen3_5_text", Qwen3-Next).
HYBRID_ARCHES = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_next", "qwen3next"})

#: Architectures whose sliding-window layers exllamav3 runs as SlidingAttention: a small
#: per-sequence recurrent STATE (the window plus two pages of K/V, whatever the context),
#: not paged KV -- exllamav3 architecture/gpt_oss.py (swa_full=False) sets recurrent_state_cls
#: = SWAState. Such a model is NOT "hybrid" (no linear-attention layers) but a speculative
#: rollback must rewind that state, and the planner must price it as state.
SWA_STATE_ARCHES = frozenset({"gpt_oss"})
_SWA_DEFAULT_WINDOW = 128      # gpt-oss: sliding_window 128
_SWA_STATE_OVERP = 512         # SlidingAttention.sliding_window_overp: 2 pages kept before the window
_PAGE = 256


class RegistryError(Exception):
    """The registry is missing, unreadable, or names something it does not have."""


def path() -> Path:
    """The registry file in use: $BIGBRAIN_REGISTRY, else site.toml's [registry] path, else the file beside this module.
    (When that file does not exist either, load() raises a RegistryError that names the variable.)"""
    env = os.environ.get(ENV_PATH)
    if env:
        return Path(env)
    configured = site.registry_path()
    return configured if configured is not None else REGISTRY_PATH


def load(p: Union[str, Path, None] = None) -> dict:
    """The registry as a dict. Raises RegistryError naming the file."""
    explicit = bool(p)
    p = Path(p) if p else path()
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        if not explicit and not os.environ.get(ENV_PATH) and site.registry_path() is None:
            raise RegistryError(f"no BigBrain registry: set {ENV_PATH} to a model_registry.json, or [registry] path in "
                                f"site.toml (nothing at {p} either; a sample is examples/model_registry.example.json, "
                                f"`bigbrain-registry --write` makes one)") from None
        raise RegistryError(f"{p} does not exist -- run `bigbrain-registry --write` (python -m bigbrain.build_registry)") from None
    except (OSError, ValueError) as e:
        raise RegistryError(f"{p} is unreadable ({type(e).__name__}: {e})") from None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), dict) \
            or not isinstance(data.get("models"), dict):
        raise RegistryError(f"{p} is not a BigBrain registry (needs 'nodes' and 'models' objects)")
    return data


_write_lock = threading.Lock()


def save(reg: dict, p: Union[str, Path, None] = None) -> Path:
    """Write atomically (temp file + os.replace): a reader never sees half a file."""
    p = Path(p) if p else path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with _write_lock:
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=1, sort_keys=False)
                f.write("\n")
            os.replace(tmp, p)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return p


# ── nodes ────────────────────────────────────────────────────────────────

def nodes(reg: dict) -> Dict[str, dict]:
    return {k: v for k, v in (reg.get("nodes") or {}).items() if isinstance(v, dict)}


def node_name(reg: dict, host_or_name) -> Optional[str]:
    """The node name for a name, an address or an alias; None when unknown."""
    if host_or_name is None:
        return None
    want = str(host_or_name).strip()
    low = want.lower()
    for name, n in nodes(reg).items():
        if name.lower() == low or str(n.get("host") or "").lower() == low:
            return name
    for name, n in nodes(reg).items():
        if low in {str(a).lower() for a in (n.get("aliases") or [])}:
            return name
    return None


def node_meta(reg: dict, host_or_name) -> dict:
    name = node_name(reg, host_or_name)
    return dict(nodes(reg)[name]) if name else {}


def host_of(reg: dict, host_or_name) -> Optional[str]:
    """The ssh address of a node (its `host`), or None."""
    meta = node_meta(reg, host_or_name)
    return meta.get("host") or None


def launch_blocker(reg: dict, host_or_name) -> Optional[str]:
    """Why the controller cannot start a worker on this PC, or None."""
    name = node_name(reg, host_or_name)
    if name is None:
        return f"{host_or_name} is not in the registry ({path().name})"
    n = nodes(reg)[name]
    if n.get("launch") is False or not n.get("host"):
        return (f"{name} ({n.get('host') or 'no address'}) has no remote-start path: "
                f"{n.get('launch_blocker') or 'launch is false in the registry'}")
    return None


def resolve_hosts(reg: dict, spec: Union[str, Iterable[str], None]) -> List[str]:
    """Addresses for a comma list (or iterable) of names/addresses/aliases, in the
    order given. None = every launchable PC, in registry order."""
    if spec is None or spec == "":
        return [n["host"] for n in nodes(reg).values() if n.get("host") and n.get("launch") is not False]
    items = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if str(s).strip()]
    out = []
    for item in items:
        h = host_of(reg, item)
        if not h:
            raise RegistryError(f"{item!r} is not a PC in the registry (known: {', '.join(nodes(reg))})")
        out.append(h)
    return out


# ── GPUs ─────────────────────────────────────────────────────────────────

def _peers(n: dict, index: int) -> List[int]:
    out = []
    for pair in n.get("nvlink") or []:
        if isinstance(pair, (list, tuple)) and index in pair:
            out += [int(x) for x in pair if int(x) != index]
    return sorted(set(out))


def gpu_list(reg: dict) -> List[dict]:
    """Every GPU of every PC, flattened: index, name, total_mib, card_key,
    card_keys (single endpoint first, then pooled ones), host, host_name,
    nvlink_peers."""
    out = []
    for name, n in nodes(reg).items():
        for g in n.get("gpus") or []:
            if not isinstance(g, dict) or not isinstance(g.get("index"), int):
                continue
            keys = [k for k in (g.get("card_keys") or []) if k]
            if g.get("card_key") and g["card_key"] not in keys:
                keys.insert(0, g["card_key"])
            out.append({
                "host": n.get("host"), "host_name": name, "index": g["index"],
                "name": g.get("name") or "", "total_mib": int(g.get("total_mib") or 0),
                "card_key": g.get("card_key") or (keys[0] if keys else ""),
                "card_keys": keys, "nvlink_peers": _peers(n, g["index"]),
            })
    return out


def find_gpu(reg: dict, host_or_name, index: int) -> Optional[dict]:
    name = node_name(reg, host_or_name)
    if name is None:
        return None
    for g in gpu_list(reg):
        if g["host_name"] == name and g["index"] == int(index):
            return g
    return None


def all_cards(reg: dict, hosts: Iterable[str]) -> List[Tuple[str, int]]:
    """(address, gpu index) for every GPU of the given PCs, in the order given."""
    out = []
    for h in hosts:
        name = node_name(reg, h)
        if name is None:
            continue
        addr = nodes(reg)[name].get("host")
        for g in gpu_list(reg):
            if g["host_name"] == name:
                out.append((addr, g["index"]))
    return out


# ── models ───────────────────────────────────────────────────────────────

def model_info(reg: dict, model: str) -> Optional[dict]:
    m = (reg.get("models") or {}).get(model)
    return copy.deepcopy(m) if isinstance(m, dict) else None


def model_host_names(reg: dict, model: str) -> List[str]:
    """Node names that hold the model, in registry node order."""
    m = (reg.get("models") or {}).get(model)
    if not isinstance(m, dict):
        return []
    paths = m.get("paths") or {}
    return [name for name in nodes(reg) if paths.get(name)]


def model_hosts(reg: dict, model: str) -> List[str]:
    """Addresses of the PCs that hold the model (what the controller ssh's to)."""
    return [nodes(reg)[n]["host"] for n in model_host_names(reg, model) if nodes(reg)[n].get("host")]


def model_path(reg: dict, model: str, host_or_name) -> Optional[str]:
    """The model folder on that PC, or None when the PC does not have it. A folder the registry marks `missing_on` (an audit found
    it gone) is still listed in `paths` so the apps can say so, but it is not a folder the controller may place a slice on."""
    name = node_name(reg, host_or_name)
    m = (reg.get("models") or {}).get(model)
    if name is None or not isinstance(m, dict):
        return None
    if name in (m.get("missing_on") or []):
        return None
    return (m.get("paths") or {}).get(name) or None


def _text_config(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) and tc.get("num_hidden_layers") else cfg


def apply_config_facts(info: dict, cfg: dict) -> dict:
    """A copy of a registry model entry with the shape facts a model's own
    config.json gives: layers, layer_types, hidden_size, vocab_size, kv heads,
    head_dim, tied embeddings, and whether it is hybrid. Fields already in the
    entry from a measurement are kept; config facts fill what is missing."""
    out = copy.deepcopy(info or {})
    if not isinstance(cfg, dict):
        return out
    tc = _text_config(cfg)
    facts = {
        "layers": tc.get("num_hidden_layers"),
        "hidden_size": tc.get("hidden_size"),
        "vocab_size": tc.get("vocab_size") or cfg.get("vocab_size"),
        "num_key_value_heads": tc.get("num_key_value_heads"),
        "head_dim": tc.get("head_dim"),
        "tie_word_embeddings": tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings")),
    }
    types = tc.get("layer_types")
    if isinstance(types, list) and types:
        facts["layer_types"] = [str(t) for t in types]
    for k, v in facts.items():
        if v is not None and out.get(k) in (None, [], ""):
            out[k] = v
    arch = str(tc.get("model_type") or cfg.get("model_type") or out.get("arch") or "")
    if arch and not out.get("arch"):
        out["arch"] = arch
    types = out.get("layer_types") or []
    out["hybrid"] = bool(out.get("hybrid")) or arch in HYBRID_ARCHES or any(
        "linear" in str(t) for t in types)
    return out


def has_swa_state(info: dict) -> bool:
    """True for a model with sliding-window layers kept as recurrent state (gpt-oss)."""
    if not isinstance(info, dict):
        return False
    if info.get("swa_state") is not None:
        return bool(info["swa_state"])
    types = info.get("layer_types") or []
    return str(info.get("arch") or "") in SWA_STATE_ARCHES and any("sliding" in str(t) for t in types)


def has_recurrent_state(info: dict) -> bool:
    """Any per-sequence recurrent state exllamav3 must advance AND rewind: GatedDeltaNet
    (hybrid) or the sliding-window state of gpt-oss."""
    return is_hybrid(info) or has_swa_state(info)


def swa_state_mib(info: dict, cache_bytes: float = 2.0) -> float:
    """MiB of ONE sliding-window layer's state for ONE sequence: K and V of kv_state_size
    positions (window + overprovision, rounded up to whole pages) x kv heads x head dim."""
    window = info.get("sliding_window")
    window = int(window) if isinstance(window, (int, float)) and window > 0 else _SWA_DEFAULT_WINDOW
    kv_state = -(-(window + _SWA_STATE_OVERP) // _PAGE) * _PAGE
    heads, hd = info.get("num_key_value_heads"), info.get("head_dim")
    if isinstance(heads, int) and isinstance(hd, int) and heads > 0 and hd > 0:
        return kv_state * heads * hd * 2 * float(cache_bytes) / (1024.0 * 1024.0)
    return 8.0      # generous fallback when the registry lacks the head facts


def is_hybrid(info: dict) -> bool:
    if not isinstance(info, dict):
        return False
    if info.get("hybrid") is not None:
        return bool(info["hybrid"])
    if str(info.get("arch") or "") in HYBRID_ARCHES:
        return True
    return any("linear" in str(t) for t in info.get("layer_types") or [])
