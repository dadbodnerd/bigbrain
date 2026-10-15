# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/site.py -- the one place BigBrain reads site-specific settings from.

Everything that belongs to YOUR network and not to the engine lives in one small TOML file, `site.toml`
(`site.example.toml` is the annotated template). Nothing in the engine hard-codes an address, a user name or a path.

Where the file is read from, in this order:
  1. $BIGBRAIN_SITE, when set (a missing file there is an error, not a silent fallback);
  2. ~/.config/bigbrain/site.toml ($XDG_CONFIG_HOME/bigbrain/site.toml when that is set), when there is one;
  3. otherwise there is no site file, and every accessor returns its documented default.
A site.toml in the current directory is NOT read on its own (since v0.2.1): point $BIGBRAIN_SITE at it.

What it can hold (every key is optional; an unknown key is an error, so a typo is not silently ignored):

    [ssh]
    user = "bigbrain"                 # default ssh user for a node whose registry entry names none (default: the local user)

    [registry]
    path = "model_registry.json"      # the model registry; a relative path is relative to site.toml. $BIGBRAIN_REGISTRY wins

    [hosts.alpha]                    # short names for deploy/release.py --host and deploy/verify_stage.py --host,
    address = "10.0.0.11"             # and the node names build_registry.py gives machines it finds by address
    user = "bigbrain"                 # optional; falls back to [ssh] user

    [card_keys]                       # build_registry.py only: how a TabbyAPI server name says which GPU it lives on
    index_prefixes = ["GPU"]          # a server called GPU0 / GPU1 lives on nvidia-smi index 0 / 1
    name_prefixes = ["BOX_"]          # a server called BOX_3090 lives on the GPU whose name contains "3090"

    [capacity]
    reader = "my_module:free_mib"     # optional: replaces the built-in nvidia-smi-over-ssh reader behind `--split capacity`

    [patches]
    hosts = "patches/hosts.toml"      # the host list patches/verify.py checks (see patches/hosts.example.toml)

This module reads files and the environment and nothing else: no ssh, no network. It is controller-side; it is not one of the
files staged to the GPU nodes (controller.PACKAGE_FILES).
"""
from __future__ import annotations

import getpass
import ipaddress
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

ENV_SITE = "BIGBRAIN_SITE"
SITE_FILE = "site.toml"


# ── host and user names (v0.2.1, security review M1) ───────────────────────
#
# A host or user name ends up in an ssh argv and, for the hosts, inside the remote shell command of a node. Both come from files a
# user may have been handed (a registry, a site file, a servers config), so they are checked against a plain charset before they are
# used anywhere: no shell metacharacters, no whitespace, and never a leading '-' (which ssh would read as an option).

_HOSTNAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,252})")
_SCOPE_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")


def valid_host(h) -> bool:
    """True for an IP address (v4, or v6 with an optional plain %scope) or a host name of letters, digits, '.', '-' and '_' that
    starts with a letter or a digit."""
    if not isinstance(h, str) or not h:
        return False
    try:
        ipaddress.ip_address(h)
        return "%" not in h or bool(_SCOPE_ID.fullmatch(h.split("%", 1)[1]))
    except ValueError:
        return bool(_HOSTNAME.fullmatch(h))


def valid_user(u) -> bool:
    """True for a POSIX-style user name: letters, digits, '_', '.', '-', not starting with a digit, '.' or '-', at most 64."""
    return isinstance(u, str) and bool(_USER.fullmatch(u))


def check_host(h, what: str = "host") -> str:
    """`h`, or ValueError naming what is wrong with it."""
    if not valid_host(h):
        raise ValueError(f"{what} {h!r} is not an IP address or a host name (letters, digits, '.', '-', '_'; it must start with "
                         f"a letter or a digit)")
    return h


def check_user(u, what: str = "ssh user") -> str:
    """`u`, or ValueError naming what is wrong with it."""
    if not valid_user(u):
        raise ValueError(f"{what} {u!r} is not a user name (letters, digits, '_', '.', '-'; it must not start with '-', '.' or a "
                         f"digit). Set [ssh] user in site.toml, or `user` on the node in the registry")
    return u


def check_ssh_target(t: str) -> str:
    """`user@host` or `host`, both parts checked; ValueError otherwise."""
    if not isinstance(t, str):
        raise ValueError(f"ssh target {t!r} is not a string")
    user, at, host = t.rpartition("@")
    check_host(host, "ssh host")
    if at:
        check_user(user)
    return t

_ALLOWED = {
    "ssh": {"user"},
    "registry": {"path"},
    "hosts": None,                        # a table of tables; checked below
    "card_keys": {"index_prefixes", "name_prefixes"},
    "capacity": {"reader"},
    "patches": {"hosts"},
}
_HOST_KEYS = {"address", "user"}


class SiteError(Exception):
    """The site file is missing (when named explicitly), unreadable, or says something this version does not understand."""


USER_SITE_NAME = "site.toml"


def _default_user_site_file() -> Path:
    """$XDG_CONFIG_HOME/bigbrain/site.toml, else ~/.config/bigbrain/site.toml (also on Windows)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "bigbrain" / USER_SITE_NAME


#: the per-user site file (a function so the tests can point it elsewhere)
user_site_file = _default_user_site_file
_warned: set = set()


def path() -> Optional[Path]:
    """The site file in use, or None when there is none. $BIGBRAIN_SITE is returned even when it does not exist
    (load() then says so); otherwise the per-user file when it exists.

    A site.toml in the CURRENT directory is never read implicitly (v0.2.1, security review S8): it chooses ssh users, hosts and a
    module to import, so running a command inside someone else's checkout must not pick theirs up. When one is there it is named
    on stderr, once, with the way to use it."""
    env = os.environ.get(ENV_SITE)
    if env:
        return Path(env).expanduser()
    mine = user_site_file()
    if mine.is_file():
        return mine
    here = Path.cwd() / SITE_FILE
    if here.is_file() and str(here) not in _warned:
        _warned.add(str(here))
        print(f"bigbrain: ignored {here}: a site file in the current directory is not read on its own. Use it with "
              f"`export {ENV_SITE}={here}`, or move it to {mine}", file=sys.stderr)
    return None


def _validate(cfg: dict, where: str) -> dict:
    for table in cfg:
        if table not in _ALLOWED:
            raise SiteError(f"{where}: unknown table [{table}] (known: {', '.join(sorted(_ALLOWED))})")
        if not isinstance(cfg[table], dict):
            raise SiteError(f"{where}: [{table}] must be a table")
        keys = _ALLOWED[table]
        if keys is not None:
            bad = sorted(set(cfg[table]) - keys)
            if bad:
                raise SiteError(f"{where}: unknown key(s) in [{table}]: {', '.join(bad)} (known: {', '.join(sorted(keys))})")
    for name, h in (cfg.get("hosts") or {}).items():
        if not isinstance(h, dict) or not h.get("address"):
            raise SiteError(f"{where}: [hosts.{name}] needs an `address`")
        bad = sorted(set(h) - _HOST_KEYS)
        if bad:
            raise SiteError(f"{where}: unknown key(s) in [hosts.{name}]: {', '.join(bad)} (known: address, user)")
        if not valid_host(h["address"]):
            raise SiteError(f"{where}: [hosts.{name}] address {h['address']!r} is not an IP address or a host name")
        if "user" in h and not valid_user(h["user"]):
            raise SiteError(f"{where}: [hosts.{name}] user {h['user']!r} is not a user name")
    u = (cfg.get("ssh") or {}).get("user")
    if u is not None and not valid_user(u):
        raise SiteError(f"{where}: [ssh] user {u!r} is not a user name")
    return cfg


_cache: Dict[Tuple[str, int, int], dict] = {}


def load(p: Union[str, Path, None] = None) -> dict:
    """The site file as a dict ({} when there is none). Raises SiteError naming the file."""
    p = Path(p).expanduser() if p else path()
    if p is None:
        return {}
    try:
        st = p.stat()
    except OSError:
        raise SiteError(f"site file {p} does not exist (${ENV_SITE} points at it; copy site.example.toml and edit it)") from None
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key in _cache:
        return _cache[key]
    try:
        with open(p, "rb") as f:
            cfg = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise SiteError(f"site file {p} is unreadable ({type(e).__name__}: {e})") from None
    _cache.clear()
    _cache[key] = _validate(cfg, str(p))
    return _cache[key]


def _base_dir() -> Path:
    p = path()
    return p.resolve().parent if p else Path.cwd()


# ── accessors ────────────────────────────────────────────────────────────

def ssh_user() -> str:
    """The default ssh user: [ssh] user, else the local user name (what plain `ssh host` would use)."""
    u = (load().get("ssh") or {}).get("user")
    if u:
        return str(u)
    try:
        return getpass.getuser()
    except (OSError, KeyError, ImportError):
        raise SiteError("no ssh user: set [ssh] user in site.toml (the local user name could not be read)") from None


def registry_path() -> Optional[Path]:
    """[registry] path as an absolute path (relative to the site file), or None."""
    r = (load().get("registry") or {}).get("path")
    if not r:
        return None
    q = Path(str(r)).expanduser()
    return q if q.is_absolute() else _base_dir() / q


def hosts() -> Dict[str, dict]:
    """{name: {"address", "user"}} for every [hosts.NAME]; `user` is filled from [ssh] user when the host names none."""
    out = {}
    for name, h in (load().get("hosts") or {}).items():
        out[name] = {"address": str(h["address"]), "user": str(h.get("user") or ssh_user())}
    return out


def ssh_target(name_or_target: str) -> str:
    """`user@address` for a [hosts] name; anything else (already `user@address`, or a bare address) is returned as it is, once it
    is checked to be one (SiteError otherwise: a value starting with '-' would be an ssh option)."""
    h = hosts().get(name_or_target)
    target = f"{h['user']}@{h['address']}" if h else name_or_target
    try:
        return check_ssh_target(target)
    except ValueError as e:
        raise SiteError(str(e)) from None


def names_by_address() -> Dict[str, str]:
    """{address: host name} from [hosts], for naming machines build_registry.py finds by address."""
    return {h["address"]: name for name, h in hosts().items()}


def card_key_prefixes() -> Tuple[List[str], List[str]]:
    """([index prefixes], [name prefixes]) from [card_keys]; both empty by default."""
    c = load().get("card_keys") or {}
    return [str(x) for x in c.get("index_prefixes") or []], [str(x) for x in c.get("name_prefixes") or []]


def capacity_reader() -> str:
    """[capacity] reader ("module:function"), or ''."""
    return str((load().get("capacity") or {}).get("reader") or "")


def patch_hosts_path() -> Optional[Path]:
    """[patches] hosts as an absolute path (relative to the site file), or None."""
    r = (load().get("patches") or {}).get("hosts")
    if not r:
        return None
    q = Path(str(r)).expanduser()
    return q if q.is_absolute() else _base_dir() / q
