# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
"""bigbrain/site.py: where the site file is read from, what it may say, and that every consumer (controller, registry,
build_registry, deploy/release.py, patches/verify.py) takes its site-specific values from it and from nothing else."""
import getpass
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "patches"))
needs_patches = pytest.mark.skipif(not (ROOT / "patches").is_dir(), reason="patches/ is not in an sdist (repo-only tooling)")

from bigbrain import build_registry as br  # noqa: E402
from bigbrain import controller as ctl  # noqa: E402
from bigbrain import site  # noqa: E402

FULL = """
[ssh]
user = "alice"

[registry]
path = "data/reg.json"

[hosts.alpha]
address = "10.0.0.11"

[hosts.beta]
address = "10.0.0.12"
user = "bob"

[card_keys]
index_prefixes = ["G"]
name_prefixes = ["N_"]

[capacity]
reader = "mod:fn"

[patches]
hosts = "hosts.toml"
"""


@pytest.fixture
def site_file(tmp_path, monkeypatch):
    def make(text=FULL, name="site.toml"):
        f = tmp_path / name
        f.write_text(text, encoding="utf-8")
        monkeypatch.setenv("BIGBRAIN_SITE", str(f))
        return f
    return make


# -- where the file comes from -------------------------------------------------------------------------------------------------------
def test_no_site_file_means_every_accessor_answers_its_default():
    assert site.path() is None and site.load() == {}
    assert site.ssh_user() == getpass.getuser()
    assert site.registry_path() is None and site.hosts() == {} and site.names_by_address() == {}
    assert site.card_key_prefixes() == ([], []) and site.capacity_reader() == "" and site.patch_hosts_path() is None
    assert site.ssh_target("bigbrain@10.0.0.9") == "bigbrain@10.0.0.9"        # not a name: returned as it is


def test_the_env_var_names_the_file(site_file):
    f = site_file()
    assert site.path() == f and site.ssh_user() == "alice"


def test_an_env_var_pointing_nowhere_is_an_error_that_names_the_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBRAIN_SITE", str(tmp_path / "gone.toml"))
    with pytest.raises(site.SiteError) as e:
        site.load()
    assert "BIGBRAIN_SITE" in str(e.value) and "gone.toml" in str(e.value)


def test_a_site_toml_in_the_current_directory_is_found(tmp_path, monkeypatch):
    (tmp_path / "site.toml").write_text('[ssh]\nuser = "carol"\n', encoding="utf-8")
    monkeypatch.setattr(site, "SITE_FILE", "site.toml")
    monkeypatch.chdir(tmp_path)
    assert site.path() == tmp_path / "site.toml" and site.ssh_user() == "carol"


def test_an_edited_file_is_read_again(site_file):
    f = site_file('[ssh]\nuser = "one"\n')
    assert site.ssh_user() == "one"
    f.write_text('[ssh]\nuser = "second"\n', encoding="utf-8")
    assert site.ssh_user() == "second"


# -- what it may say -----------------------------------------------------------------------------------------------------------------
def test_every_accessor_reads_its_key(site_file, tmp_path):
    site_file()
    assert site.ssh_user() == "alice"
    assert site.registry_path() == tmp_path / "data" / "reg.json"             # relative to the site file
    assert site.hosts() == {"alpha": {"address": "10.0.0.11", "user": "alice"},
                            "beta": {"address": "10.0.0.12", "user": "bob"}}
    assert site.ssh_target("alpha") == "alice@10.0.0.11" and site.ssh_target("beta") == "bob@10.0.0.12"
    assert site.names_by_address() == {"10.0.0.11": "alpha", "10.0.0.12": "beta"}
    assert site.card_key_prefixes() == (["G"], ["N_"]) and site.capacity_reader() == "mod:fn"
    assert site.patch_hosts_path() == tmp_path / "hosts.toml"


def test_an_absolute_path_is_kept(site_file, tmp_path):
    absolute = tmp_path / "elsewhere" / "r.json"
    site_file(f'[registry]\npath = "{absolute.as_posix()}"\n')
    assert site.registry_path() == absolute


@pytest.mark.parametrize("text,needle", [
    ('[sh]\nuser = "x"\n', "unknown table [sh]"),
    ('[ssh]\nusr = "x"\n', "unknown key(s) in [ssh]: usr"),
    ('[hosts.a]\nuser = "x"\n', "[hosts.a] needs an `address`"),
    ('[hosts.a]\naddress = "1.2.3.4"\nport = 22\n', "unknown key(s) in [hosts.a]: port"),
    ('[ssh]\nuser = ', "unreadable"),
])
def test_a_typo_is_an_error_not_a_silent_default(site_file, text, needle):
    site_file(text)
    with pytest.raises(site.SiteError) as e:
        site.load()
    assert needle in str(e.value)


def test_the_shipped_example_is_valid_and_uses_only_fake_values():
    cfg = site.load(ROOT / "site.example.toml")
    assert set(cfg) == {"ssh", "registry", "hosts", "card_keys", "patches"}
    assert cfg["ssh"]["user"] == "bigbrain"
    assert all(h["address"].startswith("10.0.0.") for h in cfg["hosts"].values())


# -- the consumers -------------------------------------------------------------------------------------------------------------------
def test_the_controllers_default_ssh_user_is_the_sites(site_file):
    assert ctl.NodeSpec(0, "h", 0, 0, 4, 0).user == getpass.getuser()          # no site file: the local user
    site_file('[ssh]\nuser = "alice"\n')
    assert ctl.NodeSpec(0, "h", 0, 0, 4, 0).user == "alice"
    assert ctl.GpuCapacity(host="h", gpu=0, total_mib=1, free_mib=1).user == "alice"
    plan = ctl.plan_auto("/m", [("h0", 0), ("h1", 0)], total_blocks=8)
    assert [n.user for n in plan.nodes] == ["alice", "alice"]


def test_a_node_with_its_own_user_keeps_it(site_file):
    site_file('[ssh]\nuser = "alice"\n')
    assert ctl.NodeSpec(0, "h", 0, 0, 4, 0, user="zed").user == "zed"


def test_read_model_config_asks_ssh_as_the_site_user(site_file, monkeypatch, tmp_path):
    site_file('[ssh]\nuser = "alice"\n')
    seen = []

    class R:
        stdout = '{"num_hidden_layers": 3}'

    monkeypatch.setattr(ctl.subprocess, "run", lambda argv, **kw: seen.append(argv) or R())
    assert ctl._read_model_config(str(tmp_path / "nomodel"), "10.0.0.11") == {"num_hidden_layers": 3}
    assert "alice@10.0.0.11" in seen[0]
    ctl._read_model_config(str(tmp_path / "nomodel"), "10.0.0.11", "bob")
    assert "bob@10.0.0.11" in seen[1]


def test_build_registry_names_a_pc_by_the_sites_hosts_and_falls_back_to_its_address(site_file):
    site_file()
    cfg = {"tabby": {"servers": {"S1": {"url": "http://10.0.0.11:5000", "model_dir": "/data/models"},
                                 "S2": {"url": "http://10.0.0.99:5000", "model_dir": "/data/models"}}}}
    assert [n["name"] for n in br.nodes_from_config(cfg)] == ["alpha", "10.0.0.99"]


def test_build_registry_measures_as_the_site_user(site_file, monkeypatch):
    site_file()
    seen = []

    class R:
        returncode, stdout, stderr = 0, "{}", ""

    monkeypatch.setattr(br.subprocess, "run", lambda argv, **kw: seen.append(argv) or R())
    br.measure("10.0.0.11", "/data/models")
    assert "alice@10.0.0.11" in seen[0]
    merged = br.merge({}, {"alpha": {"host": "h", "model_root": "/m", "gpus": [], "models": []}})
    assert merged["nodes"]["alpha"]["user"] == "alice"


def test_release_resolves_a_host_name_through_the_site_file(site_file, monkeypatch):
    import release
    site_file()
    seen = []

    class R:
        returncode, stdout, stderr = 0, b"INSTALLED\n", b""

    monkeypatch.setattr(release.subprocess, "run", lambda argv, **kw: seen.append(argv) or R())
    assert release.install_remote("beta", "~/bigbrain/releases", "ab" * 20, b"tar") == "INSTALLED"
    assert "bob@10.0.0.12" in seen[0]
    release.install_remote("carol@10.0.0.5", "~/bigbrain/releases", "ab" * 20, b"tar")
    assert "carol@10.0.0.5" in seen[1]


@needs_patches
def test_verify_takes_its_hosts_from_a_file_the_env_var_or_the_site(site_file, tmp_path, monkeypatch):
    import verify
    assert verify.load_pins()["hosts"] == {}
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[hosts.x]\nssh = "alpha"\n', encoding="utf-8")
    assert list(verify.load_pins(hosts_path=hosts)["hosts"]) == ["x"]
    monkeypatch.setenv("BIGBRAIN_PATCH_HOSTS", str(hosts))
    assert list(verify.load_pins()["hosts"]) == ["x"]
    monkeypatch.delenv("BIGBRAIN_PATCH_HOSTS")
    site_file()                                                                  # [patches] hosts = "hosts.toml" beside the site file
    assert list(verify.load_pins()["hosts"]) == ["x"]


@needs_patches
def test_verify_with_no_hosts_says_what_to_pass(capsys):
    import verify
    with pytest.raises(SystemExit) as e:
        verify.main(["--all"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--hosts" in err and "hosts.example.toml" in err


@needs_patches
def test_verify_resolves_an_ssh_name_through_the_site_file(site_file, tmp_path, monkeypatch):
    import verify
    site_file()
    hosts = tmp_path / "h.toml"
    hosts.write_text('[hosts.x]\nssh = "alpha"\n', encoding="utf-8")
    seen = []
    monkeypatch.setattr(verify.Runner, "reachable", lambda self: seen.append(self.ssh) or False)
    pins = verify.load_pins(hosts_path=hosts)
    verify.verify(pins, ["x"])
    assert seen == ["alice@10.0.0.11"]
