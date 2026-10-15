# Installing BigBrain

BigBrain is a pip package (import name `bigbrain`, v0.2.0). You install it on **one controller machine** (any machine with Python and
ssh keys to your GPU PCs; it needs no GPU): that gives you the commands below. The controller copies the eight node files to each
GPU PC over ssh and starts the nodes, so the GPU PCs need nothing from this package, only a Python environment that already has
exllamav3 and torch.

```bash
python -m venv bigbrain-env && . bigbrain-env/bin/activate                    # Windows: py -m venv bigbrain-env ; bigbrain-env\Scripts\activate
pip install "git+https://github.com/dadbodnerd/bigbrain.git"                  # the controller machine: numpy + jinja2 + the commands
pip install "git+https://github.com/dadbodnerd/bigbrain.git@v0.2.0"          # the same, pinned to a release tag
bigbrain-controller --help                                                    # it works if this prints a usage text
```

| command | what it is (same as `python -m bigbrain.<module>`) |
|---|---|
| `bigbrain-controller` | `bigbrain.controller`: plan a cluster, print the node commands, and with `--execute` stage and start it over ssh |
| `bigbrain-registry` | `bigbrain.build_registry`: measure the GPU PCs over ssh and write `model_registry.json` |
| `bigbrain-merge-models` | `bigbrain.merge_models`: add newly measured models to a registry without touching the rest |
| `bigbrain-node` | `bigbrain.worker`: one pipeline node; with `--http-port` it is node 0, the OpenAI-compatible endpoint |
| `bigbrain-server` | `bigbrain.server`: node 0 as the endpoint, explicitly |
| `bigbrain-bench` | `bigbrain.bench`: the speed grid and the parity proof against a running server |
| `bigbrain-parity` | `bigbrain.parity_check`: the pipeline against a one-GPU reference |

The controller starts the nodes itself (`python -m bigbrain.worker` from the staged copy), so you normally run only `bigbrain-controller`,
`bigbrain-registry` and `bigbrain-bench` yourself. The other commands are for starting a node by hand.

## 1. Requirements

| where | what |
|---|---|
| Controller machine | Python 3.11 or newer (`tomllib` is why 3.11), `pip`, `git` if you install from a git URL, an `ssh` client, and key-based ssh login to every GPU PC. No torch, no GPU; pip brings numpy and jinja2. |
| Every GPU PC that holds a slice | Linux with NVIDIA GPUs and `nvidia-smi`; a Python environment with **exllamav3 `>=1.6.0,<1.7`**, a matching **torch** (CUDA), **numpy** and **jinja2**; a POSIX shell (the controller sends `cd ... && python -m ...` commands); `python3` on the PATH for the registry builder. The nodes in the author's cluster ran Python 3.12. `pip install bigbrain` is **not** needed there: the controller copies the files. |
| Models | An EXL3 quantisation folder (`config.json`, `*.safetensors`, tokenizer files) **on the local disk of every PC that will hold a slice**. Nothing copies weights for you. |
| Network | The PCs reach each other on TCP (ports below). A 1 GbE LAN is enough; see ARCHITECTURE.md for what crosses it. |
| Tests only | a checkout and `pip install -e ".[dev]"` (numpy, jinja2, pytest, build, twine). torch is optional: without it about a fifth of the tests skip. |

The engine imports exllamav3 internals (`exllamav3.modules.attn.prepare_for_attn`, `exllamav3.cache.recurrent_util`). That is why the version
is pinned to the 1.6 line. Do not "just upgrade" exllamav3 on a node without running `bigbrain-parity` afterwards.

An environment you already have is usually the right one: for example the venv of a working TabbyAPI install contains exllamav3, torch,
numpy and jinja2. BigBrain does not call TabbyAPI; it only borrows an interpreter.

## 2. Prepare each GPU PC

1. Find (or create) the Python environment that has exllamav3 and check it:

   ```bash
   /opt/venvs/alpha/bin/python -m pip show exllamav3 | head -3        # Version must be 1.6.x
   /opt/venvs/alpha/bin/python -c "import torch, numpy, jinja2; print(torch.cuda.device_count(), 'GPU(s)')"
   ```

   Install exllamav3 and torch the way the [exllamav3 README](https://github.com/turboderp-org/exllamav3) tells you; the right torch
   build depends on your driver and GPU generation, so this repository does not guess it for you.

   **If you want pip to bring torch and exllamav3 too** (a fresh venv on a GPU PC), the `gpu` extra asks for `torch` and
   `exllamav3>=1.6.0,<1.7`. Do not run it first on a bare machine, for two reasons that are exllamav3's and torch's, not ours:

   - the **exllamav3 package on PyPI has no prebuilt CUDA extension**. Installing it from PyPI compiles one, which needs the CUDA toolkit
     (`nvcc`) and a C++ compiler on that PC, and takes a while. The prebuilt, CUDA-matched wheels are on
     [exllamav3's GitHub releases](https://github.com/turboderp-org/exllamav3/releases) (the file name carries the exllamav3 version,
     the CUDA build, the torch version and the Python tag, for example `exllamav3-1.6.1+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl`);
   - a plain `pip install torch` may pick a build that does not match your driver. Use the index URL for your CUDA version from pytorch.org.

   So, in the node's venv, in this order:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128        # the CUDA build that matches your driver (pytorch.org)
   pip install "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.1/exllamav3-1.6.1+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl"
                                                                              # pick the wheel that matches your torch, CUDA and Python
   pip install "bigbrain[gpu] @ git+https://github.com/dadbodnerd/bigbrain.git"   # last: torch and exllamav3 are already there, so nothing is built
   ```

   The last line is optional on a node (the controller copies the engine files anyway). It is the way to get one venv that holds
   everything, for example for `bigbrain-parity` on a single GPU. The wheel's version (`1.6.1+cu128...` counts as 1.6.1) satisfies
   `exllamav3>=1.6.0,<1.7` and any installed torch satisfies `torch`, which is why pip builds and downloads neither.

2. Put the model folder on local disk, for example `/data/models/gpt-oss-120b-4.0bpw-exl3/`. The folder name becomes the model name in
   the registry. Every PC that is going to hold a slice needs its own copy (the path may differ per PC).

3. Make sure the user you will log in as can run `nvidia-smi` and write to the stage directory (default `/tmp/bigbrain`; pass
   `--stage-root` for another: give an absolute path, a leading `~` is not expanded).

4. Open the TCP ports between the PCs. With the defaults, a pipeline of N nodes uses:

   | port | on | who connects to it |
   |---|---|---|
   | `16900 + i - 1` for node *i* >= 1 (16900, 16901, ...) | node *i* | node *i-1* (the activation link) |
   | `16900 + N - 1` (the token port) | node 0 | the last node (it sends back the sampled token id) |
   | the `--serve-port` you choose (examples use 16980) | node 0 | your clients (HTTP) |

   Example for `ufw` on a PC, allowing only your LAN (`10.0.0.0/24` here; use your own subnet):

   ```bash
   sudo ufw allow from 10.0.0.0/24 to any port 16900:16920 proto tcp
   sudo ufw allow from 10.0.0.0/24 to any port 16980 proto tcp
   ```

   **Do not expose these ports beyond a trusted LAN** (see SECURITY.md): the activation and token links have no authentication.

## 3. Set up the controller machine

Install the package (the commands at the top of this file). To work on the code instead, or to run the tests, use a checkout:

```bash
git clone https://github.com/dadbodnerd/bigbrain
cd bigbrain
python -m pip install -e ".[dev]"                   # an editable install: numpy, jinja2, pytest, build, twine
python -m pytest -q                                 # no GPU or network needed
```

`patches/` and `deploy/` are **not** part of the installed package: they are repo tools (`deploy/release.py`, `deploy/verify_stage.py`,
`patches/verify.py`) that you run from a checkout, and they use the installed `bigbrain` if there is one, else the checkout's `src/`.
Nor does pip install the templates: `site.example.toml` and `examples/model_registry.example.json` are in the repository (and in the
source distribution), so copy them from a clone.

The controller reaches every PC with `ssh -o BatchMode=yes USER@HOST ...`, so passwordless key login must already work:

```bash
ssh-keygen -t ed25519                              # if you have no key yet
ssh-copy-id bigbrain@10.0.0.11                     # once per PC
ssh -o BatchMode=yes bigbrain@10.0.0.11 true && echo ssh ok
```

### Site configuration

Everything that belongs to your network (ssh user, host names, where the registry lives) is in one file, `site.toml`, which is git-ignored.
Copy the template and edit it; every key is optional:

```bash
cp site.example.toml site.toml
```

At minimum set `[ssh] user`. `[hosts.NAME] address = "..."` gives a PC a short name for `deploy/release.py --host NAME`. CONFIGURATION.md
describes every key. The file is found in `./site.toml` or at `$BIGBRAIN_SITE`.

### The model registry

The planner needs to know which model folder is on which PC and which GPUs each PC has. `bigbrain-registry` measures all of it over ssh
(read-only: `nvidia-smi`, the `config.json` and `.safetensors` headers of each model folder) and writes `model_registry.json`:

```bash
bigbrain-registry --write --registry model_registry.json \
    --node alpha=10.0.0.11:/data/models --node beta=10.0.0.12:/srv/models
```

`--node NAME=HOST:MODEL_ROOT` is repeatable; every sub-folder of `MODEL_ROOT` that has a `config.json` is registered as a model. Leave out
`--write` to print the result without writing it. Re-run it whenever you add a model: it fills in what is new and never discards the
fields you set by hand.

Then edit `model_registry.json` once and set, per node, **`python`**: the interpreter that has exllamav3 (for example
`/opt/venvs/alpha/bin/python`). Without it the controller starts `python`, whatever that resolves to in the remote login shell. Hand-set
fields (`python`, `user`, `aliases`, `launch`, `launch_blocker`, `link_mb_s`, `loopback_mb_s`, `notes`) survive a rebuild.
`examples/model_registry.example.json` shows every field. Point BigBrain at the file with `[registry] path` in `site.toml` (the
template already says `model_registry.json`, relative to the site file) or with `$BIGBRAIN_REGISTRY`. With a pip install and no
`site.toml`, `export BIGBRAIN_REGISTRY="$PWD/model_registry.json"` is all it takes (the ssh user then defaults to your local user name).

## 4. Plan, then start

The controller is a **dry run unless you pass `--execute`**. It prints the layer split and the exact command for every node, and it
stages nothing:

```bash
python -c "import secrets; print(secrets.token_hex(24))" > api_key       # the bearer key for the endpoint
bigbrain-controller \
    --model gpt-oss-120b-4.0bpw-exl3 --all-cards --hosts alpha,beta \
    --max-num-tokens 16384 --slots 4 \
    --serve-port 16980 --api-token "$(cat api_key)"
```

- `--model NAME` resolves each PC's model path and python from the registry. `--all-cards` uses every GPU of the `--hosts` PCs that
  have the model (or name cards yourself, in pipeline order: `--gpus alpha:0 --gpus alpha:1 --gpus beta:0`).
- The split is **by free memory** (`--split capacity`, the default with `--model`). It reads each card's free memory with `nvidia-smi`
  over ssh (read-only, also in a dry run). `--split even` cuts by layer count and reads no memory; it takes the layer count from the
  model's `config.json` (locally, else over ssh) or from `--total-blocks N`. With `--model-dir PATH` instead of `--model` there is no
  registry and `--split even` is the only choice.
- `--max-num-tokens` is the context **per slot** (a multiple of 256); `--slots` is how many sequences run at once. Both are priced into
  the plan. If the cards cannot hold the model at that size the controller refuses and says how many MiB short it is: lower the slots or
  the context, or add a card.
- A plan with several slots or speculation needs `--serve-port` (the HTTP endpoint on node 0) and a key.

When the plan looks right, add `--execute`. The controller stages the eight engine files to `--stage-root` on every PC (over ssh stdin),
starts the nodes in reverse rank order (listeners first), and then stays in the foreground, polling them. Run it in `tmux`/`screen`.
Each node writes `<stage-root>/node_<rank>.log` on its own PC. Loading a 60 GB model takes minutes; follow a log with
`ssh bigbrain@10.0.0.11 tail -f /tmp/bigbrain/node_0.log`.

For the speed features see `--spec-draft ngram` (prompt lookup, no extra model) and `--spec-draft model --spec-model DIR` (a small draft
model with the same tokenizer, on node 0's PC); CONFIGURATION.md lists the flags.

## 5. Check that it works

```bash
curl -s http://10.0.0.11:16980/health                        # 200 once every node has loaded and shaken hands; 503 before
curl -s http://10.0.0.11:16980/v1/models -H "Authorization: Bearer $(cat api_key)"
curl -s http://10.0.0.11:16980/v1/chat/completions -H "Authorization: Bearer $(cat api_key)" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Name three prime numbers."}],"max_tokens":48,"temperature":0}'
```

`docs/API.md` lists the routes. To check that the pipeline computes what a single GPU would, and to measure speed:

```bash
bigbrain-bench grid --url http://10.0.0.11:16980 --key-file api_key --concurrency 1 4 --spec none ngram
```

`bigbrain-parity` compares a running server against a one-GPU reference dump (it has `reference`, `run`, `compare` and `http`
sub-commands; read its `--help`).

## 6. Stop it

Stop node 0's process (SIGTERM): it tells the other nodes to exit and every node frees its VRAM. To signal every node of one stage
directory on a PC, print the teardown command and run it there:

```bash
python -c "from bigbrain.controller import teardown_command; print(teardown_command('/tmp/bigbrain'))"
# run the printed line on each PC:  ssh bigbrain@10.0.0.11 '<the printed line>'
```

It signals only BigBrain processes whose working directory is that stage directory, and always exits 0. Use `sig="KILL"` for a stuck node.
It finds nodes started as `python -m bigbrain.worker` / `bigbrain.server` (what the controller and `deploy/run_node.sh` do) or from a
wrapper that does `from bigbrain import worker`. A node you started through the `bigbrain-node` or `bigbrain-server` command has no
module name in its command line: stop it with SIGTERM to node 0 (above) or through whatever started it.

## What is next

OPERATIONS.md covers running it day to day (health, metrics, releases, systemd, troubleshooting), CONFIGURATION.md every setting and
flag, ARCHITECTURE.md how it works, and SECURITY.md what the key does and does not protect.
