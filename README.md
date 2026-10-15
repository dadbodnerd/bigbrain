# BigBrain

**Serve one model that is too big for any single GPU across several GPUs and several PCs, over plain TCP, on top of
[exllamav3](https://github.com/turboderp-org/exllamav3).**

BigBrain is a small pipeline-parallel inference engine. Each node loads a contiguous slice of the model's layers into its own VRAM;
the residual stream hops from node to node over TCP; only the sampled token id travels back. Node 0 serves an OpenAI-compatible
HTTP endpoint (`/v1/chat/completions`, `/v1/completions`, streaming) behind a bearer key. It is aimed at the home lab: a few
desktop GPUs spread over two or three PCs on an ordinary 1 GbE LAN, running a 100B-class model that none of them could hold alone.

```
 client --HTTP--> node 0 (tokenizer, embedding, layers 0..a, HTTP server)
                    |  TCP: residual stream
                    v
                  node 1 (layers a..b) --TCP--> ... --TCP--> last node (layers ..N, norm, lm_head, sampling)
                    ^                                                          |
                    +---------------- TCP: the sampled token id ---------------+
```

- **Pipeline parallelism, not tensor parallelism.** Only one activation tensor per token crosses a machine boundary, so a LAN is
  enough. Several sequences stay in flight at once (`--slots N`), so the nodes work on different sequences at the same moment.
- **Speculative decoding** for greedy requests: n-gram prompt lookup, or a small draft model on node 0.
- **A capacity planner** that cuts the layers in proportion to each card's free memory, prices the KV cache for N slots, and
  refuses a plan that does not fit instead of starting one that will run out of memory.
- **Plain-file deployment.** The engine is eight Python files that the controller copies to each node over ssh. No daemon, no
  container, no cluster manager.
- **Tested without a GPU.** The engine's logic runs against fake layers on CPU; the test suite needs no hardware.

It is **not** a TabbyAPI fork, not a tensor-parallel engine, and not a layer-offloading (NVMe streaming) engine: the whole
model stays resident in VRAM across the nodes.

## Is it for you?

Use it if you own several GPUs in more than one machine, you want to run a model larger than your biggest single machine's VRAM,
and you are comfortable with ssh, a venv with exllamav3, and reading logs. **If the model fits on one GPU, run it on one GPU**:
every extra node adds a network hop (measured below: five cards beat six).

This is v0.2.0, an early release (the first one you can `pip install`). Read [Status and limitations](#status-and-limitations) before you plan around it.

## Measured results

All numbers below were measured on the author's own cluster with BigBrain itself (exllamav3 1.6.x, no patches), serving the model
as one endpoint. Hardware is described in [Tested hardware](#tested-hardware).

| model (EXL3 quant, weights) | cards (PCs) | layer split | slots x context | aggregate tok/s | single-stream tok/s | prefill tok/s |
|---|---|---|---|---|---|---|
| gpt-oss-120b, 4.00 bpw, 63 GB | 3x RTX 3090 + RTX 5070 + RTX 3060 (2 PCs) | 9/9/9/5/4 | 8 x 16384 | **207.5** | **58.3** | 1927 |
| gpt-oss-120b, 4.00 bpw, 63 GB | 3x RTX 3090 + RTX 5070 + RTX 5070 Ti (3 PCs, the Ti is a Windows node behind an ssh tunnel) | 9/9/9/5/4 | 8 x 16384 | 205.1 | 43.2 | 1526 |
| gpt-oss-120b, 4.00 bpw, 63 GB | all six: 3x RTX 3090 + RTX 5070 + RTX 3060 + RTX 5070 Ti (3 PCs) | 8/8/8/4/3/5 | 8 x 16384 | 194.6 | 39.0 | 1535 |
| gpt-oss-120b, 4.00 bpw, 63 GB | the same six cards | 8/8/8/4/3/5 | 6 x 16384 | 178.5 | 39.3 | 1517 |
| gpt-oss-120b, 4.00 bpw, 63 GB | the same six cards | 8/8/8/4/3/5 | 4 x 16384 | 151.2 | 38.7 | 1459 |
| Mistral-Small-4-119B, 4.03 bpw, 62 GB | 3x RTX 3090 + RTX 5070 (2 PCs) | 10/11/10/5 | 8 x 16384 | 220.6 (a) | 63.8 | 2437 |
| Mistral-Small-4-119B, 4.03 bpw, 62 GB | 3x RTX 3090 + RTX 5070 + RTX 5070 Ti (3 PCs) | 9/9/9/5/4 | 8 x 16384 | 187.1 | 43.3 | 1902 |
| Qwen3.8-Flash-Next, 3.05 bpw, 85 GB (+ a 32.6 GB n-gram table on node 0's disk) | 3x RTX 3090 + RTX 5070 + RTX 5070 Ti (3 PCs) | 12/12/12/6/6 | 8 x 16384 | 140.3 | 41.8 | 490 |
| Devstral-2-123B (dense, 88 layers), 4.00 bpw, 65 GB | 3x RTX 3090 + RTX 5070 + RTX 5070 Ti (3 PCs) | 20/23/24/11/10 | 4 x 12288 | 14.9 | 6.6 | 277 |

How to read it:

- *Aggregate* is completion tokens divided by wall time while the slots were kept busy with a real workload (15 requests of about
  5K prompt tokens, a code-review job; (a) is a longer run of 56 requests per pass), so it includes prefill and queueing. *Single-stream* and *prefill* come from a probe with one
  request alone (a prompt of about 7K tokens). They are measurements of one cluster, not a benchmark suite.
- The pipeline is bound by hop latency, not by compute: the cards were 40-60% busy at best, and one of the PCs has a 1 Gb/s NIC.
  More slots raise the aggregate (151 -> 179 -> 195 tok/s at 4 -> 6 -> 8 slots on six cards) at the price of per-request speed.
- Five cards beat six: an extra card is an extra hop. The Windows node reached through an ssh tunnel cost speed too (single stream
  58 vs 43 tok/s for gpt-oss-120b).
- A dense model with full attention on every layer is slow to prefill (Devstral: 277 tok/s) and needs `--watchdog-s 30` (a controller
  flag) so the 5-second link watchdog does not trip during a long prefill. It also cannot hold 8 slots of 16K on five of these cards (its KV
  cache is 2048 elements per token per layer).
- Architectures that ran: gpt-oss, Ministral3 (Devstral), Mistral4 MLA (Mistral-Small-4), the Qwen3.5/3.8 recurrent hybrids
  (GatedDeltaNet layers), and Qwen4Exp (Flash-Next: PLE layers, hyper-connection streams, an n-gram table).

## Quick start

You need Python 3.11+ on the machine you run the commands from (the *controller machine*: no GPU needed), ssh keys to the GPU PCs, and an
EXL3 model folder on each of them. [INSTALL.md](https://github.com/dadbodnerd/bigbrain/blob/main/INSTALL.md) has the full walk-through;
this is the short version.

```bash
pip install "git+https://github.com/dadbodnerd/bigbrain.git"              # or ...bigbrain.git@v0.2.0 to pin a release; installs numpy + jinja2
bigbrain-controller --help                                                # every command takes --help (list below)

bigbrain-registry --write --registry model_registry.json \
    --node alpha=10.0.0.11:/data/models --node beta=10.0.0.12:/srv/models
                                                                          # measures the PCs over ssh (read-only) and writes the registry
export BIGBRAIN_REGISTRY="$PWD/model_registry.json"                       # or [registry] path in a site.toml (copy site.example.toml from a clone)
python -c "import secrets; print(secrets.token_hex(24))" > api_key        # the bearer key; keep this file private

# plan and print the exact commands, touch nothing (the default is a dry run):
bigbrain-controller --model gpt-oss-120b-4.0bpw-exl3 --all-cards --hosts alpha,beta \
    --max-num-tokens 16384 --slots 4 --serve-port 16980 --api-token "$(cat api_key)"
# add --execute to stage the engine on the PCs and start the nodes
```

The commands: `bigbrain-controller` (plan a cluster, stage and start it), `bigbrain-registry` (measure the PCs, write the registry),
`bigbrain-merge-models` (add models to a registry), `bigbrain-bench` (speed grid and parity proof), `bigbrain-parity` (pipeline vs one GPU),
`bigbrain-node` and `bigbrain-server` (one node by hand; the controller starts them for you). Each is also `python -m bigbrain.<module>`.

Install variants:

| you want | command |
|---|---|
| the controller and the tools (what the quick start uses) | `pip install "git+https://github.com/dadbodnerd/bigbrain.git"` |
| a pinned release | `pip install "git+https://github.com/dadbodnerd/bigbrain.git@v0.2.0"` |
| torch + exllamav3 in the same venv (a GPU PC) | `pip install "bigbrain[gpu] @ git+https://github.com/dadbodnerd/bigbrain.git"`, **after** installing torch and an exllamav3 wheel (below) |
| the code and the tests | `git clone https://github.com/dadbodnerd/bigbrain && cd bigbrain && pip install -e ".[dev]" && python -m pytest -q` |

**The `gpu` extra and exllamav3.** The GPU PCs do not need this package: the controller copies the engine files to them. They need an
environment with exllamav3 (`>=1.6.0,<1.7`) and torch. The exllamav3 package on PyPI contains **no prebuilt CUDA extension**, so `pip install
exllamav3` compiles one and needs the CUDA toolkit and a C++ compiler; the prebuilt wheels are on
[exllamav3's GitHub releases](https://github.com/turboderp-org/exllamav3/releases) and must match your torch, CUDA and Python. Install torch
(from pytorch.org, for your driver), then the matching exllamav3 wheel, and only then the `gpu` extra, which finds both already there. The
commands are in INSTALL.md section 2.

(`--all-cards` with `--model` cuts the layers in proportion to each card's free memory, which it reads with `nvidia-smi` over ssh,
also in a dry run. `examples/model_registry.example.json` and `site.example.toml` in the repository show every field with fake values. The dry run
prints the bearer key inside the node commands: see [SECURITY.md](https://github.com/dadbodnerd/bigbrain/blob/main/SECURITY.md).)

Then talk to node 0:

```bash
curl http://10.0.0.11:16980/health                    # open; 503 until every node has loaded
curl http://10.0.0.11:16980/v1/chat/completions -H "Authorization: Bearer $(cat api_key)" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

## Documentation

| file | what it covers |
|---|---|
| [INSTALL.md](https://github.com/dadbodnerd/bigbrain/blob/main/INSTALL.md) | requirements, preparing the PCs, configuring, a first run, stopping |
| [CONFIGURATION.md](https://github.com/dadbodnerd/bigbrain/blob/main/CONFIGURATION.md) | `site.toml`, the model registry, environment variables, the flags that matter |
| [OPERATIONS.md](https://github.com/dadbodnerd/bigbrain/blob/main/OPERATIONS.md) | running and stopping a cluster, health and metrics, releases, systemd, troubleshooting |
| [ARCHITECTURE.md](https://github.com/dadbodnerd/bigbrain/blob/main/ARCHITECTURE.md) | how the pipeline, wire protocol, batching, speculation and the planner work |
| [docs/API.md](https://github.com/dadbodnerd/bigbrain/blob/main/docs/API.md) | the HTTP endpoint on node 0 |
| [SECURITY.md](https://github.com/dadbodnerd/bigbrain/blob/main/SECURITY.md) | what the bearer key does and does not protect; run it on a trusted LAN only |
| [patches/README.md](https://github.com/dadbodnerd/bigbrain/blob/main/patches/README.md) | the optional patches for upstream TabbyAPI and exllamav3 (separate licences) |

## Status and limitations

**v0.2.0 works on the author's cluster and is tested without hardware; it is not polished.** In particular:

- **Installed from git, not from PyPI.** `pip install "git+https://github.com/dadbodnerd/bigbrain.git"` works and the package builds as an sdist and
  a wheel, but the name `bigbrain` is taken on PyPI by an unrelated project, so there is no `pip install bigbrain` from the index yet (a
  different distribution name would be needed; the import name would stay `bigbrain`). `patches/` and `deploy/` are repo tools, not part of
  the installed package. Until v0.2.0 the code imported as `tools.bigbrain`; that name is gone.
- **The nodes need exllamav3 1.6.x** (`>=1.6.0,<1.7`). The engine imports exllamav3 internals, so an exllamav3 upgrade can break
  it. torch and exllamav3 are only installed by the optional `gpu` extra, and only sensibly after a CUDA-matched wheel (see Quick start).
- **The model folder must exist on every node's local disk**, at the path in the registry. Nothing copies or shards weights for you.
- **Fail-stop.** If a node or a link dies, the pipeline is marked dead (`/health` answers 503 with `engine.dead`) and you restart
  it. There is no recovery or re-balancing.
- **Every node must run the same engine code.** The handshake refuses a pipeline whose nodes disagree on capabilities.
- **A Windows node works but is manual and slower.** The controller cannot start workers on Windows (no remote-exec path); you start
  the node yourself and reach it through an ssh tunnel (`--bind-host 127.0.0.1`).
- **Text-only, one model per endpoint, `n` must be 1, no logprobs.** `tools` are rendered into the chat template but tool calls are
  not parsed out of the output. Speculative decoding applies to greedy requests only.
- **Greedy output equality on real hardware is a measured property, not a proven one.** On CPU with fake layers every speculative
  and batched run is token-identical to the plain run; exllamav3's kernels are not promised batch-invariant, so `bigbrain-bench`
  and `bigbrain-parity` exist to check it on your hardware.
- **No TLS, no per-user auth, and an unauthenticated inter-node protocol.** See [SECURITY.md](https://github.com/dadbodnerd/bigbrain/blob/main/SECURITY.md).
- **`--split capacity` reads free memory with `nvidia-smi` over ssh.** Memory that another process already holds counts as used.
  You can plug in your own reader (see CONFIGURATION.md).
- **No CI yet.** The suite was last run on Windows 11, against the installed wheel and from a checkout: 566 passed, 4 skipped on Python 3.14 with torch; 456 passed, 114 skipped (the torch cases) on Python 3.11 and 3.14 without it. It has not been run on Linux for this release.

### Tested hardware

- **Engine tests:** CPU only (Windows 11, Python 3.11 and 3.14). The nodes themselves ran Python 3.12.
- **Live runs:** NVIDIA GPUs only: three RTX 3090 (24 GB), an RTX 5070 (12 GB), an RTX 3060 (12 GB) and an RTX 5070 Ti (16 GB), in
  two Linux PCs plus one Windows PC, on a 1 GbE LAN (one NIC in the pipeline is 1 Gb/s). Pipelines of 4 to 6 cards.
- **Not tested:** AMD or Apple GPUs (exllamav3 is CUDA), 10 GbE or faster links, more than six cards, more than three PCs, a
  controller running on Windows against a live cluster, and any model or architecture not listed above.

## Repository layout

```
pyproject.toml     the package: name, version 0.2.0, dependencies, the `gpu` and `dev` extras, the seven console scripts
src/bigbrain/      the package (import name `bigbrain`)
  __init__ pipe worker server engine paged spec draft     the node runtime: these eight files are the ones the controller stages
  controller registry build_registry merge_models footprint site parity_check bench      controller side: planner, registry, tools
patches/           optional patches for upstream TabbyAPI and exllamav3: scripts, pins, a read-only verifier (repo-only: not in the wheel or the sdist)
deploy/            release.py (git archive -> an immutable release dir), verify_stage.py, run_node.sh, bigbrain-node@.service (repo-only)
examples/          model_registry.example.json
tests/             the tests, fakes and fixtures
docs/              API.md
site.example.toml  the site configuration template
```

## Licence and credits

BigBrain's own code and documentation are under the [MIT licence](https://github.com/dadbodnerd/bigbrain/blob/main/LICENSE). The files under `patches/tabbyapi/` modify TabbyAPI and are
offered under the **AGPL-3.0-or-later** (`patches/tabbyapi/LICENSE`); `patches/exllamav3/` carries exllamav3's MIT notice. No TabbyAPI or
exllamav3 source file is included here. [patches/NOTICE.md](https://github.com/dadbodnerd/bigbrain/blob/main/patches/NOTICE.md) explains which licence covers what.

BigBrain stands on [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp, which does all of the actual model
execution, and the patches are written against [TabbyAPI](https://github.com/theroyallab/tabbyAPI) by the Royal Lab. It was
written in a home lab, with AI coding assistants working under the author's direction.
