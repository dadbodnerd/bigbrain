# Configuration

Four things configure BigBrain, from most to least permanent:

1. **`site.toml`**: your network (ssh user, host names, where the registry lives). Controller side. Git-ignored.
2. **The model registry** (`model_registry.json`): which model is on which PC, and each PC's GPUs. Controller side. Git-ignored.
3. **Environment variables.**
4. **Command-line flags** of the controller (the plan) and of the nodes (what the controller generates for you).

Nothing in the engine hard-codes an address, a user name or a path. With no `site.toml` at all, the ssh user is the local user name and
you pass everything else on the command line.

## 1. `site.toml`

Found in this order: `$BIGBRAIN_SITE` (a missing file there is an error, not a silent fallback), then `./site.toml` in the current
directory, otherwise there is no site file. **Every key is optional and an unknown table or key is an error**, so a typo cannot be silently
ignored. A relative path inside the file is relative to the file. `site.example.toml` is the annotated template (fake values).

```toml
[ssh]
user = "bigbrain"          # ssh user for a node whose registry entry names none. Default: the local user name.

[registry]
path = "model_registry.json"   # the model registry. $BIGBRAIN_REGISTRY, when set, wins over this.

[hosts.alpha]              # a short name for a PC. Used by deploy/release.py --host / deploy/verify_stage.py --host,
address = "10.0.0.11"      # by patches/hosts.toml `ssh = "alpha"`, and as the node name build_registry gives a PC it finds by address.
user = "bigbrain"          # optional; falls back to [ssh] user

[card_keys]                # build_registry.py only, and only with --config: how TabbyAPI server NAMES say which GPU they live on
index_prefixes = ["ALPHA_GPU"]   # a server called ALPHA_GPU1 lives on nvidia-smi index 1
name_prefixes  = ["BETA_"]       # a server called BETA_3090 lives on the GPU whose name contains "3090"

[capacity]                 # optional: replace the built-in free-memory reader behind `--split capacity`
reader = "my_probe:free_mib"     # a function `free_mib(registry, cards) -> {(host, gpu): MiB}` in a module on PYTHONPATH

[patches]
hosts = "patches/hosts.toml"     # the host list patches/verify.py checks (patches/hosts.example.toml is the template)
```

The `[capacity] reader` contract: `cards` is a list of `(host, gpu_index)` pairs exactly as the controller uses them; return a dict
with the MiB a worker may use on each card. A card missing from the answer makes the controller stop and tell you. The built-in
reader runs one read-only `nvidia-smi --query-gpu=index,memory.free` per PC over ssh and counts everything another process already
holds as used. Use your own reader when something else knows better what a card may give up, for example a scheduler that will unload
another server first.

## 2. The model registry

`model_registry.json` is the one data file the controller reads. Where it is read from, in this order: an explicit `--registry PATH` (for the
tools that take one); `$BIGBRAIN_REGISTRY`; `site.toml`'s `[registry] path`; a `model_registry.json` beside `tools/bigbrain/registry.py`;
otherwise an error that names the variable. `examples/model_registry.example.json` is a complete example. `build_registry` writes the real
one by measuring your PCs; `merge_models` adds models from an audit without touching anything else.

```
{ "version": 1,
  "nodes":  { "<node name>": { ...PC... } },
  "models": { "<model folder name>": { ...model... } } }
```

### A node (one PC)

| key | meaning |
|---|---|
| `host` | the ssh and TCP address of the PC |
| `aliases` | other names `--hosts` / `--gpus` accept for it |
| `user` | ssh user. Falls back to `[ssh] user` in `site.toml`, then the local user |
| `python` | **set this by hand**: the interpreter that has exllamav3 and torch (for example `/opt/venvs/alpha/bin/python`) |
| `model_root` | the folder that holds the model folders on this PC (what `build_registry` scanned) |
| `launch` | `false` marks a PC the controller must not start workers on (for example Windows); `launch_blocker` says why |
| `link_mb_s`, `loopback_mb_s` | the LAN speed to this PC and the same-PC hop speed in MB/s; the plan reports them as the estimated speed of each hop (it does not change the split). Defaults 110 (about 1 GbE) and 2000 |
| `gpus` | the PC's GPUs (below) |
| `nvlink` | pairs of GPU indices joined by NVLink, e.g. `[[0, 1]]` |
| `notes` | free text |

A GPU: `index` (the `nvidia-smi` index), `name`, `total_mib`, and optionally `card_key` / `card_keys`, labels for the servers (for
example a TabbyAPI instance) that live on that card. The engine only carries them through to the plan so a front end or a
custom `[capacity] reader` can use them.

The controller starts every worker with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `cuda:N` is `nvidia-smi` index N even in a box that mixes
GPU generations. Index the cards the way `nvidia-smi` does.

### A model

| key | meaning |
|---|---|
| `paths` | `{node name: model folder on that PC}`: where the model lives. A PC without an entry cannot hold a slice |
| `layers`, `layer_types`, `hidden_size`, `vocab_size`, `num_key_value_heads`, `head_dim`, `arch`, `bits` | shape facts from the model's `config.json` |
| `kv_elements_per_token` | K+V elements one token adds to the cache across the whole model (full-attention layers only), from `tools/bigbrain/footprint.py` |
| `kv_sliding_elements_per_token`, `sliding_window` | the same for sliding-window layers, which exllamav3 keeps as a small per-sequence state (gpt-oss) |
| `layer_mib`, `embed_mib`, `head_mib`, `other_mib` | measured size of each layer, the embedding, the LM head and anything else. Without them the planner estimates evenly from `size_gib` and says so |
| `size_gib` | total size of the weight files |
| `hybrid` | the model has linear-attention (recurrent) layers |
| `missing_on` | node names where an audit found the folder gone (set by `merge_models`) |

`footprint.kv_shape` is a lower bound by design: every branch returns what the architecture really caches, and a missing field answers 0
rather than a guess.

## 3. Environment variables

| variable | used by | meaning |
|---|---|---|
| `BIGBRAIN_SITE` | everything controller-side | path of the site file (else `./site.toml`) |
| `BIGBRAIN_REGISTRY` | controller, `build_registry`, `merge_models` | path of the registry; beats `site.toml` |
| `BIGBRAIN_PATCH_HOSTS` | `patches/verify.py` | path of the hosts file; beats `site.toml`'s `[patches] hosts`; `--hosts FILE` beats both |
| `BIGBRAIN_SERVERS_CONFIG` | `build_registry` | default for `--config` |
| `BIGBRAIN_API_KEY` | node 0 (`server`), `bench`, `parity_check` | the bearer key, when no `--api-key-file` / `--api-token` is given |
| `BIGBRAIN_BIND` | every node | default for `--bind-host` (the address the pipe listeners bind; default `0.0.0.0`) |
| `EXL3_DFLASH_LONGCTX_TOKENS`, `EXL3_DFLASH_LONGCTX_WINDOW` | exllamav3 *after* `patches/exllamav3/dflash2_tp.py` | context length above which DFlash2 verifies fewer drafted tokens (defaults 100000, 4; 0 = off) |

## 4. Flags

### The controller (`python -m tools.bigbrain.controller`)

Dry run unless `--execute`. Choose **one** way to say which cards: `--model NAME` with `--all-cards` (every GPU of the `--hosts` PCs that have
the model) or `--gpus HOST:GPU` repeated in pipeline order, or `--node HOST:GPU:START:END:INPORT` repeated for a fully explicit plan.

| flag | default | meaning |
|---|---|---|
| `--model NAME` / `--model-dir PATH` | | a registry model name (paths and pythons come from the registry), or one explicit path on node 0's PC |
| `--hosts LIST` | every launchable PC | comma list of node names, addresses or aliases to draw cards from |
| `--split auto\|even\|capacity` | `auto` | `capacity` cuts layers in proportion to each card's usable memory (needs `--model`); `even` by layer count; `auto` is `capacity` with `--model` |
| `--max-num-tokens` | 8192 | context **per slot**, a multiple of 256 |
| `--ctx` | `--max-num-tokens` | `--split capacity` only: the context the plan is sized for, and the nodes get it as their `--max-num-tokens` |
| `--slots` | 1 | sequences in flight at once; every node gets the same value |
| `--serve-port`, `--api-token` | | serve an OpenAI endpoint on node 0 behind this key (required for `--slots > 1` or speculation) |
| `--spec-draft none\|ngram\|model` (`--spec`) | `none` | speculative decoding for greedy requests |
| `--spec-k` | 4 | most draft tokens per round (1 to 15) |
| `--spec-model`, `--draft-device`, `--draft-mib` | | the draft model folder on node 0's PC, the CUDA device it loads on, and its VRAM (the planner reserves it) |
| `--inflight`, `--mb-size` | 0, 0 | decode steps in flight (0 = number of nodes) and sequences per step (0 = automatic) |
| `--watchdog-s` | 0 (the nodes' own default, 5) | seconds of silence on a link before a node declares its peer dead; forwarded to **every** node; at least 2. Raise to 30 for a dense model with a slow prefill, or a hop through an ssh tunnel |
| `--headroom-mib` | 1024 | free memory the planner keeps per GPU (prefill scratch, allocator slack) |
| `--node-overhead-mib` | 1024 | CUDA context, torch and exllamav3 runtime per worker process |
| `--cache-bytes` | 2.0 | KV bytes per element (2 = the FP16 cache the nodes use) |
| `--stage-root` | `/tmp/bigbrain` | the directory on each PC the engine is staged into and the nodes run from. Absolute path |
| `--python-bin` | `python` | the interpreter on the nodes when the registry gives none; `--node-python` / `--node-model` override per node |
| `--mem-fraction` | | cap each worker's CUDA memory fraction (shared GPUs) |
| `--total-blocks` | from `config.json` | the model's layer count, to avoid reading it over ssh |
| `--execute` | off | stage and launch over ssh |
| `--prompt`, `--max-new` | `Hello`, 32 | a one-shot generation instead of serving (single sequence only) |

### The nodes (`python -m tools.bigbrain.worker`; node 0 with `--http-port` is the server)

You normally never type these: the controller builds them. They matter when you start a node by hand (`deploy/run_node.sh`).

| flag | default | meaning |
|---|---|---|
| `--model-dir`, `--device`, `--slice START END`, `--rank`, `--total` | `cuda:0` | which model folder, which GPU, which layers `[START, END)`, which position in the pipeline |
| `--in-port` | | nodes >= 1: the port to listen on for the previous node |
| `--next-host`, `--next-port` | | nodes < last: where the next node listens |
| `--token-host`, `--token-port` | | the token link: node 0 listens on `--token-port`, the last node connects to `--token-host:--token-port` |
| `--max-num-tokens` | 8192 | context per slot |
| `--slots` (alias `--max-batch-size`) | 1 | sequences held at once (the larger of the two wins). **Same value on every node** |
| `--max-draft` | 0 | most drafted tokens a verify step carries; **same on every node** when speculating (set to `--spec-k` automatically when `--spec-draft` is on) |
| `--spec-draft`, `--spec-k`, `--spec-model`, `--draft-device`, `--ngram-min`, `--ngram-max` | none, 4, | speculation; node 0 acts on them, every node accepts them |
| `--prefill-chunk` | 2048 | prompt tokens per prefill hop (bounds activation memory and lets the next node start early) |
| `--watchdog-s` | 5.0 | seconds without any frame (heartbeats come every 0.5 s) before a link is declared dead; minimum 2.0. **Raise to 30** for a hop through an ssh tunnel, or when a node can compute for a long time without sending (a dense model's slow prefill) |
| `--idle-dead` | 30 | nodes >= 1: declare the upstream peer gone after this many seconds without liveness |
| `--bind-host` | `$BIGBRAIN_BIND` or `0.0.0.0` | the address the listeners bind (use `127.0.0.1` when the peer arrives through an ssh tunnel) |
| `--connect-settle` | 1.5 | seconds to watch a fresh connect for an ssh tunnel's accept-then-close (0 = off) |
| `--mem-fraction` | | cap this process's CUDA memory fraction |
| `--diag` | off | last node: always report top-1 minus top-2 logit margins (parity diagnosis) |
| `--no-async-send` | | write frames from the compute thread instead of a per-link sender thread |

Server-only flags (node 0):

| flag | default | meaning |
|---|---|---|
| `--host`, `--http-port` | `0.0.0.0`, 16890 | where the HTTP endpoint listens. (The examples in these docs use 16980.) |
| `--model-name` | the model folder's name | the id `/v1/models` reports |
| `--api-key-file`, `--api-token`, `--api-key-env` | env `BIGBRAIN_API_KEY` | the bearer key. The file is read; `--api-token` is the literal (visible in `ps`; it is what the controller passes); the env var is the fallback. With none, the server refuses to start |
| `--insecure-no-auth` | off | serve without a key (loopback tests only) |
| `--queue-size`, `--retry-after` | 8, 5 | requests that may wait behind the running ones before the answer is 503; the `Retry-After` seconds |
| `--default-max-tokens` | 0 | `max_tokens` when a request gives none (0 = up to the end of the context) |
| `--prefill-window` | 2 | prompt chunks in flight |
| `--hello-timeout` | 0 | give up the nodes' start-up handshake after this many seconds (0 = wait for as long as the nodes take to load) |
| `--token-timeout` | 600 | seconds to wait for one token from the last node, prefill included |

### Other tools

`build_registry`: `--node NAME=HOST:ROOT` (repeatable), `--config FILE`, `--registry PATH`, `--write`. `merge_models`: `--registry PATH
--update update.json [--write]`. `bench`: `grid` and `compare`. `parity_check`: `reference`, `run`, `compare`, `http`. Each has `--help`.

`tools/bigbrain/speed_flags.json` holds flag fragments (`slots`, and `draft` for `ngram` or a draft model) that a custom launch script can append to every
node's command; the controller builds the same flags itself from `--slots` and `--spec-draft`, so you only need the file if you start nodes some other way.
