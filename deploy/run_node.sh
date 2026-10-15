#!/bin/sh
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 dadbodnerd
# deploy/run_node.sh -- start one BigBrain node from its stage dir. Used by bigbrain-node@.service and by hand.
#
#   run_node.sh <stage-dir>
#
# It always starts `-m bigbrain.worker`, node 0 included: worker.main hands any command line that contains --http-port
# to server.main, so node 0 comes out as the OpenAI endpoint. There is no `server` mode here: the controller's teardown
# is written for the worker module.
#
# The stage dir holds the staged package (bigbrain/*.py, as the controller or your own launch script put it there; a BB_PYTHON
# environment that already has `pip install bigbrain` needs no staged copy) and two small files written by whoever starts the run:
#   node.args   the node's command line, one line, e.g.
#               --model-dir /data/models/X --device cuda:0 --slice 0 10 --rank 0 --total 4 --http-port 16980 --api-key-file api_key ...
#               Never put a key in it: use --api-key-file api_key (a 0600 file in the stage dir).
#   node.env    optional, sourced as shell: BB_PYTHON=/path/to/venv/bin/python (required), and any other variable the node needs
#               (CUDA_VISIBLE_DEVICES=...). Not in git (the .gitignore has node.env).
#
# `cd <stage>` first, then exec: the process's working directory is the stage dir, which is how the controller's teardown finds
# a node (it signals every process whose cwd is the stage dir). `exec` so the unit's MainPID is python itself and a stop is a
# TERM to it, not to a shell.
set -eu
stage=${1:?usage: run_node.sh <stage-dir>}
[ "$#" -eq 1 ] || { echo "usage: run_node.sh <stage-dir>   (the node's own arguments are in <stage-dir>/node.args)" >&2; exit 64; }
cd "$stage"
[ -f node.args ] || { echo "run_node.sh: $stage/node.args is missing" >&2; exit 66; }
if [ -f node.env ]; then
    # shellcheck disable=SC1091
    . ./node.env
fi
: "${BB_PYTHON:?run_node.sh: set BB_PYTHON (the venv python that has exllamav3) in $stage/node.env}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# shellcheck disable=SC2046
exec "$BB_PYTHON" -m bigbrain.worker $(cat node.args)
