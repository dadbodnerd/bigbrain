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
#               The arguments are split on blanks and NOT glob-expanded (no quoting: a path with a space is not supported here).
#   node.env    optional, and EXECUTED: it is sourced as shell. BB_PYTHON=/path/to/venv/bin/python (required), and any other variable
#               the node needs (`export CUDA_VISIBLE_DEVICES=...`: a plain assignment is not exported). Not in git (the .gitignore
#               has node.env).
#
# The stage dir and node.env must be owned by the user running this, must not be symlinks and must not be writable by group or
# others, or the script refuses (exit 77): `python -m` runs the staged code from the stage dir, and node.env is shell, so whoever
# can write either runs code as the node (security review M4). A private dir such as ~/.cache/bigbrain/stage (chmod 700) is right.
#
# `cd <stage>` first, then exec: the process's working directory is the stage dir, which is how the controller's teardown finds
# a node (it signals every process whose cwd is the stage dir). `exec` so the unit's MainPID is python itself and a stop is a
# TERM to it, not to a shell.
set -eu
stage=${1:?usage: run_node.sh <stage-dir>}
[ "$#" -eq 1 ] || { echo "usage: run_node.sh <stage-dir>   (the node's own arguments are in <stage-dir>/node.args)" >&2; exit 64; }
case $stage in -*) stage=./$stage ;; esac

# private_or_die PATH WHAT: refuse PATH unless it is ours, not a symlink, and not group- or world-writable
private_or_die() {
    if [ -L "$1" ] || [ ! -O "$1" ] || [ -n "$(find "$1" -prune \( -perm -020 -o -perm -002 \) -print)" ]; then
        echo "run_node.sh: refusing $2 $1: it must be owned by $(id -un), not a symlink, and not writable by group or others" >&2
        exit 77
    fi
}

[ -d "$stage" ] || { echo "run_node.sh: $stage is not a directory" >&2; exit 66; }
private_or_die "$stage" "the stage dir"
cd "$stage"
[ -f node.args ] || { echo "run_node.sh: $stage/node.args is missing" >&2; exit 66; }
if [ -e node.env ] || [ -L node.env ]; then
    private_or_die node.env "node.env"
    # shellcheck disable=SC1091
    . ./node.env
fi
: "${BB_PYTHON:?run_node.sh: set BB_PYTHON (the venv python that has exllamav3) in $stage/node.env}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
# split node.args on blanks only (the default IFS: node.env may not change it), and never glob-expand it: an argument with * or ?
# stays as it is
set -f
unset IFS
# shellcheck disable=SC2046
exec "$BB_PYTHON" -m bigbrain.worker $(cat node.args)
