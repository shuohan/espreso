#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)

docker run --gpus device=1 --rm -v $dir:$dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir -w $dir/tests -t \
    psf-est ./test_interp.py
