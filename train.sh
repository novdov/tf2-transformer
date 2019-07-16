#!/usr/bin/env bash

PYTHONPATH="." python3 transformer/train.py \
--output_dir=$1 \
--mode=$2