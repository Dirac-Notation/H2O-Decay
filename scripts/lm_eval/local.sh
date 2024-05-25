#!/bin/bash

## Inference, and generate output json file
task=$1
model=$2
model_arch=$3
shots=$4

python -u run_lm_eval_harness.py --input-path lm/${task}-${shots}.jsonl --output-path lm/${task}-${shots}-${model_arch}-local.jsonl --model-name ${model} --model-type ${model_arch} --heavy_ratio 0 --recent_ratio 0.2 --enable_small_cache