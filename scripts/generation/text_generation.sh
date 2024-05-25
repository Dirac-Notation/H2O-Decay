#!/bin/bash

model_arch=${1:-llama}
model_name=${2:-meta-llama/Llama-2-7b-hf}
cache_ratio=${3:-0.3}
penalty=${4:-0.2}

python -u run_text_generation.py -model_arch $model_arch \
    --model_name $model_name \
    --cache_ratio $cache_ratio --penalty $penalty