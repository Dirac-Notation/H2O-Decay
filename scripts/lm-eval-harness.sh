#!/bin/bash

CUDA=$1

CUDA_VISIBLE_DEVICES=$CUDA python lm-eval-harness.py \
    --task_list openbookqa winogrande piqa arc_easy arc_challenge \
    --model_list meta-llama/Llama-2-7b-hf huggyllama/llama-7b \
    --fewshot_list 1 0 \
    --ratio_list 0.1 0.2 0.3 0.4 0.5 0.6 0.8

# Dataset
## openbookqa winogrande piqa arc_easy arc_challenge mathqa

# Models
## meta-llama/Llama-2-7b-hf huggyllama/llama-7b facebook/opt-6.7b facebook/opt-2.7b