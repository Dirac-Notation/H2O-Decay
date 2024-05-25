#!/bin/bash

## Evaluate results
task=$1
model=$2
model_arch=$3
shots=$4
method=$5

python -u evaluate_task_result.py --result-file lm/${task}-${shots}-${model_arch}-${method}.jsonl --task-name ${task} --num-fewshot ${shots} --model-type ${model_arch}