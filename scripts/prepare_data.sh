#!/bin/bash

task=$1
shots=$2

python -u generate_task_data.py --output-file lm/${task}-${shots}.jsonl --task-name ${task} --num-fewshot ${shots}
