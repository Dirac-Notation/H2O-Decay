# ## Obtain inference data
task=$1
shots=0
# python -u generate_task_data.py --output-file ${task}-${shots}.jsonl --task-name ${task} --num-fewshot ${shots}

## Inference, and generate output json file
model=$2
model_arch=$3
python -u run_lm_eval_harness.py --input-path ${task}-${shots}.jsonl --output-path ${task}-${shots}-${model_arch}-full.jsonl --model-name ${model} --model-type ${model_arch}

## Evaluate results
python -u evaluate_task_result.py --result-file ${task}-${shots}-${model_arch}-full.jsonl --task-name ${task} --num-fewshot ${shots} --model-type ${model_arch}




