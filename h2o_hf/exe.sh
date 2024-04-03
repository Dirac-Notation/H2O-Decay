# Step 1: Prepare inference text
task=openbookqa
shots=5
python3 -u generate_task_data.py \
  --output-file ${task}-${shots}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots}

# Step 2 (Full Cache Baseline): Generate the output from LLaMA-7b with Full Cache
model=huggyllama/llama-7b
model_arch=llama
python3 -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch}
  
# Step 2 ("Local" Baseline): Generate the output from LLaMA-7b with 20% kv of the most recent tokens
model=huggyllama/llama-7b
model_arch=llama
python3 -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0 \
  --recent_ratio 0.2

# Step 2 (H2O): Generate the output from LLaMA-7b with H2O
model=huggyllama/llama-7b
model_arch=llama
python3 -u run_lm_eval_harness.py \
  --input-path ${task}-${shots}.jsonl \
  --output-path ${task}-${shots}-${model_arch}.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1

# Step 3: Evaluate the performance of generated text
python3 -u evaluate_task_result.py \
  --result-file ${task}-${shots}-${model_arch}.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch}