## Inference, and generate output json file
task=$1
model=$2
model_arch=$3
shots=$4

python -u run_lm_eval_harness.py --input-path lm/${task}-${shots}.jsonl --output-path lm/${task}-${shots}-${model_arch}-h2o_basic.jsonl --model-name ${model} --model-type ${model_arch} --heavy_ratio 0.1 --recent_ratio 0.1 --penalty 1.0 --enable_small_cache
python -u run_lm_eval_harness.py --input-path lm/${task}-${shots}.jsonl --output-path lm/${task}-${shots}-${model_arch}-h2o_decay.jsonl --model-name ${model} --model-type ${model_arch} --heavy_ratio 0.2 --recent_ratio 0.0 --penalty 0.1 --enable_small_cache