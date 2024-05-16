task="xsum"
shots=$1
method=$2
HEAVY_RATIO=$3
RECENT_RATIO=$4
PENALTY=$5

if [[ ${method} == 'h2o' ]]; then
    CUDA_VISIBLE_DEVICES=1 python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_h2o_hh${1}_local${2}.jsonl \
        --model_arch llama \
        --model_name huggyllama/llama-7b \
        --heavy_ratio ${HEAVY_RATIO} \
        --recent_ratio ${RECENT_RATIO} \
        --penalty ${PENALTY} \
        --enable_h2o_cache
elif [[ ${method} == 'full' ]]; then
    CUDA_VISIBLE_DEVICES=1 python -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_full.jsonl \
        --model_arch llama \
        --model_name huggyllama/llama-7b
else
    echo 'unknown argument for method'
fi
