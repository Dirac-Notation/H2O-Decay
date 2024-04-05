task="xsum"
shots="0"
method=$1
GPU="0"
HH_SIZE=$2
RECENT_SIZE=$3
VERSION=$4

if [[ ${method} == 'h2o' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python3 -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_h2o_hh${1}_local${2}.jsonl \
        --model_arch opt \
        --model_name facebook/opt-6.7b \
        --heavy_ratio ${HH_SIZE} \
        --recent_ratio ${RECENT_SIZE} \
        --version ${VERSION} \
        --enable_h2o_cache
elif [[ ${method} == 'full' ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python3 -u run_summarization.py \
        --input_path data/summarization_data/${task}_${shots}shot.jsonl \
        --output_path summary_results/${task}_${shots}shot_full.jsonl \
        --model_arch opt \
        --model_name facebook/opt-6.7b \
else
    echo 'unknown argment for method'
fi
