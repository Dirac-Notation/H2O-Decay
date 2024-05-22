## LLaMA-7B
for dataset in piqa copa openbookqa winogrande mathqa
do
  bash scripts/lm_eval/prepare_data.sh $dataset 0

  bash scripts/lm_eval/full_cache.sh $dataset huggyllama/llama-7b llama 0
  bash scripts/lm_eval/h2o.sh $dataset huggyllama/llama-7b llama 0
  bash scripts/lm_eval/local.sh $dataset huggyllama/llama-7b llama 0

    for method in full h2o_basic h2o_decay local
    do
        bash script/lm_eval/evaluate.sh winogrande huggyllama/llama-7b llama 0 $method
    done
done