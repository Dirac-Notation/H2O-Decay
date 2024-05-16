import argparse
import json
import os.path

from tqdm import tqdm
import torch
import copy
from copy import deepcopy
import dataclasses

import math
import matplotlib.pyplot as plt 

from rouge import Rouge
import logging
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
}

TAGET_MODULE = {
    "llama": LlamaAttention_heavy_hitter,
    "opt": OPTAttention_Mask
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_arch", type=str, default="llama")
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument("--penalty", type=float, default=0.1)
    
    parser.add_argument('--enable_h2o_cache', action='store_true')

    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args)

    model_arch = args.model_arch
    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half().eval()
    
    if args.enable_h2o_cache:
        print('Enabling H2O KV cache')
        checkpoint = copy.deepcopy(model.state_dict())
        
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        config.penalty = args.penalty
        
        if args.recent_ratio == 0.0 and args.penalty != 1.0:
            print("Enable Decay")
        
        model = ENABLE_Heavy_Hitter_FUNCTIONS[model_arch](model, config)
        model.load_state_dict(checkpoint)
        model.half().eval()

    model.to(args.device)

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
    requests = requests[:args.sample_num]

    results = []
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []

    with torch.no_grad():
        with tqdm(requests) as pbar:
            for request in pbar:
                result = {'request': request, 'result': {}}
                prompt = request['article']
                label = request['summary_gt']
                temperature = request['temperature']
                stop = request['stop']

                input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=request['max_tokens'] + len(input_ids[0]),
                    temperature=temperature,
                    top_k=args.k,
                    top_p=request['top_p'],
                    do_sample=True,
                    num_return_sequences=request['n'],
                    return_dict_in_generate=True, output_scores=True,
                )
                
                if args.enable_h2o_cache:
                    for name, m in model.named_modules():
                        if isinstance(m, TAGET_MODULE[model_arch]):
                            m._reset_masks()

                tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
                logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
                top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

                generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
                generate_text = generate_text[: generate_text.find(stop[0])]

                scores = rouge.get_scores(generate_text, label)[0]
                
                rouge1_score_list.append(scores['rouge-1']['r'])
                rouge2_score_list.append(scores['rouge-2']['r'])
                rougel_score_list.append(scores['rouge-l']['r'])

                result['result'] = {
                    "choices": [
                        {
                            "text": generate_text,
                            "logprobs": {
                                "tokens": tokens, 
                                "token_logprobs": logprobs, 
                                "top_logprobs": top_logprobs, 
                                "text_offset": []
                            }, 
                            "finish_reason": "length"
                        }
                    ], 
                    "request_time": {
                        "batch_time": 0, 
                        "batch_size": 1}
                }
                
                results.append(result)
                pbar.set_description("rouge_1: {:.3f}, rouge_2: {:.3f}, rouge_l: {:.3f}".format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))

    print("rouge_1: {:.3f}, rouge_2: {:.3f}, rouge_l: {:.3f}".format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
    
    # with open(output_path, 'w') as f:
    #     for result in results:
    #         f.write(json.dumps(result) + '\n')