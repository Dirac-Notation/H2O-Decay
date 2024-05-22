#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models
"""


import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from rouge import Rouge

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_arch", type=str, default='llama')
    parser.add_argument("--model_name", type=str, default='huggyllama/llama-7b')

    parser.add_argument("--cache_ratio", type=float, default=0.2)
    parser.add_argument("--penalty", type=float, default=0.1)

    parser.add_argument("--length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.warning(f"device: {args.device}")
    set_seed(args)

    rogue = Rouge()

    # Change to your custom prompt text
    # prompt_text = 'In the year 2087, humanity has achieved remarkable technological advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecological restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance.'
    prompt_text = 'In a small, bustling cafe nestled in the heart of a vibrant city, a serendipitous event unfolded, leaving a lasting impression on all who witnessed it. As the patrons sat sipping their coffees and engaging in animated conversations, a talented street musician entered the cafe, carrying a weathered guitar and radiating an aura of creativity.'

    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    ######## Generate with Full Cache
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half().eval()
    
    check_point = copy.deepcopy(model.state_dict())

    # input_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(model.device)
    input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)

    model.to(args.device)
    generate_ids = model.generate(input_ids, max_new_tokens=args.length)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result = result.replace(prompt_text, "")
    print("################## Generated Context with Full Cache ###################")
    print(result)
    print()

    ######### Enable HH
    config.heavy_ratio = args.cache_ratio/2
    config.recent_ratio = args.cache_ratio/2
    config.penalty = 1.0
    
    model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    generate_ids_hh = model.generate(input_ids, max_new_tokens=args.length)
    result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result_hh = result_hh.replace(prompt_text, "")
    print("################## Generated Context with Heavy Hitter Oracle ###################")
    print(result_hh)
    print()

    score = rogue.get_scores(result_hh, result, avg=True)

    print(score)
    print()

    ######### Enable Decay
    config.heavy_ratio = args.cache_ratio
    config.recent_ratio = 0.0
    config.penalty = args.penalty
    
    model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
    model.load_state_dict(check_point)
    model.half().eval().to(args.device)

    generate_ids_hh = model.generate(input_ids, max_new_tokens=args.length)
    result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result_hh = result_hh.replace(prompt_text, "")
    print("################## Generated Context with H2O Decay ###################")
    print(result_hh)
    print()

    score = rogue.get_scores(result_hh, result, avg=True)

    print(score)
    print()