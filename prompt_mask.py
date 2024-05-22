import copy
import torch

from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_name = "huggyllama/llama-7b"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval()

prompt_text = "Dan gathered together extra clothes and extra food in case of a disaster, but the _ got wet and went bad."

config.heavy_ratio = 0.0
config.recent_ratio = 0.4
config.penalty = 1.0

check_point = copy.deepcopy(model.state_dict())
convert_kvcache_llama_heavy_recent(model, config)
model.load_state_dict(check_point)
model.half().eval().cuda()
del check_point

input_ids = tokenizer(prompt_text, add_special_tokens=True, return_tensors='pt').input_ids.cuda()

result = model(input_ids)