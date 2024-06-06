import copy
import torch

from typing import Union

from transformers import AutoModelForCausalLM, AutoConfig

from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent
from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent
from lm_eval.models.huggingface import HFLM

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "OPTForCausalLM": convert_kvcache_opt_heavy_recent,
    "LlamaForCausalLM": convert_kvcache_llama_heavy_recent
}

def lm_model(model_name: str,
             lm: HFLM=None,
             check_point=None,
             device="cpu",
             heavy_ratio: float = 0.1,
             recent_ratio: float = 0.1,
             penalty: float = 1.0,
             penalty_mode: bool=True
            ):
    config = AutoConfig.from_pretrained(model_name)
    config.heavy_ratio = heavy_ratio
    config.recent_ratio = recent_ratio
    config.penalty = penalty
    config.penalty_mode = penalty_mode

    lm.model.cpu()

    ENABLE_Heavy_Hitter_FUNCTIONS[config.architectures[0]](lm.model, config=config)

    lm.model.load_state_dict(check_point)
    
    lm.model.eval().half().to(device)