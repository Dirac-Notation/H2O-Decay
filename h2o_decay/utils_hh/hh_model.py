import copy

from transformers import AutoModelForCausalLM, AutoConfig

from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent

def hh_model(model_name: str, version: int, heavy_ratio: float, recent_ratio: float, device):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config.heavy_ratio = heavy_ratio
    config.recent_ratio = recent_ratio
    config.version = version

    checkpoint = copy.deepcopy(model.state_dict())

    model = convert_kvcache_opt_heavy_recent(model, config=config)
    model.load_state_dict(checkpoint)

    return model