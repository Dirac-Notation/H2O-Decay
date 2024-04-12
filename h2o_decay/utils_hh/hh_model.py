import copy

from transformers import AutoModelForCausalLM, AutoConfig

from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent

def hh_model(model_name: str,
             version: int = 1,
             heavy_ratio: float = 0.1,
             recent_ratio: float = 0.1,
             penalty: float = 0.5
            ):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    config.heavy_ratio = heavy_ratio
    config.recent_ratio = recent_ratio
    config.version = version
    config.penalty = penalty

    checkpoint = copy.deepcopy(model.state_dict())

    model = convert_kvcache_opt_heavy_recent(model, config=config)
    model.load_state_dict(checkpoint)

    return model

def reset_mask(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = reset_mask(module)

        if hasattr(module, "_reset_masks"):
            module._reset_masks()
    return model