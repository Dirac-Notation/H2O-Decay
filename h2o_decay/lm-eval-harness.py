from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_hh.hh_model import hh_opt, hh_llama

import copy
import json
import torch

# load task dataset
initialize_tasks("INFO")
task_list = ["openbookqa", "winogrande", "arc_easy", "arc_challenge", "piqa"]
# task_list = ["winogrande"]
task = tasks.get_task_dict(task_list)

# model_name = "gpt2"
model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "facebook/opt-2.7b"

# Load the model
lm = huggingface.HFLM(model_name)

results = evaluator.evaluate(lm, task)

print(f"Full")
print(evaluator.make_table(results))
if "groups" in results:
    print(evaluator.make_table(results, "groups"))
print("==============================================")

lm.model = hh_llama(model_name, version=1, heavy_ratio=0.1, recent_ratio=0.1, penalty=1)
# lm.model = hh_opt(model_name, version=1, heavy_ratio=0.1, recent_ratio=0.1, penalty=1)
torch.cuda.empty_cache()
lm.model.eval().half().cuda()

# Run the evaluation
results = evaluator.evaluate(lm, task)

print(f"H2O")
print(evaluator.make_table(results))
if "groups" in results:
    print(evaluator.make_table(results, "groups"))
print("==============================================")

for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    lm.model = hh_llama(model_name, version=2, heavy_ratio=0.2, recent_ratio=0.0, penalty=i)
    # lm.model = hh_opt(model_name, version=2, heavy_ratio=0.2, recent_ratio=0.0, penalty=i)
    torch.cuda.empty_cache()
    lm.model.eval().half().cuda()

    # Run the evaluation
    results = evaluator.evaluate(lm, task)

    print(f"penalty: {i}")
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))
    print("==============================================")