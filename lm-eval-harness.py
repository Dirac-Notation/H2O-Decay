from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_lm_eval.lm_model import lm_model

import copy
import json
import torch

def lm_test(lm_model: huggingface.HFLM, task_dict: dict):
    results = evaluator.evaluate(lm_model, task_dict)

    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))
    print("==============================================")

# load task dataset
initialize_tasks()
task_list = ["openbookqa", "winogrande", "piqa", "copa"]
# task_list = ["mmlu"]

num_fewshot = 1

# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "huggyllama/llama-7b"
# model_name = "facebook/opt-2.7b"
# model_name ="mosaicml/mpt-7b"

# Load the model
lm = huggingface.HFLM(model_name, batch_size=2)

# # Full Result
# task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
# print("Full")
# lm_test(lm, task)

for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
    print(f"================={ratio}=================")

    # local
    task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
    lm_model(model_name=model_name, lm=lm, heavy_ratio=0.0, recent_ratio=ratio, penalty=1.0)
    lm.model.eval().half().cuda()
    print("Local")
    lm_test(lm, task)

    # h2o
    task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
    lm_model(model_name=model_name, lm=lm, heavy_ratio=ratio/2, recent_ratio=ratio/2, penalty=1.0)
    lm.model.eval().half().cuda()
    print("H2O")
    lm_test(lm, task)

    # decay
    task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
    lm_model(model_name=model_name, lm=lm, heavy_ratio=ratio, recent_ratio=0.0, penalty=0.1)
    lm.model.eval().half().cuda()
    print("h2o-decay")
    lm_test(lm, task)