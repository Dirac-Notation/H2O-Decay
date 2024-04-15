from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_hh.hh_model import hh_model

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
# task_list = ["openbookqa", "winogrande", "arc_easy", "arc_challenge", "piqa"]
task_list = ["piqa"]
task = tasks.get_task_dict(task_list)

# model_name = "gpt2"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "facebook/opt-2.7b"

# Load the model
lm = huggingface.HFLM(model_name)

# # Full Result
# print("Full")
# lm_test(lm, task)

# H2O Result
hh_model(model_name=model_name, lm=lm, heavy_ratio=0.2, recent_ratio=0.0, penalty=0.2)
torch.cuda.empty_cache()
lm.model.eval().half().cuda()
print("H2O")
lm_test(lm, task)


# # Penalty Result
# for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#     hh_model(model_name=model_name, lm=lm, heavy_ratio=0.4, recent_ratio=0.0, penalty=i)
#     torch.cuda.empty_cache()
#     lm.model.eval().half().cuda()
#     print(i)
#     lm_test(lm, task)