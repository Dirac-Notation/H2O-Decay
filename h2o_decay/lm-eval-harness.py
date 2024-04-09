from lm_eval import tasks, evaluator, utils
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks, include_path

from utils_hh.hh_model import hh_model

import copy
import json
import torch

# load task dataset
initialize_tasks("INFO")
task_name = "piqa"
task = tasks.get_task_dict([task_name])

model_name = "facebook/opt-2.7b"
# model_name = "gpt2"

# Load the model
lm = huggingface.HFLM(model_name)

lm.model = hh_model(model_name, version=2, heavy_ratio=0.2, recent_ratio=0.0, device=lm.device)
torch.cuda.empty_cache()
lm.model.eval().half().cuda()

# Run the evaluation
results = evaluator.evaluate(lm, task)

print(evaluator.make_table(results))
if "groups" in results:
    print(evaluator.make_table(results, "groups"))