from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_hh.hh_model import hh_model

import copy
import json
import torch

# load task dataset
initialize_tasks("INFO")
task_list = ["piqa", "winogrande", "hellaswag"]
task = tasks.get_task_dict(task_list)

import pdb; pdb.set_trace()