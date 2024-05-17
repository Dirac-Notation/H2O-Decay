import copy
import torch

from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_lm_eval.lm_model import lm_model

def lm_test(lm_model: huggingface.HFLM, task_dict: dict):
    results = evaluator.evaluate(lm_model, task_dict)

    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))
    print("==============================================")

# load task dataset
initialize_tasks()
task_list = ["openbookqa", "winogrande", "piqa", "copa", "mathqa", "arc_easy", "arc_challenge"]
# task_list = ["mmlu"]
model_list = ["meta-llama/Llama-2-7b-hf", "huggyllama/llama-7b", "facebook/opt-6.7b", "facebook/opt-2.7b"]
fewshot_list = [1, 0]
# ratio_list = [0.1, 0.2]
# ratio_list = [0.3, 0.4, 0.5]
# ratio_list = [0.6, 0.8]
ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


for model_name in model_list:
    print(f"model: {model_name}")
    # Load the model
    lm = huggingface.HFLM(model_name, batch_size="2")
    
    lm.model.eval().half().cpu()
    check_point = copy.deepcopy(lm.model.state_dict())
    lm.model.cuda()

    for num_fewshot in fewshot_list:
        print(f"fewshot: {num_fewshot}")
        
        # # Full Result
        # task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
        # print("Full")
        # lm_test(lm, task)

        for ratio in ratio_list:
            print(f"================={ratio}=================")

            # # local
            # task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
            # lm_model(model_name=model_name, lm=lm, check_point=check_point, heavy_ratio=0.0, recent_ratio=ratio, penalty=1.0)
            # lm.model.eval().half().cuda()
            # print("Local")
            # lm_test(lm, task)

            # # h2o
            # task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
            # lm_model(model_name=model_name, lm=lm, check_point=check_point, heavy_ratio=ratio/2, recent_ratio=ratio/2, penalty=1.0)
            # lm.model.eval().half().cuda()
            # print("H2O")
            # lm_test(lm, task)

            # decay
            task = tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation")
            lm_model(model_name=model_name, lm=lm, check_point=check_point, heavy_ratio=0.3, recent_ratio=0.0, penalty=ratio)
            lm.model.eval().half().cuda()
            print("H2O-decay")
            lm_test(lm, task)

    del lm
    torch.cuda.empty_cache()