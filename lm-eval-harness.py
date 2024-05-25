import copy
import torch

from lm_eval import tasks, evaluator
from lm_eval.models import huggingface
from lm_eval.tasks import initialize_tasks

from utils_lm_eval.lm_model import lm_model

def lm_test(lm_model: huggingface.HFLM, task_list, num_fewshot):
    results = evaluator.evaluate(lm_model, tasks.get_task_dict(task_list, num_fewshot=num_fewshot, fewshot_split="validation"))

    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))
    print("==============================================")

# load task dataset
initialize_tasks()
task_list = ["openbookqa", "winogrande", "piqa", "copa", "mathqa", "arc_easy", "arc_challenge"]
model_list = ["meta-llama/Llama-2-7b-hf", "huggyllama/llama-7b", "facebook/opt-6.7b", "facebook/opt-2.7b"]
fewshot_list = [1, 0]
ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

for model_name in model_list:
    print(f"model: {model_name}")
    # Load the model
    lm = huggingface.HFLM(model_name, batch_size="2")
    
    lm.model.eval().half().cpu()
    check_point = copy.deepcopy(lm.model.state_dict())
    lm.model.cuda()

    for num_fewshot in fewshot_list:
        print(f"fewshot: {num_fewshot}")
        
        # Full Result
        print("Full")
        lm_test(lm, task_list, num_fewshot)

        for ratio in ratio_list:
            print(f"================={ratio}=================")

            config = {
                "Local": (0.0, ratio, 1.0),
                "H2O": (ratio/2, ratio/2, 1.0),
                "A2SF": (ratio, 0.0, 0.1)
            }

            for method, (select, local, penalty) in config.items():
                lm_model(model_name=model_name, lm=lm, check_point=check_point, heavy_ratio=select, recent_ratio=local, penalty=penalty)
                lm.model.eval().half().cuda()
                print(method)
                lm_test(lm, task_list, num_fewshot)

    del lm
    torch.cuda.empty_cache()