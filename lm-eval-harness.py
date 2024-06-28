import copy
import torch
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language models on various tasks.")
    
    parser.add_argument("--task_list", nargs="+", help="List of tasks to evaluate")
    parser.add_argument("--model_list", nargs="+", help="List of model names")
    parser.add_argument("--fewshot_list", nargs="+", type=int, help="List of fewshot values")
    parser.add_argument("--ratio_list", nargs="+", type=float, help="List of ratio values")

    args = parser.parse_args()

    task_list = args.task_list
    model_list = args.model_list
    fewshot_list = args.fewshot_list
    ratio_list = args.ratio_list

    initialize_tasks()

    for num_fewshot in fewshot_list:
        print(f"fewshot: {num_fewshot}")
        
        for model_name in model_list:
            
            lm = huggingface.HFLM(model_name, batch_size="auto")
            
            lm.model.cpu()
            check_point = copy.deepcopy(lm.model.state_dict())
            lm.model.cuda()

            print(f"model: {model_name}")
            
            # print("Full")
            # lm_test(lm, task_list, num_fewshot)

            for ratio in ratio_list:
                print(f"================={ratio}=================")

                config = {
                    # "Local": (0.0, ratio, 1.0, True, False),
                    # "H2O": (ratio/2, ratio/2, 1.0, True, False),
                    # "A2SF_ZERO": (ratio, 0.00, 0.1, True, False),
                    # "A2SF_RECENT": (ratio-0.05, 0.05, 0.1, True, False),
                    # "A2SF_TENDANCY_ZERO": (ratio, 0.00, 0.1, False, False),
                    # "A2SF_TENDANCY_RECENT": (ratio-0.05, 0.05, 0.1, False, False),
                    # "NOHIS_ZERO": (ratio, 0.00, 0.0, True, False),
                    # "NOHIS_RECENT": (ratio-0.05, 0.05, 0.0, True, False),
                    # "IDEAL": (ratio, 0.00, 1.0, True, True),
                    # "IDEAL_VALUE": (ratio, 0.00, 1.0, False, True),
                    "A2SF_DIMENSION": (ratio, 0.00, 0.1, True, False),
                }

                for method, (select, local, penalty, penalty_mode, ideal) in config.items():
                    lm_model(
                        model_name=model_name,
                        lm=lm,
                        check_point=check_point,
                        device="cuda",
                        heavy_ratio=select,
                        recent_ratio=local,
                        penalty=penalty,
                        penalty_mode=penalty_mode,
                        ideal=ideal
                    )
                    print(method)
                    lm_test(lm, task_list, num_fewshot)
            
            lm.model.cpu()