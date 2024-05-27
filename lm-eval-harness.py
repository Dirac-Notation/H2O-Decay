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

    for model_name in model_list:
        # Load the model
        lm = huggingface.HFLM(model_name, batch_size="auto", device="cpu")
        
        check_point = copy.deepcopy(lm.model.state_dict())
        import pdb; pdb.set_trace()
        lm.model.cuda()

        print(f"model: {model_name}")
        for num_fewshot in fewshot_list:
            print(f"fewshot: {num_fewshot}")
            
            print("Full")
            lm_test(lm, task_list, num_fewshot)

            for ratio in ratio_list:
                print(f"================={ratio}=================")

                config = {
                    "Local": (0.0, ratio, 1.0),
                    "H2O": (ratio/2, ratio/2, 1.0),
                    "A2SF": (ratio, 0.0, 0.5)
                }

                for method, (select, local, penalty) in config.items():
                    lm_model(model_name=model_name, lm=lm, check_point=check_point, heavy_ratio=select, recent_ratio=local, penalty=penalty)
                    lm.model.eval().half().cuda()
                    print(method)
                    lm_test(lm, task_list, num_fewshot)

        lm.model.cpu()
        del lm
        torch.cuda.empty_cache()