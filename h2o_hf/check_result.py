import json
import numpy as np

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = load_jsonl("summary_results/xsum_0shot_full.jsonl")

for i in data:
    tmp = i["result"]["choices"][0]["logprobs"]["token_logprobs"]
    print(np.exp(-sum(tmp)/len(tmp)), end=" ")