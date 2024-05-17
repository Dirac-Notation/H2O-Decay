import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import os
import numpy as np

# model_name = "bert-base-uncased"
model_name = "huggyllama/llama-7b"

forget = 1.0

path = "bert"

if "bert" in model_name:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

prompt_text = "Dan gathered together extra clothes and extra food in case of a disaster, but the _ got wet and went bad."

input_ids = torch.tensor([tokenizer.encode(prompt_text)])

model.eval()

with torch.no_grad():
    if "bert" in model_name:
        outputs = model(input_ids)
    else:
        outputs = model(input_ids, output_attentions=True)

    attentions = outputs[-1]

print("Number of layers:", len(attentions), "  (from the transformer block)")
print("Number of batches:", len(attentions[0]))
print("Number of heads:", len(attentions[0][0]))
print("Number of tokens:", len(attentions[0][0][0]))
print("Number of tokens:", len(attentions[0][0][0][0]))

xlabel = [tokenizer.decode(i) for i in input_ids[0]]
xlabel = [i.replace(" ", "") for i in xlabel]
xlabel = [f"{j} {i}" for i,j in enumerate(xlabel)][::-1]

attention = torch.stack(attentions)

if "bert" not in model_name:
    mask = torch.ones_like(attention).tril(0)
    attention = mask*attention + (1-mask)*torch.finfo(attention.dtype).min

softmax = torch.nn.functional.softmax(attention, dim=-1)

if "bert" not in model_name:
    divider = torch.arange(softmax.shape[-1], 0, -1) - 1
    penalty = forget**divider
    penalty = penalty.unsqueeze(1)
    softmax *= penalty

total_score = torch.sum(softmax, dim=-2).numpy()

for i in range(len(attentions)):
    if not os.path.exists(f"tmp/{path}/{i}"):
        os.makedirs(f"tmp/{path}/{i}")
        
    for j in range(len(attentions[0][0])):
        plt.figure(figsize=(10, 10))
        scores = total_score[i][0][j][::-1]
        top5_indices = np.argsort(scores)[-5:]
        
        color = ["tab:blue"] * len(scores)
        for idx in top5_indices:
            color[idx] = "tab:red"
        
        plt.barh(xlabel, scores, color=color)
        plt.xlabel("Accumulative Attention Score", fontsize=20)
        plt.ylabel("Sequences", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        
        plt.savefig(f"tmp/{path}/{i}/{j}.png")
        plt.close()