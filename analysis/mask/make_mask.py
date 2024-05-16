import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)
prompt_text = "Dan gathered together extra clothes and extra food in case of a disaster, but the _ got wet and went bad."

input_ids = torch.tensor([tokenizer.encode(prompt_text, add_special_tokens=True)])
xlabel = [tokenizer.decode(i) for i in input_ids[0]]
xlabel = [i.replace(" ", "") for i in xlabel]
xlabel = [f"{i}" for i in xlabel]
xticks_positions = np.arange(0, len(xlabel), 1)

for file_name in ["a2sf_010", "a2sf_050", "h2o", "local"]:

    asdf = np.load(f"analysis/mask/{file_name}.npy").squeeze(0).astype(float)

    if not os.path.exists(f"analysis/mask/{file_name}"):
        os.makedirs(f"analysis/mask/{file_name}")

    tensor_list = [asdf]

    plt.figure(figsize=(4*len(tensor_list), 4))

    for j in tensor_list:
        for ln in range(32):
            tmp = j[ln]

            tmp *= -0.5
            tmp += 0.5

            ones = np.ones_like(tmp)
            ones = np.triu(ones, 1)

            tmp += ones*tmp

            plt.figure(figsize=(10,10))
            plt.xticks(xticks_positions, xlabel, rotation=90, fontsize=20)
            plt.yticks([])
            plt.imshow(tmp, cmap="gray")
            plt.tight_layout()
            plt.savefig(f"analysis/mask/{file_name}/test_{ln}.png")
            plt.close()