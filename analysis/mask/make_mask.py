import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

dir_path = os.path.dirname(__file__)

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)
prompt_text = "Dan gathered together extra clothes and extra food in case of a disaster, but the _ got wet and went bad."

input_ids = torch.tensor([tokenizer.encode(prompt_text)])
xlabel = [tokenizer.decode(i) for i in input_ids[0]]
xlabel = [i.replace(" ", "") for i in xlabel]
xlabel = [f"{i}" for i in xlabel]
xticks_positions = np.arange(0, len(xlabel), 1)

for file_name in ["a2sf_010", "a2sf_050", "h2o", "local", "ideal"]:
    
    file_path = os.path.join(dir_path, f"{file_name}.npy")
    result_path = os.path.join(dir_path, f"{file_name}")
    
    tensor = np.load(file_path).squeeze(0).astype(float)

    if file_name == "local":
        tensor = np.triu(tensor, -9)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plt.figure(figsize=(4, 4))

    for ln in range(32):
        tmp = tensor[ln]

        tmp *= -0.5
        tmp += 0.5

        ones = np.ones_like(tmp)
        ones = np.triu(ones, 1)

        tmp += ones*tmp

        plt.figure(figsize=(10,10))
        plt.xticks(xticks_positions, xlabel, rotation=90, fontsize=20)
        plt.yticks([])
        plt.imshow(tmp, cmap="Blues_r")
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, f"test_{ln}.png"))
        plt.close()