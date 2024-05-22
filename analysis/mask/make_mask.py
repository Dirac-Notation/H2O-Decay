import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

dir_path = os.path.dirname(__file__)

for dataset in ["mathqa"]:#"winogrande", "piqa", "openbookqa", "arc_e"]:

    for layer in tqdm(range(32)):

        for method in ["no_pruning", "h2o", "local", "ideal", "a2sf_010", "a2sf_050"]:
            file_path = os.path.join(dir_path, dataset, method, f"{layer}.npy")
            result_path = os.path.join(dir_path, dataset, "mask", str(layer), method)
            
            tensor = np.load(file_path).squeeze(0).astype(float)

            if method == "local":
                tensor = np.triu(tensor, -(np.count_nonzero(tensor[0,-1])-2))

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            for ln in range(32):
                tmp = np.cbrt(tensor[ln])

                ones = np.ones_like(tmp)
                ones = np.triu(ones, 1)

                tmp += ones*tmp

                plt.figure(figsize=(10,10))
                plt.xticks([])
                plt.yticks([])
                plt.imshow(tmp, cmap="Blues")
                plt.tight_layout()
                plt.savefig(os.path.join(result_path, f"test_{ln}.png"))
                plt.close()