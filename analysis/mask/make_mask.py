import os
import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

dir_path = os.path.dirname(__file__)

source_list = ["no_pruning", "h2o", "ideal", "A2SF_ZERO", "A2SF_TW_ZERO", "NOHIS_ZERO", "A2SF_RECENT", "A2SF_TW_RECENT", "NOHIS_RECENT"]

row = 3
column = math.ceil(len(source_list)/3)

for dataset in ["mathqa", "winogrande", "piqa", "openbookqa", "arc_e"]:
    dataset_path = os.path.join(dir_path, "npy", dataset)

    for layer in tqdm(range(32)):
        data_dict = {}
        result_path = os.path.join(dir_path, "mask", dataset, str(layer))
        
        for method in source_list:
            data_dict[method] = np.load(os.path.join(dataset_path, method, f"{layer}.npy")).squeeze(0).astype(float)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for ln in range(32):
            plt.figure(figsize=(column*7, row*7))
            
            for idx, (method, data) in enumerate(data_dict.items()):
                tmp = np.cbrt(data[ln])

                plt.subplot(row, column, idx+1)
                plt.title(method, fontsize=20)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(tmp, cmap="Blues")
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, f"test_{ln}.png"))
            plt.close()