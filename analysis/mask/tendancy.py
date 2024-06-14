import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def tendancy(vector, step=1):
    # raise len(vector.shape) != 1
    
    result = []
    for i in range(vector.shape[0]-step):
        present = vector[i]
        future = vector[i+step]
        result.append(np.abs(future - present))
    
    return result
        

dir_path = os.path.dirname(__file__)

datasets = ["arc_e", "mathqa", "openbookqa", "piqa", "winogrande"]

for dataset in tqdm(datasets):
    for layer in range(32):
        for head in range(32):
            data = np.load(os.path.join(dir_path, "npy", dataset, "no_pruning", f"{layer}.npy"))

            x = range(1, 50)
            m = []
            s = []

            for step in x:

                final = []

                for i in range(data.shape[-1]-step-1):
                    final += tendancy(data[0,head,i:,i], step)

                final = np.array(final)

                m.append(np.mean(final))
                s.append(np.std(final))

            folder_path = os.path.join(dir_path, "tendancy", dataset, str(layer))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            plt.subplot(2,1,1)
            plt.title("MEAN of Attention Score Difference Between Each Step")
            plt.xlabel("Step Difference")
            plt.ylabel("MEAN")
            plt.plot(x, m)

            plt.subplot(2,1,2)
            plt.title("STD of Attention Score Difference Between Each Step")
            plt.xlabel("Step Difference")
            plt.ylabel("STD")
            plt.plot(x, s)

            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"layer_{layer}_head_{head}.png"))
            plt.close()