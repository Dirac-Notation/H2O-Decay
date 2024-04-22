import torch
import matplotlib.pyplot as plt
import os

batch_num = 0

data = torch.load("scores_piqa.pt")[batch_num]

# head_num = 1

for i in range(32):
    map = data[i, :, :]
    for j in range(data.shape[-1]):
        tmp = map[j:, j]

        if not os.path.exists(f"result/{i}"):
            os.makedirs(f"result/{i}")

        plt.plot(tmp)
        plt.ylim((-0.05, 1.05))
        plt.savefig(f"result/{i}/{j}.png")
        plt.close()
