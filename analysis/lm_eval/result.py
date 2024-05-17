import os
import matplotlib.pyplot as plt
import numpy as np

path = os.path.dirname(__file__)

file = open(os.path.join(path, "result.txt")).readlines()

for idx in range(56):

    data = file[6*idx:6*(idx+1)-1]

    data_list = []
    for i in range(1, 5, 1):
        tmp = [j.strip() for j in data[i].strip().split("\t")]
        tmp = list(map(float, tmp))
        data_list.append(tmp)

    title = data[0]
    dataset = title.split("|")[-1].strip()
    labels = ["Full", "Local", "H2O", "A2SF"]
    x_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
    y_values = np.array(data_list)

    plt.figure(figsize=(5, 5))
    plt.title(title)
    for i, label in enumerate(labels):
        if label == "Full":
            plt.plot(x_values, y_values[i], linestyle="dashed", label=label)
        else:
            plt.plot(x_values, y_values[i], label=label, marker="o", markersize=5)

    plt.xlabel("Cache Ratio", fontsize=10)
    plt.ylabel("Accuracy")
    plt.gca().invert_xaxis()
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    dir_path = os.path.join(path, "plot", dataset)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, f"{title}.png"))
    plt.close()