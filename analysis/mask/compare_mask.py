import os
import numpy as np
import matplotlib.pyplot as plt

def compare_mask(mask1, mask2):
    result = list()
    
    for i in range(mask1.shape[1]):
        mask_a = mask1[0,i]
        mask_b = mask2[0,i]
        
        intersection = np.count_nonzero(np.logical_and(mask_a, mask_b))/np.count_nonzero(mask_a)
        
        result.append(intersection)
    
    result = np.array(result)
    
    return np.mean(result)

mask_list = {
    "local": [],
    "h2o": [],
    "a2sf_010": [],
    "a2sf_050": [],
    "a2sf_090": []
}

for layer in range(32):
    
    dir_path = os.path.dirname(__file__)
    attention_path = os.path.join(dir_path, "ideal", f"{layer}.npy")

    ideal_mask = np.load(attention_path) # 1, 32, 25, 25
    
    for mask_name in mask_list.keys():
        mask_path = os.path.join(dir_path, mask_name, f"{layer}.npy")
        mask = np.load(mask_path) # 1, 32, 25, 25
        
        if mask_name == "local":
            mask = np.triu(mask, -9)
        
        mask_list[mask_name].append(compare_mask(ideal_mask, mask))

for a, b in mask_list.items():
    plt.plot(b, label=a)
    
    mean = sum(b)/len(b)
    print(f"{a}: {mean:.3f}")

plt.legend()
plt.savefig(os.path.join(dir_path, "Similarity.png"))