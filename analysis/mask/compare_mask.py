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

layer = "first_layer"

dir_path = os.path.dirname(__file__)
dir_path = os.path.join(dir_path, layer)
attention_path = os.path.join(dir_path, "attention_scores.npy")

attention = np.load(attention_path)[0] # 1, 32, 25, 25

ideal_mask = np.zeros_like(attention, dtype=bool)
for head in range(attention.shape[-3]):
    for row in range(attention.shape[-2]):
        row_data = attention[0, head, row, :]
        largest_indices = row_data.argsort()[-10:]
        ideal_mask[0, head, row, largest_indices] = True

ideal_mask = np.tril(ideal_mask, 0)

mask_list = ["ideal", "local", "h2o", "a2sf_010", "a2sf_050"]

for mask_name in mask_list:
    if mask_name == "ideal":
        mask = ideal_mask
    else:
        mask_path = os.path.join(dir_path, f"{mask_name}.npy")
        mask = np.load(mask_path) # 1, 32, 25, 25
        
        if mask_name == "local":
            mask = np.triu(mask, -9)
    
    # masked_attention = attention * mask
    
    # attention_sum = np.sum(masked_attention, axis=-1)
    
    # average = np.mean(masked_attention)
    
    print(f"{mask_name}: {compare_mask(ideal_mask, mask)}")