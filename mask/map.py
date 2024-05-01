import numpy as np
import matplotlib.pyplot as plt
import os

def print_list(input_list: list):
    for item in input_list:
        print(item)

def strip_string(input_string: str):
    return input_string.strip()

def dot_remove(input_string: str):
    return input_string.replace(".", "")

with open("test.txt", "r") as file:
    raw_lines = file.readlines()

cleaned_lines = "\r\n".join(list(map(strip_string, raw_lines)))

grouped_lines = []
grouped_lines_text = ''

processed_lines = cleaned_lines.replace(",\r\n", ", ")
processed_lines = processed_lines.replace("tensor([", "")
processed_lines = processed_lines.replace("], device='cuda:0')", "")
processed_lines = processed_lines.replace("], device='cuda:0', dtype=torch.float16)", "")
processed_lines = processed_lines.replace("],\r\ndevice='cuda:0', dtype=torch.float16)", "")
processed_lines = processed_lines.replace(", ", "\t")
processed_lines = processed_lines.splitlines()

for index, line in enumerate(processed_lines):
    group_index = index % 32

    if len(grouped_lines) <= group_index:
        grouped_lines.append([])
    
    grouped_lines[group_index].append(line.strip())

for group in grouped_lines:
    grouped_lines_text += "\r\n".join(group)
    grouped_lines_text += "\r\n"
    grouped_lines_text += "\r\n"

path = "result/base"

# with open("result.txt", "w") as f:
#     f.write(grouped_lines_text)

if (not os.path.exists(path)):
    os.makedirs(path)

for idx in range(32):
    first_group = grouped_lines[idx]
    # first_group = grouped_lines[0]

    integer_groups = []

    for line in first_group:
        split_line = line.split("\t")
        split_line = list(map(dot_remove, split_line))
        integer_groups.append(list(map(int, split_line)))

    prompt_len = len(integer_groups[-1]) - 64

    image_array = np.zeros((64, prompt_len+64))

    for i, j in enumerate(integer_groups):
        image_array[i, :len(j)] = j

    image_array /= 2
    image_array += 0.5

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            if (i+prompt_len < j):
                image_array[i,j] = 0

    # plt.plot(image_array[:,idx])
    plt.imshow(image_array)
    plt.savefig(f"{path}/token_{idx}.png")
    plt.close()

print("prompt length : ", prompt_len)