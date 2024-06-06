from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os

file_path = os.path.dirname(__file__)

filenames = []

models = ["LLaMA 2 7B", "LLaMA 7B", "OPT 6.7B", "OPT 2.7B"]
datasets = ["piqa", "openbookqa", "arc_easy", "arc_challenge"]
# datasets = ["winogrande", "piqa", "openbookqa", "copa", "mathqa", "arc_easy", "arc_challenge"]

indices = [(2,0), (2,2), (3,0), (3,2), (0,1), (0,3), (1,1), (1,3)]

for i,j in indices:
    filenames.append(os.path.join(file_path, "plot_010", datasets[i], f"{models[j]} | 1-shot | {datasets[i]}.png"))

images = [Image.open(fname) for fname in filenames]

images = [img.resize(images[0].size) for img in images]

np_images = [np.array(img) for img in images]

merged_image = np.hstack([np.vstack(np_images[i*2:(i+1)*2]) for i in range(4)])

merged_image = Image.fromarray(merged_image)

merged_image.save(os.path.join(file_path, "Plot.png"))
