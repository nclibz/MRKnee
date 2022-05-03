# %%
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def calc_img_stats(img_paths: List[str]) -> Tuple[float, torch.Tensor]:
    nimages = 0
    mean = 0.0
    var = 0.0
    for path in tqdm(paths):
        imgs = torch.from_numpy(np.load(path)).to(torch.float32)
        # Remove channel, and flatten img pixels
        imgs = imgs.reshape(imgs.shape[0], -1)
        # Update total number of images
        nimages += imgs.shape[0]
        # Compute mean and std here
        mean += imgs.mean(1).sum()
        var += imgs.var(1).sum()

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)
    print(f"mean: {mean} std: {std}")
    return mean, std


# %%
### MRNET ####

mrnet = pd.read_csv(
    "data/mrnet/train-meniscus.csv",
    header=None,
    names=["id", "lbl"],
    dtype={"id": str, "lbl": np.int64},
)
ids = mrnet["id"].to_list()

paths = ["data/mrnet/train/sagittal/" + id + ".npy" for id in ids]

mean, std = calc_img_stats(paths)


# %%
#### OAI ###
oai = pd.read_csv("data/oai/train-meniscus.csv")

fnames = oai[oai.plane == "COR"]["fname"].to_list()

paths = ["data/oai/imgs/" + fname for fname in fnames]

mean, std = calc_img_stats(paths)


# %%


import matplotlib.pyplot as plt

imgs = np.load(paths[0])
img = imgs[5, :, :]
plt.imshow(img, cmap="gray")
print(img.mean())  # %%
# %%
