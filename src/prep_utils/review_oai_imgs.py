# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

targets = pd.read_csv("data/oai/targets.csv")
targets


# %%
targets.dtypes
# %%

fpaths = targets.assign(fpath="data/oai/imgs/" + targets.fname).fpath.to_list()

# %%
for fpath in fpaths:
    imgs = np.load(fpath)

    n_imgs = imgs.shape[0]
    n_cols = 6
    n_rows = int(np.ceil(n_imgs / n_cols))
    case_name = fpath.split("/")[-1]
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 30))

    for i, ax in zip(range(n_imgs), axs.flatten()):
        img = imgs[i, :, :]
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    
    fig.suptitle(case_name, fontsize = 50)
    fig.tight_layout()
    plt.show()
    input("Next?")




# %%
