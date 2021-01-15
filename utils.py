import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch


def imshow(inp, title=None, plane_slice=0):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp[plane_slice].squeeze(0), nrow=6)
    inp = inp.numpy().transpose((1, 2, 0)).astype(np.int16)
    #inp = np.clip(inp, 0, 1)
    plt.imshow(inp, aspect='auto')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


'''
bedre imshow 
plt.imshow(imgs[1,:,:], cmap='gray', vmin=0, vmax=255) 
Tager et plane som input og så evt bruge noget af det her

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
  plt.imshow(img)
  plt.axis('off')

'''


def calc_norm_data(dl, plane_int):
    sum = 0
    meansq = 0
    count = 0

    for _, data in enumerate(dl):
        data = data[0][plane_int]
        mask = data.ne(0.)
        data = data[mask]
        sum += data.sum()
        meansq = meansq + (data**2).sum()
        count += data.shape[0]

    total_mean = sum/count
    total_var = (meansq/count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    print("mean: " + str(total_mean))
    print("std: " + str(total_std))
    return total_mean, total_std
