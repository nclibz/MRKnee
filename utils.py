import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch


def show_batch(img_tens):
    inp = img_tens.squeeze(1).numpy()
    fig = plt.figure(figsize=(20, 20))
    n_imgs = inp.shape[0]
    n_rows = np.ceil(n_imgs / 6)
    for i in range(n_imgs):
        fig.add_subplot(n_rows, 6, i+1)
        plt.imshow(inp[i, :, :], cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


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
