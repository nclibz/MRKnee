import matplotlib.pyplot as plt
import numpy as np
import torch
import heapq


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


def print_top_losses(loss_dict, n):
    k_high = heapq.nlargest(n, loss_dict, key=loss_dict.get)
    print("Sample : Loss")
    for k in k_high:
        print(k, " : ", loss_dict.get(k))


def do_aug(imgs, transf):
    img_dict = {}
    target_dict = {}
    for i in range(imgs.shape[0]):
        if i == 0:
            img_dict['image'] = imgs[i, :, :]
        else:
            img_name = 'image'+f'{i}'
            img_dict[img_name] = imgs[i, :, :]
            target_dict[img_name] = 'image'

    transf.add_targets(target_dict)
    out = transf(**img_dict)
    out = list(out.values())
    return out  # returns list of np arrays
