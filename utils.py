import matplotlib.pyplot as plt
import torchvision
import numpy as np


def imshow(inp, title=None, plane_slice=0):
    """Imshow for Tensor."""
    inp = torchvision.utils.make_grid(inp[plane_slice].squeeze(0), nrow=6)
    inp = inp.numpy().transpose((1, 2, 0)).astype(np.int16)
    #inp = np.clip(inp, 0, 1)
    plt.imshow(inp, aspect='auto')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
