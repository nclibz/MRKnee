# %%
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch

from src.model import MRKnee

# %%

# INSTANTIATE MODEL

backbone = "efficientnet_b1"
plane = "axial"
ckpt = "src/models/acl_axial.ckpt"
img_sz = 240
device = "cpu"


model = MRKnee.load_from_checkpoint(ckpt, planes=[plane], backbone=backbone)
model.to(device=device)


# %%
# INPUT
paths = []
for root, dirs, files in os.walk(os.path.abspath("data/valid")):
    for file in files:
        if plane in root:
            paths.append(os.path.join(root, file))


path = paths[0]


# %%


def do_aug(imgs, transf):
    img_dict = {}
    target_dict = {}
    for i in range(imgs.shape[0]):
        if i == 0:
            img_dict["image"] = imgs[i, :, :]
        else:
            img_name = "image" + f"{i}"
            img_dict[img_name] = imgs[i, :, :]
            target_dict[img_name] = "image"
    transf = A.Compose(transf)
    transf.add_targets(target_dict)
    out = transf(**img_dict)
    out = list(out.values())
    return out  # returns list of np arrays


imgs = np.load(path)
imgs = do_aug(imgs, transf=[A.CenterCrop(img_sz, img_sz)])
imgs = torch.as_tensor(imgs, dtype=torch.float32)
imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255
if plane == "axial":
    MEAN, SD = 66.4869, 60.8146
elif plane == "sagittal":
    MEAN, SD = 60.0440, 48.3106
elif plane == "coronal":
    MEAN, SD = 61.9277, 64.2818
imgs = (imgs - MEAN) / SD
imgs = imgs.unsqueeze(1)  # create channel dim
imgs = imgs.unsqueeze(0)
imgs = imgs.to(device=device)


# %%

model.backbones[-1][10]
# %%
## JEG TROR DET ER DEN HER DER VIRKER
# a dict to store the activations
activation = {}


def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


h1 = model.backbones[-1][10].register_forward_hook(getActivation("2dconv"))

model.eval()
res = model(imgs)

# %%
def model_wrapper(input_batch):  # nx1x240x240
    print(input_batch.shape)
    # create batch dim
    print(input_batch.shape)
    res = model(input_batch)
    print(res.size)
    return res


model_wrapper(imgs)


# %%
from trulens.nn.models import get_model_wrapper

wrapped_model = get_model_wrapper(model, input_shape=(1, img_sz, img_sz), device=device)

from trulens.nn.attribution import InputAttribution
from trulens.nn.attribution import IntegratedGradients

infl = InputAttribution(wrapped_model)
attrs_input = infl.attributions(imgs.unsqueeze(0))


# %%
###
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz

# ImageClassifier takes a single input tensor of images Nx3x32x32,
# and returns an Nx10 tensor of class probabilities.
# It contains an attribute conv4, which is an instance of nn.conv2d,
# and the output of this layer has dimensions Nx50x8x8.
# It is the last convolution layer, which is the recommended
# use case for GuidedGradCAM.


guided_gc = GuidedGradCam(model, model.backbones[-1][10])


# Computes guided GradCAM attributions for class.
# attribution size matches input size, Nx3x32x32
attribution = guided_gc.attribute(imgs, 0)


# %%

_ = viz.visualize_image_attr(
    np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
    method="blended_heat_map",
    alpha_overlay=0.6,
)

# %%
model.backbones[-1][11]
# %%
