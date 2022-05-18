import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from ipywidgets import interact, Dropdown, IntSlider
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from torch.utils.data.dataloader import DataLoader


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Everything seeded with {seed}")


def show_batch(imgs, from_dl=False):
    if from_dl:
        imgs = imgs.squeeze(1).numpy()

    fig = plt.figure(figsize=(30, 30))
    n_imgs = imgs.shape[0]
    n_rows = int(np.ceil(n_imgs / 6))
    for i in range(n_imgs):
        fig.add_subplot(n_rows, 6, i + 1)
        plt.imshow(imgs[i, :, :], cmap="gray")
        plt.axis("off")
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


################# KNEEPLOT ###############


def load_one_stack(case, data_path=None, plane="coronal"):
    fpath = "{}/{}/{}.npy".format(data_path, plane, case)
    return np.load(fpath)


def load_stacks(case, data_path=None):
    x = {}
    planes = ["coronal", "sagittal", "axial"]
    for i, plane in enumerate(planes):
        x[plane] = load_one_stack(case, plane=plane, data_path=data_path)
    return x


def load_cases(train=True):
    if train:
        case_list = pd.read_csv(
            "data/train-acl.csv",
            names=["case", "label"],
            header=None,
            dtype={"case": str, "label": np.int64},
        )["case"].tolist()
        data_path = "data/train"
    else:
        case_list = pd.read_csv(
            "data/valid-acl.csv",
            names=["case", "label"],
            header=None,
            dtype={"case": str, "label": np.int64},
        )["case"].tolist()
        data_path = "data/valid"
    cases = {}

    for case in case_list:
        x = load_stacks(case, data_path)
        cases[case] = x
    return cases


class KneePlot:
    def __init__(self, cases, figsize=(15, 5)):
        self.cases = cases

        self.planes = {case: ["coronal", "sagittal", "axial"] for case in self.cases}

        self.slice_nums = {}
        for case in self.cases:
            self.slice_nums[case] = {}
            for plane in ["coronal", "sagittal", "axial"]:
                self.slice_nums[case][plane] = self.cases[case][plane].shape[0]

        self.figsize = figsize

    def _plot_slices(self, case, im_slice_coronal, im_slice_sagittal, im_slice_axial):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)

        ax1.imshow(self.cases[case]["coronal"][im_slice_coronal, :, :], "gray")
        ax1.set_title(f"MRI slice {im_slice_coronal} on coronal plane")

        ax2.imshow(self.cases[case]["sagittal"][im_slice_sagittal, :, :], "gray")
        ax2.set_title(f"MRI slice {im_slice_sagittal} on sagittal plane")

        ax3.imshow(self.cases[case]["axial"][im_slice_axial, :, :], "gray")
        ax3.set_title(f"MRI slice {im_slice_axial} on axial plane")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def draw(self):
        case_widget = Dropdown(options=list(self.cases.keys()), description="Case")
        case_init = list(self.cases.keys())[0]

        slice_init_coronal = self.slice_nums[case_init]["coronal"] - 1
        slices_widget_coronal = IntSlider(
            min=0,
            max=slice_init_coronal,
            value=slice_init_coronal // 2,
            description="Coronal",
        )

        slice_init_sagittal = self.slice_nums[case_init]["sagittal"] - 1
        slices_widget_sagittal = IntSlider(
            min=0,
            max=slice_init_sagittal,
            value=slice_init_sagittal // 2,
            description="Sagittal",
        )

        slice_init_axial = self.slice_nums[case_init]["axial"] - 1
        slices_widget_axial = IntSlider(
            min=0,
            max=slice_init_axial,
            value=slice_init_axial // 2,
            description="Axial",
        )

        def update_slices_widget(*args):
            slices_widget_coronal.max = self.slice_nums[case_widget.value]["coronal"] - 1
            slices_widget_coronal.value = slices_widget_coronal.max // 2

            slices_widget_sagittal.max = self.slice_nums[case_widget.value]["sagittal"] - 1
            slices_widget_sagittal.value = slices_widget_sagittal.max // 2

            slices_widget_axial.max = self.slice_nums[case_widget.value]["axial"] - 1
            slices_widget_axial.value = slices_widget_axial.max // 2

        case_widget.observe(update_slices_widget, "value")
        interact(
            self._plot_slices,
            case=case_widget,
            im_slice_coronal=slices_widget_coronal,
            im_slice_sagittal=slices_widget_sagittal,
            im_slice_axial=slices_widget_axial,
        )

    def resize(self, figsize):
        self.figsize = figsize
