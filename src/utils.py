from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import heapq
import pandas as pd

# from ipywidgets import interact, Dropdown, IntSlider
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import albumentations as A


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


def calc_norm_data(dl, plane_int):
    sum = 0
    meansq = 0
    count = 0

    for _, data in enumerate(dl):
        data = data[0][plane_int]
        mask = data.ne(0.0)
        data = data[mask]
        sum += data.sum()
        meansq = meansq + (data ** 2).sum()
        count += data.shape[0]

    total_mean = sum / count
    total_var = (meansq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    print("mean: " + str(total_mean))
    print("std: " + str(total_std))
    return total_mean, total_std


def print_top_losses(loss_dict, n):
    k_high = heapq.nlargest(n, loss_dict, key=loss_dict.get)
    print("Sample : Loss")
    for k in k_high:
        print(k, " : ", loss_dict.get(k))


def get_preds(
    datadir="data",
    diagnosis="acl",
    stage="train",
    planes=["axial", "sagittal", "coronal"],
    ckpt_dir="models/",
    backbones=["efficientnet_b1"] * 3,
    device="cuda",
    num_workers=4,
    **kwargs,
):
    from data import MRKneeDataModule  # to prevent circular imports
    from model import MRKnee

    model_ckpts = [f"{ckpt_dir}{diagnosis}_{plane}.ckpt" for plane in planes]
    preds_dict = {}

    for plane, model_ckpt, backbone in zip(planes, model_ckpts, backbones):
        # model setup
        model = MRKnee.load_from_checkpoint(
            model_ckpt, planes=[plane], backbone=backbone, **kwargs
        )
        model.freeze()
        model = model.to(device=torch.device(device))

        if "b0" in backbone:
            img_sz = 224
        elif "b1" in backbone:
            img_sz = 240

        transf = {"train": [A.CenterCrop(img_sz, img_sz)], "valid": [A.CenterCrop(img_sz, img_sz)]}

        # data setup
        dm = MRKneeDataModule(
            datadir, diagnosis, planes=[plane], indp_normalz=True, clean=False, transf=transf
        )
        if stage == "train":
            ds = dm.train_ds
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
        elif stage == "valid":
            ds = dm.val_ds
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)

        # gather preds
        preds_list = []
        for i, batch in enumerate(dl):
            imgs, label, sample_id, weight = batch
            imgs = imgs[0].to(device=torch.device(device))
            preds = model(imgs)
            preds = torch.sigmoid(preds)
            preds_list.append(preds.item())
        preds_dict[plane] = preds_list
        if plane == planes[0]:
            preds_dict["lbls"] = [lbl for id, lbl in ds.cases]
            preds_dict["ids"] = [id for id, lbl in ds.cases]
    return pd.DataFrame(preds_dict)


def compare_clfs(clfs, X, y, X_val, y_val):
    for name, clf in clfs.items():
        clf.fit(X, y)
        cv_score = np.mean(cross_val_score(clf, X, y))
        preds = clf.predict(X_val)
        auc = roc_auc_score(y_val, preds)
        print(f"{name}: CV_SCORE: {cv_score:.4f} VAL_AUC: {auc:.4f}")


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
            min=0, max=slice_init_coronal, value=slice_init_coronal // 2, description="Coronal"
        )

        slice_init_sagittal = self.slice_nums[case_init]["sagittal"] - 1
        slices_widget_sagittal = IntSlider(
            min=0, max=slice_init_sagittal, value=slice_init_sagittal // 2, description="Sagittal"
        )

        slice_init_axial = self.slice_nums[case_init]["axial"] - 1
        slices_widget_axial = IntSlider(
            min=0, max=slice_init_axial, value=slice_init_axial // 2, description="Axial"
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
