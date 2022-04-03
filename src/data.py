# %%
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing import Any

import albumentations as A
import numpy as np
from numpy.random import default_rng


# %%

# %%

# %%
class DS(ABC, Dataset):
    """ABC for datasets"""

    def __init__(
        self,
        datadir,
        stage,
        diagnosis,
        plane,
        clean,
        transforms,
        imgs_in_ram=False,
    ) -> None:
        self.stage = stage
        self.datadir = datadir
        self.plane = plane
        self.diagnosis = diagnosis
        self.clean = clean
        self.ids = None
        self.lbls = None
        self.weight = None
        self.img_dir = None
        self.imgs_in_ram = imgs_in_ram
        self.train_imgsize = None
        self.test_imgsize = None
        self.transforms = transforms.set_transforms(stage, plane)

    @abstractmethod
    def get_cases(self, path: str) -> Tuple[List[str], List[int]]:
        """Read metadata and return tuple with list of ids and lbls"""
        pass

    def calculate_weights(self, lbls: List[int]) -> Tensor:
        """calculates lbl weights"""
        pos_count = np.sum(lbls)
        neg_count = len(lbls) - pos_count
        return torch.as_tensor(neg_count / pos_count, dtype=torch.float32).unsqueeze(0)

    def load_npy_img(self, img_dir, id):
        """loads npy img"""
        path = os.path.join(img_dir, id + ".npy")
        imgs = np.load(path)
        return imgs

    def __getitem__(self, idx):
        label = self.lbls[idx]
        label = torch.as_tensor(label, dtype=torch.float32).unsqueeze(0)
        id = self.ids[idx]
        if self.imgs_in_ram:  # if imgs are already loaded in ram
            imgs = id
        else:
            imgs = self.load_npy_img(self.img_dir, id)

        # Rescale intensities to range between 0 and 255 -> tror ikke den gør noget!
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255
        imgs = imgs.astype(np.uint8)

        imgs = self.transforms(imgs)

        imgs = torch.from_numpy(imgs).float()
        imgs = imgs.unsqueeze(1)  # add channel

        return imgs, label, id, self.weight

    def __len__(self):
        return len(self.lbls)


# %%
class MRNet(DS):
    """MRNet dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_dir = os.path.join(self.datadir, self.stage, self.plane)

        exclude = {
            "train": {
                "sagittal": [
                    "0003",
                    "0275",
                    "0544",
                    "0582",
                    "0665",
                    "0776",
                    "0795",
                    "0864",
                    "1043",
                ],
                "axial": ["0665", "1043"],
                "coronal": ["0310", "0544", "0610", "0665", "1010", "1043"],
            },
            "valid": {"sagittal": ["1159", "1230"], "axial": ["1136"], "coronal": []},
        }

        self.exclusions = exclude[self.stage][self.plane] if self.clean else None

        # get cases
        path_metadata = f"{self.datadir}/{self.stage}-{self.diagnosis}.csv"
        self.ids, self.lbls = self.get_cases(path_metadata)

        # Calculate weights
        self.weight = self.calculate_weights(self.lbls)

    def get_cases(self, path):
        """load metadata and return tupple with list of ids and list of lbls"""
        cases = pd.read_csv(
            path, header=None, names=["id", "lbl"], dtype={"id": str, "lbl": np.int64}
        )

        # Exclude cases
        if self.stage == "train" and self.exclusions:
            cases = cases[~cases["id"].isin(self.exclusions)]

        ids = cases["id"].tolist()
        lbls = cases["lbl"].tolist()

        return ids, lbls


class KneeMRI(DS):
    """Stajdur kneemri dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_dir = os.path.join(self.datadir, "imgs")
        path_metadata = os.path.join(self.datadir, "metadata.csv")
        self.ids, self.lbls = self.get_cases(path_metadata)
        self.weight = self.calculate_weights(self.lbls)
        # TODO: Kan bruge 320 men nogle få er 288 -> Ekskludere?

    def get_cases(self, path: str) -> Tuple[List[str], List[int]]:
        cases = pd.read_csv(path)
        cases["ids"] = cases["volumeFilename"].str.replace(".pck", "", regex=False)
        cases["aclDiagnosis"] = cases["aclDiagnosis"].replace(2, 1)

        ids = cases["ids"].tolist()
        lbls = cases["aclDiagnosis"].tolist()

        return ids, lbls


class SkmTea(DS):
    """Stanford skm-tea dataset"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_dir = os.path.join(self.datadir, "imgs")
        path_metadata = os.path.join(self.datadir, "targets.csv")
        self.ids, self.lbls = self.get_cases(path_metadata)
        self.weight = self.calculate_weights(self.lbls)

    def get_cases(self, path: str) -> Tuple[List[str], List[int]]:
        cases = pd.read_csv(path)
        ids = cases["scan_id"].tolist()
        lbls = cases[self.diagnosis].tolist()

        return ids, lbls


class OAI(DS):
    """OAI DATASET"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_dir = os.path.join(self.datadir, "imgs")
        path_metadata = os.path.join(self.datadir, "targets.csv")
        self.ids, self.lbls = self.get_cases(path_metadata)
        self.weight = self.calculate_weights(self.lbls)

    def get_cases(self, path: str) -> Tuple[List[str], List[int]]:
        cases = pd.read_csv(path)

        if self.plane == "coronal":
            cases = cases[cases.plane == "COR"]
        elif self.plane == "sagittal":
            cases = cases[cases.plane == "SAG"]

        cases = cases.assign(
            img_id=cases.id.astype(str) + "_" + cases.side + "_" + cases.plane
        )
        lbls = cases[self.diagnosis].to_list()
        ids = cases["img_id"].to_list()

        if self.imgs_in_ram:
            ids = [self.load_npy_img(self.img_dir, id) for id in ids]

        return ids, lbls


# %%
class MRNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datadir,
        diagnosis,
        transforms,
        plane,
        clean,
        trim_train,
        num_workers=1,
        pin_memory=True,
        shuffle_train=True,
    ):
        super().__init__()
        self.diagnosis = diagnosis
        self.transforms = transforms
        self.datadir = datadir
        self.plane = plane
        self.clean = clean
        self.trim_train = trim_train
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train

        self.train_ds = MRNet(
            datadir=self.datadir,
            stage="train",
            diagnosis=self.diagnosis,
            plane=self.plane,
            clean=self.clean,
            trim=self.trim_train,
            transforms=self.transforms,
        )

        self.val_ds = MRNet(
            datadir=self.datadir,
            stage="valid",
            diagnosis=self.diagnosis,
            plane=self.plane,
            clean=self.clean,
            trim=False,
            transforms=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


# %%
