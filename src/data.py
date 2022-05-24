# %%
import pathlib
from abc import ABC, abstractmethod
from ctypes import Union
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# %%


class DataReader(ABC):
    def __init__(
        self,
        stage: str,
        diagnosis: str,
        plane: str,
        protocol: str,
        clean: bool,
        train_imgsize: Tuple[int, int],
        test_imgsize: Tuple[int, int],
        datadir: str,
        img_dir: str,
    ) -> None:
        self.stage = stage
        self.diagnosis = diagnosis
        self.plane = plane
        self.protocol = protocol
        self.clean = clean
        self.train_imgsize = train_imgsize
        self.test_imgsize = test_imgsize
        self.datadir = datadir
        self.img_dir = img_dir
        self.img_path = pathlib.Path(self.datadir, self.img_dir)
        self.stats = None

    @abstractmethod
    def get_cases(self) -> Tuple[List[str], List[int], List[pathlib.Path]]:
        """Read metadata and return tuple with list of ids, lbls, and fpaths"""

    @abstractmethod
    def get_stats(self) -> Tuple[int, int]:
        """Returns tupple with (mean, SD)"""


class DS(Dataset):
    """Generic dataset"""

    def __init__(
        self,
        datareader: DataReader,
        transforms,
        use_3d: bool = False,
        img_depth_3d: int = 32,
        dev_run_samples: Optional[int] = None,
    ) -> None:
        self.datareader = datareader
        self.ids, self.lbls, self.fpaths = self.datareader.get_cases()
        self.weight = self.calculate_weights(self.lbls)
        self.use_3d = use_3d
        self.img_depth_3d = img_depth_3d
        self.dev_run_samples = dev_run_samples
        self.transforms = (
            None if transforms is None else transforms.set_transforms(self.datareader)
        )

    def calculate_weights(self, lbls: List[int]) -> Tensor:
        """calculates lbl weights"""
        pos_count = np.sum(lbls)
        neg_count = len(lbls) - pos_count
        return torch.as_tensor(neg_count / pos_count, dtype=torch.float32).unsqueeze(0)

    def pad_depth(self, imgs, min_depth):
        d, h, w = imgs.shape
        d_pad = min_depth - d
        padding = torch.zeros(d_pad, h, w)
        return torch.concat([imgs, padding])

    def get_middle_slices(self, imgs, n_slices):
        n_imgs = imgs.shape[0]
        middle = n_imgs // 2
        left = int(middle - (n_slices / 2))
        right = int(middle + (n_slices / 2))
        return imgs[left:right, :, :]

    def __getitem__(self, idx):
        label = self.lbls[idx]
        label = torch.as_tensor(label, dtype=torch.float32).unsqueeze(0)
        id = self.ids[idx]
        fpath = self.fpaths[idx]
        imgs = np.load(fpath)

        if self.transforms is None:
            imgs = torch.Tensor(imgs).float()
        else:
            imgs = self.transforms(imgs)  # -> returns tensor

        # STANDARDIZE DEPTH SIZE IF USING 3D MODEL
        if self.use_3d and imgs.size(0) < self.img_depth_3d:
            imgs = self.pad_depth(imgs, self.img_depth_3d)
        elif self.use_3d and imgs.size(0) > self.img_depth_3d:
            imgs = self.get_middle_slices(imgs, n_slices=self.img_depth_3d)

        # add channel
        if self.use_3d:
            imgs = imgs.unsqueeze(0)
        else:
            imgs = imgs.unsqueeze(1)

        return imgs, label, id, self.weight

    def __len__(self):

        if self.dev_run_samples is None:
            return len(self.lbls)

        return self.dev_run_samples


class OAI(DataReader):
    def __init__(
        self,
        stage: str,
        diagnosis: str,
        plane: str,
        protocol: str,
        clean: bool,
        train_imgsize: Tuple[int, int] = (256, 256),
        test_imgsize: Tuple[int, int] = (256, 256),
        datadir: str = "data/oai",
        img_dir: str = "imgs",
    ) -> None:
        super().__init__(
            stage,
            diagnosis,
            plane,
            protocol,
            clean,
            train_imgsize,
            test_imgsize,
            datadir,
            img_dir,
        )
        self.stats = {
            "TSE": {"coronal": (66.51, 45.51), "sagittal": (14.69, 13.47)},
            "MPR": {"axial": (18.06, 18.91), "coronal": (23.26, 20.10)},
            "DESS": {"sagittal": (9.69, 8.03)},
        }

    def get_stats(self):
        return self.stats[self.protocol][self.plane]

    def get_cases(self):

        path = f"{self.datadir}/{self.stage}-{self.diagnosis}.csv"

        cases = pd.read_csv(path)

        if self.plane == "coronal":
            cases = cases[cases.plane == "COR"]
        elif self.plane == "sagittal":
            cases = cases[cases.plane == "SAG"]
        elif self.plane == "axial":
            cases = cases[cases.plane == "AX"]

        cases = cases[cases.protocol == self.protocol]

        ids = cases["img_id"].to_list()
        lbls = cases[self.diagnosis].to_list()
        fnames = cases["fname"].to_list()
        fpaths = [self.img_path / fname for fname in fnames]

        return ids, lbls, fpaths


# %%
class MRNet(DataReader):
    """MRNet dataset"""

    def __init__(
        self,
        stage: str,
        diagnosis: str,
        plane: str,
        protocol: str,
        clean: bool,
        train_imgsize: Tuple[int, int] = (256, 256),
        test_imgsize: Tuple[int, int] = (256, 256),
        datadir: str = "data/mrnet",
        img_dir: str = "imgs",
    ) -> None:
        super().__init__(
            stage,
            diagnosis,
            plane,
            protocol,
            clean,
            train_imgsize,
            test_imgsize,
            datadir,
            img_dir,
        )
        self.stats = {
            "coronal": (59.70, 62.69),
            "axial": (63.69, 60.57),
            "sagittal": (58.82, 48.11),
        }

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

    def get_stats(self):
        return self.stats[self.plane]

    def get_cases(self):
        """load metadata and return tupple with list of ids and list of lbls"""

        path = f"{self.datadir}/{self.stage}-{self.diagnosis}.csv"

        cases = pd.read_csv(
            path, header=None, names=["id", "lbl"], dtype={"id": str, "lbl": np.int64}
        )

        # Exclude cases
        if self.stage == "train" and self.exclusions:
            cases = cases[~cases["id"].isin(self.exclusions)]

        ids = cases["id"].tolist()
        lbls = cases["lbl"].tolist()
        fpaths = [self.img_path / self.plane / (id + ".npy") for id in ids]

        return ids, lbls, fpaths


def get_dataloader(
    datareader,
    augs,
    use_3d,
    n_workers: int = 2,
):
    ds = DS(datareader, augs, use_3d=use_3d)

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=True if datareader.stage == "train" else False,
        num_workers=n_workers,
        pin_memory=True,
    )
    return dl


# class KneeMRI(DS):
#     """Stajdur kneemri dataset"""

#     def __init__(self, *args, **kwargs):
#         self._datadir = "data/kneemri"
#         self._img_dir = "imgs"
#         super().__init__(*args, **kwargs)

#         assert self.plane == "sagittal"

#     def get_cases(self, datadir: str, stage: str, diagnosis: str) -> Tuple[List[str], List[int]]:
#         path = os.path.join(datadir, "metadata.csv")
#         cases = pd.read_csv(path)
#         cases["ids"] = cases["volumeFilename"].str.replace(".pck", "", regex=False)
#         cases["aclDiagnosis"] = cases["aclDiagnosis"].replace(2, 1)

#         ids = cases["ids"].tolist()
#         lbls = cases["aclDiagnosis"].tolist()

#         return ids, lbls


# class SkmTea(DS):
#     """Stanford skm-tea dataset"""

#     def __init__(self, *args, **kwargs):
#         self._datadir = "data/skm-tea"
#         self._img_dir = "imgs"
#         super().__init__(*args, **kwargs)

#     def get_cases(self, datadir: str, stage: str, diagnosis: str) -> Tuple[List[str], List[int]]:
#         path = os.path.join(datadir, "metadata.csv")
#         cases = pd.read_csv(path)
#         ids = cases["scan_id"].tolist()
#         lbls = cases[self.diagnosis].tolist()

#         return ids, lbls


# %%
# class MRNetDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         datadir,
#         diagnosis,
#         transforms,
#         plane,
#         clean,
#         trim_train,
#         num_workers=1,
#         pin_memory=True,
#         shuffle_train=True,
#     ):
#         super().__init__()
#         self.diagnosis = diagnosis
#         self.transforms = transforms
#         self.datadir = datadir
#         self.plane = plane
#         self.clean = clean
#         self.trim_train = trim_train
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.shuffle_train = shuffle_train

#         self.train_ds = MRNet(
#             datadir=self.datadir,
#             stage="train",
#             diagnosis=self.diagnosis,
#             plane=self.plane,
#             clean=self.clean,
#             trim=self.trim_train,
#             transforms=self.transforms,
#         )

#         self.val_ds = MRNet(
#             datadir=self.datadir,
#             stage="valid",
#             diagnosis=self.diagnosis,
#             plane=self.plane,
#             clean=self.clean,
#             trim=False,
#             transforms=self.transforms,
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_ds,
#             batch_size=1,
#             shuffle=self.shuffle_train,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_ds,
#             batch_size=1,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#         )
