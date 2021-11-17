# %%
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler


# %%


# %%


class MRDS(Dataset):
    def __init__(
        self,
        datadir,
        stage,
        diagnosis,
        transforms,
        plane,
        clean,
        trim,
    ):
        super().__init__()
        self.stage = stage
        self.datadir = datadir
        self.plane = plane
        self.transforms = transforms.set_transforms(stage, plane)
        self.diagnosis = diagnosis
        self.trim = trim

        # get list of exclusions

        exclusions = {
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

        if clean:
            exclude = exclusions[stage][plane]
        else:
            exclude = []

        # get cases
        path = f"{datadir}/{stage}-{diagnosis}.csv"
        self.cases = pd.read_csv(
            path, header=None, names=["id", "lbl"], dtype={"id": str, "lbl": np.int64}
        )
        if stage == "train" and exclude:
            self.cases = self.cases[~self.cases["id"].isin(exclude)]

        lbls = self.cases[["lbl"]].to_numpy()
        pos_count = np.sum(lbls)
        neg_count = len(lbls) - pos_count
        self.weight = torch.as_tensor(neg_count / pos_count, dtype=torch.float32).unsqueeze(0)

    def trim_imgs(self, imgs):
        remove_n = imgs.shape[0] // 10
        return imgs[remove_n:-remove_n, :, :]

    def __getitem__(self, idx):

        row = self.cases.iloc[idx, :]
        label = row["lbl"]
        id = row["id"]
        path = f"{self.datadir}/{self.stage}/{self.plane}/{id}.npy"
        imgs = np.load(path)

        if self.trim:
            imgs = self.trim_imgs(imgs)

        imgs = self.transforms(imgs, plane=self.plane, stage=self.stage)

        imgs = torch.from_numpy(imgs).float()
        imgs = imgs.unsqueeze(1)  # add channel

        label = torch.as_tensor(label, dtype=torch.float32).unsqueeze(0)

        return imgs, label, id, self.weight

    def __len__(self):
        return len(self.cases)


# %%


class MRKneeDataModule(pl.LightningDataModule):
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

        self.train_ds = MRDS(
            self.datadir,
            "train",
            self.diagnosis,
            self.transforms,
            self.plane,
            clean=self.clean,
            trim=self.trim_train,
        )

        self.val_ds = MRDS(
            self.datadir,
            "valid",
            self.diagnosis,
            self.transforms,
            self.plane,
            clean=self.clean,
            trim=False,
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
