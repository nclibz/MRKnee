# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import glob
import pickle
import numpy as np
import csv
torch.manual_seed(17)


# TODO:
# getitem skal outputte alle 3 planes
# lave visualiser for at se hvad dl spytter ud
# lave en yaml til conda
# lave transforms


# %%

class MRDS(Dataset):
    def __init__(self, stage, diagnosis, plane, transform=None):
        super().__init__()
        path = f'data/{stage}/{plane}/'
        self.transform = transform

        # label_dict
        with open(f'data/{stage}-{diagnosis}.csv', "r") as f:
            self.label_dict = {f'{path}{row[0]}.npy': int(row[1])
                               for row in list(csv.reader(f))}

        self.labels = list(self.label_dict.values())
        self.paths = list(self.label_dict.keys())

        # calculate  pos_weight
        pos_count = np.sum(self.labels)
        neg_count = len(self.labels) - pos_count
        self.weight = torch.as_tensor(neg_count / pos_count).unsqueeze(0)

    def __getitem__(self, index):
        # load images
        series = torch.from_numpy(np.load(self.paths[index])).to(dtype=torch.float32)

        # transforms
        if self.transform:
            series = self.transform(series)

        # convert to 3chan
        series = torch.stack((series,)*3, axis=1)

        #  label
        label = torch.as_tensor(int(self.labels[index])).unsqueeze(
            0).to(dtype=torch.float32)

        # sample_id
        sample_path = self.paths[index]
        return series, label, sample_path, self.weight

    def __len__(self):
        return len(self.labels)


# %%


class MRKneeDataModule(pl.LightningDataModule):

    def __init__(self, diagnosis, plane):
        super().__init__()
        self.diagnosis = diagnosis
        self.plane = plane
        self.train_transforms = transforms.Compose([
            transforms.CenterCrop(224)
        ])
        self.train_ds = MRDS('train', self.diagnosis,
                             self.plane, self.train_transforms)
        self.val_ds = MRDS('valid', self.diagnosis, self.plane)

    # create datasets

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False)
