# %%
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import csv

from torch.utils.data.sampler import WeightedRandomSampler
from utils import do_aug
# %%


class MRDS(Dataset):
    def __init__(self, datadir,
                 stage,
                 diagnosis,
                 transf=None,
                 planes=['axial', 'sagittal', 'coronal'],
                 n_chans=1,
                 indp_normalz=True,
                 w_loss=True,
                 same_range=True):
        super().__init__()
        self.stage = stage
        self.datadir = datadir
        self.planes = planes
        self.n_chans = n_chans
        self.transf = transf
        self.diagnosis = diagnosis
        self.indp_normalz = indp_normalz
        self.w_loss = w_loss
        self.same_range = same_range

        # get cases
        with open(f'{datadir}/{stage}-{diagnosis}.csv', "r") as f:
            self.cases = [(row[0], int(row[1]))
                          for row in list(csv.reader(f))]

        if w_loss:
            lbls = [lbl for _, lbl in self.cases]
            pos_count = np.sum(lbls)
            neg_count = len(lbls) - pos_count
            self.weight = torch.as_tensor(
                neg_count / pos_count, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, index):

        id, label = self.cases[index]

        imgs = [self.prep_imgs(id, plane)
                for plane in self.planes]

        label = torch.as_tensor(label, dtype=torch.float32).unsqueeze(0)

        return imgs, label, id, self.weight

    def prep_imgs(self, id, plane):
        path = f'{self.datadir}/{self.stage}/{plane}/{id}.npy'
        imgs = np.load(path)

        if self.transf:
            imgs = do_aug(imgs, self.transf[self.stage])

        imgs = torch.as_tensor(imgs, dtype=torch.float32)

        if self.same_range:
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255

        # normalize
        if self.indp_normalz:
            if plane == 'axial':
                MEAN, SD = 66.4869, 60.8146
            elif plane == 'sagittal':
                MEAN, SD = 60.0440, 48.3106
            elif plane == 'coronal':
                MEAN, SD = 61.9277, 64.2818
        else:
            MEAN, SD = 58.09, 49.73

        imgs = (imgs - MEAN)/SD

        if self.n_chans == 1:
            imgs = imgs.unsqueeze(1)
        else:
            imgs = torch.stack((imgs,)*3, axis=1)

        return imgs

    def __len__(self):
        return len(self.cases)


# %%

class MRKneeDataModule(pl.LightningDataModule):

    def __init__(self,
                 datadir,
                 diagnosis,
                 transf=None,
                 planes=['axial', 'sagittal', 'coronal'],
                 upsample=False,
                 n_chans=1,
                 w_loss=True,
                 indp_normalz=False,
                 num_workers=1,
                 pin_memory=True,
                 same_range=True,
                 ** kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.upsample = upsample
        self.w_loss = w_loss
        self.indp_normalz = indp_normalz
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.same_range = same_range

        assert(self.upsample != self.w_loss)

        self.train_ds = MRDS(datadir,
                             'train',
                             diagnosis,
                             transf,
                             planes,
                             n_chans,
                             w_loss=w_loss,
                             indp_normalz=self.indp_normalz,
                             same_range=self.same_range)
        self.val_ds = MRDS(datadir,
                           'valid',
                           diagnosis,
                           transf,
                           planes,
                           n_chans,
                           w_loss=w_loss,
                           indp_normalz=self.indp_normalz,
                           same_range=self.same_range)
        if self.upsample:
            lbls = [lbl for _, lbl in self.train_ds.cases]
            class_counts = np.bincount(lbls)
            class_weights = 1 / torch.Tensor(class_counts)
            samples_weight = [class_weights[t] for t in lbls]
            self.sampler = WeightedRandomSampler(samples_weight, 1)

    def train_dataloader(self):
        if self.upsample:
            trainloader = DataLoader(self.train_ds, batch_size=1,
                                     sampler=self.sampler,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory)
        else:
            trainloader = DataLoader(self.train_ds, batch_size=1,
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     pin_memory=self.pin_memory)

        return trainloader

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


# %%
