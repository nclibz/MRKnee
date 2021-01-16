# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import csv
import imgaug.augmenters as iaa


# MAX_PIXEL_VAL = 255
# MEAN = 58.09
# STD = 49.73

# %%

class MRDS(Dataset):
    def __init__(self, datadir,
                 stage,
                 diagnosis,
                 transf=None,
                 planes=['axial', 'sagittal', 'coronal'],
                 upsample=True,
                 n_chans=1):
        super().__init__()
        self.stage = stage
        self.datadir = datadir
        self.planes = planes
        self.n_chans = n_chans
        self.transf = transf

        # get cases
        with open(f'{datadir}/{stage}-{diagnosis}.csv', "r") as f:
            self.cases = [(row[0], int(row[1]))
                          for row in list(csv.reader(f))]

        if self.stage == 'train' and upsample:
            neg_cases = [case for case in self.cases if case[1] == 0]
            pos_cases = [case for case in self.cases if case[1] == 1]
            pos_count = len(pos_cases)
            neg_count = len(neg_cases)
            w = round(
                neg_count/pos_count) if pos_count < neg_count else round(pos_count/neg_count)
            self.cases = (neg_cases * int(w)) + \
                pos_cases if neg_count < pos_count else (pos_cases*int(w))+neg_cases

    def __getitem__(self, index):

        id, label = self.cases[index]

        imgs = [self.prep_imgs(id, plane)
                for plane in self.planes]

        label = torch.as_tensor(label, dtype=torch.float32).unsqueeze(0)

        return imgs, label, id

    def prep_imgs(self, id, plane):
        path = f'{self.datadir}/{self.stage}/{plane}/{id}.npy'
        imgs = np.load(path)

        # transforms

        if self.transf:
            imgs = self.transf[self.stage](images=imgs)

        imgs = torch.as_tensor(imgs, dtype=torch.float32)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255

        # normalize
        if plane == 'axial':
            MEAN, SD = 66.4869, 60.8146
        elif plane == 'sagittal':
            MEAN, SD = 60.0440, 48.3106
        elif plane == 'coronal':
            MEAN, SD = 61.9277, 64.2818

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
                 upsample=True,
                 n_chans=1, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.train_ds = MRDS(datadir,
                             'train',
                             diagnosis,
                             transf,
                             planes,
                             upsample,
                             n_chans)
        self.val_ds = MRDS(datadir,
                           'valid',
                           diagnosis,
                           transf,
                           planes,
                           upsample,
                           n_chans)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, shuffle=True, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, **self.kwargs)


# %%
