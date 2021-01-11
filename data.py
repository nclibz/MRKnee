# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import csv


MAX_PIXEL_VAL = 255
MEAN = 58.09
STD = 49.73

# series = (series - series.min()) / (series.max() - series.min()) * MAX_PIXEL_VAL  # rescaling

# %%


class MRDS(Dataset):
    def __init__(self, datadir,  stage, diagnosis, transform=None, debug=False, upsample=True):
        super().__init__()
        self.transform = transform
        self.stage = stage
        self.datadir = datadir
        self.debug = debug

        # load data
        with open(f'{datadir}/{stage}-{diagnosis}.csv', "r") as f:
            self.cases = [(row[0], int(row[1]))
                          for row in list(csv.reader(f))]
        if upsample:
            neg_cases = [case for case in self.cases if case[1] == 0]
            pos_cases = [case for case in self.cases if case[1] == 1]
            pos_count = len(pos_cases)
            neg_count = len(neg_cases)
            w = neg_count//pos_count if pos_count < neg_count else pos_count//neg_count
            self.cases = (neg_cases * int(w)) + \
                pos_cases if neg_count < pos_count else (pos_cases*int(w))+neg_cases

    def __getitem__(self, index):

        id, label = self.cases[index]

        imgs = [self.prep_imgs(id, plane)
                for plane in ['axial', 'sagittal', 'coronal']]

        if self.debug:
            imgs = imgs[0]

        label = torch.as_tensor(label, dtype=torch.float32).unsqueeze(0)

        return imgs, label, id

    def prep_imgs(self, id, plane):
        path = f'{self.datadir}/{self.stage}/{plane}/{id}.npy'
        imgs = torch.from_numpy(np.load(path)).to(dtype=torch.float32)

        # transforms
        if self.transform:

            imgs = self.transform(imgs)

        # convert to 3chan
        imgs = torch.stack((imgs,)*3, axis=1)

        return imgs

    def __len__(self):
        return len(self.cases)


# %%
class MRKneeDataModule(pl.LightningDataModule):

    def __init__(self, datadir, diagnosis, num_workers=0, transf=True, debug=False):
        super().__init__()
        self.datadir = datadir
        self.diagnosis = diagnosis
        self.num_workers = num_workers
        self.debug = debug
        self.train_transforms = None
        self.val_transforms = None
        if transf:
            self.train_transforms = transforms.Compose([
                transforms.RandomAffine(25, translate=(0.25, 0.25)),
                transforms.CenterCrop(240),
                # transforms.Normalize(mean=[MEAN], std=[STD]) afprøver bn layer som første input istedet
            ])
            self.val_transforms = transforms.Compose([
                transforms.CenterCrop(240),
                # transforms.Normalize(mean=[MEAN], std=[STD])
            ])
        self.train_ds = MRDS(datadir, 'train', self.diagnosis,
                             self.train_transforms, debug=self.debug)
        self.val_ds = MRDS(datadir, 'valid', self.diagnosis,
                           self.val_transforms, debug=self.debug)

    # create datasets

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers)


# %%
# TESTING
# md = MRKneeDataModule('data', 'meniscus', transf=False)
# len(md.train_ds)
# %%
