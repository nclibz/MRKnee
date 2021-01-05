# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import csv
torch.manual_seed(17)


# TODO:
# getitem skal outputte alle 3 planes
# lave visualiser for at se hvad dl spytter ud
# lave en yaml til conda
# lave transforms
#    Skal lige undersøge normaliseringen??
#   imagenet: normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#   std=[0.229, 0.224, 0.225])

# %%


class MRDS(Dataset):
    def __init__(self, datadir,  stage, diagnosis, transform=None):
        super().__init__()
        self.transform = transform
        self.stage = stage
        self.datadir = datadir

        # label_dict
        with open(f'{datadir}/{stage}-{diagnosis}.csv', "r") as f:
            self.label_dict = {row[0]: int(row[1])
                               for row in list(csv.reader(f))}

        self.labels = list(self.label_dict.values())
        self.ids = list(self.label_dict.keys())

        # calculate  pos_weight
        pos_count = np.sum(self.labels)
        neg_count = len(self.labels) - pos_count
        self.weight = torch.as_tensor(neg_count / pos_count).unsqueeze(0)

    def prep_series(self, plane, index):
        id = self.ids[index]
        path = f'{self.datadir}/{self.stage}/{plane}/{id}.npy'
        series = torch.from_numpy(np.load(path)).to(dtype=torch.float32)

        # transforms
        if self.transform:
            series = self.transform(series)

        # convert to 3chan
        series = torch.stack((series,)*3, axis=1)
        return series

    def __getitem__(self, index):
        # load imgs

        series = [self.prep_series(plane, index)
                  for plane in ['axial', 'sagittal', 'coronal']]

        #  label
        label = torch.as_tensor(int(self.labels[index])).unsqueeze(
            0).to(dtype=torch.float32)

        # sample_id
        sample_id = self.ids[index]
        return series, label, sample_id, self.weight

    def __len__(self):
        return len(self.labels)


# %%
class MRKneeDataModule(pl.LightningDataModule):

    def __init__(self, datadir, diagnosis):
        super().__init__()
        self.datadir = datadir
        self.diagnosis = diagnosis
        self.train_transforms = transforms.Compose([
            transforms.CenterCrop(224)
        ])
        self.val_transforms = transforms.Compose([
            transforms.CenterCrop(224)
        ])
        self.train_ds = MRDS(datadir, 'train', self.diagnosis,  self.train_transforms)
        self.val_ds = MRDS(datadir, 'valid', self.diagnosis, self.val_transforms)

    # create datasets

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=2)

# %%
# TESTING
#md = MRKneeDataModule('data', 'acl')
# md.train_ds[1]
# %%
