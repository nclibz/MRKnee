# %%
import pytorch_lightning as pl
import numpy as np
from torch.functional import _return_counts
from model import MRKnee
from augmentations import Augmentations

np.random.seed(12)
from utils import show_batch
from data import MRDS, MRKneeDataModule
from callbacks import Callbacks
import torch.nn.functional as F

import pandas as pd


diagnosis = "acl"
plane = "sagittal"
backbone = "efficientnet_b0"


# %%
model = MRKnee(
    backbone=backbone,
    drop_rate=0.0,
    final_drop=0.0,
    learning_rate=0.0001,
    log_auc=True,
    log_ind_loss=False,
    adam_wd=0.01,
    max_epochs=20,
    precision=32,
)

augs = Augmentations(
    model,
    shift_limit=0.20,
    scale_limit=0.20,
    rotate_limit=30,
    reverse_p=0.5,
    same_range=True,
    indp_normalz=True,
)

dm = MRKneeDataModule(
    datadir="data",
    diagnosis=diagnosis,
    plane=plane,
    transforms=augs,
    clean=True,
    num_workers=1,
    pin_memory=True,
    trim_train=True,
)

# TODO: Lave cfg class?
cfg = dict()
cfg.update(model.__dict__)
cfg.update(augs.__dict__)
cfg.update(dm.__dict__)


# %%

dl = dm.train_dataloader()

dl = iter(dl)

# %%

model.training_step(next(dl), batchidx=1)
