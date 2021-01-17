# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import MRKnee
from data import MRKneeDataModule
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import albumentations as A


# %%
%load_ext autoreload
%autoreload 2
# %%


IMG_SZ = 240
PLANES = ['axial']  # , 'sagittal', 'coronal'
N_CHANS = 1

data_args = {
    'datadir': 'data',
    'diagnosis': "acl",
    'planes': PLANES,
    'n_chans': N_CHANS,
    'num_workers': 2,
    'transf': {
        'train': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)]),
        'valid': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)])
    }
}

model_args = {
    'backbone': 'efficientnet_b1',
    'pretrained': True,
    'learning_rate': 1e-3,
    'drop_rate': 0.5,
    'freeze_from': -1,
    'unfreeze_epoch': 0,
    'planes': PLANES,
    'n_chans': N_CHANS,
    'log_auc': True,
    'log_ind_loss': True
}


AUGMENT = {
    'train': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)]),
    'valid': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)])
}


# %%
pl.seed_everything(123)

dm = MRKneeDataModule(**data_args)
model = MRKnee(**model_args, log_data_args=data_args)

# Callbacks
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="loss/val_loss", save_top_k=2, mode="min")
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
tb_logger = pl_loggers.TensorBoardLogger('logs/')

# MODEL

# TRAINER
trainer = pl.Trainer(gpus=1,
                     precision=16,
                     #max_epochs = 2,
                     #overfit_batches = 10,
                     limit_train_batches=0.4,
                     num_sanity_val_steps=0,
                     logger=tb_logger,
                     progress_bar_refresh_rate=25,
                     # callbacks=[checkpoint],
                     deterministic=True)
#overfit_batches = 10
# max_epochs = 2


# %%
trainer.fit(model, dm)
# %%
