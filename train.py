# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import MRKnee
from data import MRKneeDataModule
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# %%
%load_ext autoreload
%autoreload 1

data_args = {
    datadir = 'data',
    diagnosis = "acl",
    planes = ['axial'],  # , 'sagittal', 'coronal'
    n_chans = 1,

    IMG_SZ = 240
    CHANS = 1

}

model_args =
BACKBONE = 'efficientnet_b1'


AUGMENT = {
    'train': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)]),
    'valid': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)])
}


# %%
pl.seed_everything(123)

dm = MRKneeDataModule(DATADIR,
                      DIAGNOSIS,
                      planes=PLANES,
                      num_workers=2,
                      n_chans=CHANS,
                      transf=AUGMENT)

# Callbacks
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", save_top_k=2, mode="max")
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
tb_logger = pl_loggers.TensorBoardLogger('logs/')

# MODEL
model = MRKnee(backbone=BACKBONE,
               pretrained=True,
               drop_rate=0.5,
               learning_rate=1e-3,
               unfreeze_epoch=0,
               freeze_from=-1,
               planes=PLANES,
               n_chans=CHANS,
               log_ind_loss=True)

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

# %%
