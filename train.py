# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import MRKnee
from data import MRDS
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# %%
%load_ext autoreload
%autoreload 0


NUM_WORKERS = 2
DIAGNOSIS = "acl"
DATADIR = 'data'
TRANSFORMS = True
N_PLANES = 1
N_CHANS = 1
UPSAMPLE = True


# %%
if __name__ == '__main__':
    pl.seed_everything(123)


# Callbacks
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", save_top_k=2, mode="max")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    tb_logger = pl_loggers.TensorBoardLogger('logs/')

# DATA
    train_ds = MRDS(DATADIR, 'train', DIAGNOSIS, transforms=TRANSFORMS,
                    n_planes=N_PLANES, upsample=UPSAMPLE, n_chans=N_CHANS)

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=NUM_WORKERS)

    val_ds = MRDS(DATADIR, 'valid', DIAGNOSIS, transforms=TRANSFORMS,
                  n_planes=N_PLANES, upsample=UPSAMPLE, n_chans=N_CHANS)

    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=NUM_WORKERS)

# MODEL

    model = MRKnee(backbone='tf_efficientnet_b0_ns',
                   n_planes=N_PLANES, learning_rate=1e-5)

# TRAIN
    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         max_epochs=1,
                         limit_val_batches=100,
                         num_sanity_val_steps=0,
                         logger=tb_logger,
                         callbacks=[checkpoint, lr_monitor],
                         deterministic=True,
                         benchmark=False
                         )
    trainer.fit(model, train_dl, val_dl)


# %%

# %%
