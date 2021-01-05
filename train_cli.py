# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import MRKnee
from data import MRKneeDataModule
from argparse import ArgumentParser

# %%
# %load_ext autoreload
# %autoreload 2


# TODO:

# lave  argparser? behøver jeg vel egentlig kun til submissions??
# Tune hyperparams
# error analysis - find top losses - visualise!


# %%

if __name__ == '__main__':
    dm = MRKneeDataModule(diagnosis="meniscus")

    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_auc", save_top_k=2, mode="max")
    model = MRKnee()
    trainer = pl.Trainer(gpus=1, fast_dev_run=True,
                         logger=tb_logger, callbacks=[checkpoint])
    trainer.fit(model, dm)

# %%
