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

# lave  argparser? - behøver jeg næsten kun til submission?
# Tune hyperparams
# error analysis - find top losses - visualise!


# %%
if __name__ == '__main__':
    dm = MRKneeDataModule(datadir='data', diagnosis="meniscus", num_workers=2)

    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_auc", save_top_k=2, mode="max")

    EPOCHS = 10
    STEPS = len(dm.train_ds)

    model = MRKnee(model_name='efficientnet_b0', total_steps=EPOCHS*STEPS)
    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         overfit_batches=1,
                         num_sanity_val_steps=0,
                         logger=tb_logger,
                         callbacks=[checkpoint])
    trainer.fit(model, dm)


# %%
