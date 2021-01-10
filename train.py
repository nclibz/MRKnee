# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import MRKnee
from data import MRKneeDataModule
from argparse import ArgumentParser

# %%
%load_ext autoreload
%autoreload 2


# %%
if __name__ == '__main__':
    pl.seed_everything(123)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_auc", save_top_k=2, mode="max")

    DEBUG = True

    dm = MRKneeDataModule(datadir='data', diagnosis="meniscus",
                          num_workers=2, debug=DEBUG)
    model = MRKnee(backbone='tf_efficientnet_b0_ns', debug=DEBUG, learning_rate=1e-5)
    trainer = pl.Trainer(gpus=1,
                         precision=16,
                         max_epochs=2,
                         limit_train_batches=10,
                         num_sanity_val_steps=0,
                         logger=tb_logger,
                         callbacks=[checkpoint],
                         deterministic=True)
    trainer.fit(model, dm)


# %%
