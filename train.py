# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import MRKnee
from data import MRKneeDataModule
import albumentations as A


# %%
%load_ext autoreload
%autoreload 2
# %%


IMG_SZ = 240
PLANES = ['axial']  # , 'sagittal', 'coronal'
N_CHANS = 1
DIAGNOSIS = 'acl'

data_args = {
    'datadir': 'data',
    'diagnosis': DIAGNOSIS,
    'planes': PLANES,
    'n_chans': N_CHANS,
    'num_workers': 2,
    'transf': {
        'train': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)]),
        'valid': A.Compose([A.CenterCrop(IMG_SZ, IMG_SZ)])
    }
}

model_args = {
    'backbone': 'efficientnet_b0',
    'pretrained': True,
    'learning_rate': 1e-3,
    'drop_rate': 0.5,
    'freeze_from': -1,
    'unfreeze_epoch': 0,
    'planes': PLANES,
    'n_chans': N_CHANS,
    'log_auc': False,
    'log_ind_loss': True
}


# %%
pl.seed_everything(123)

dm = MRKneeDataModule(**data_args)
model = MRKnee(**model_args, log_data_args=data_args)

# Callbacks
model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{val_loss:.2f}',
                                   save_weights_only=True,
                                   save_top_k=3,
                                   monitor='val_loss',
                                   period=1)


lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
tb_logger = pl_loggers.TensorBoardLogger(
    'logs/', name=f'{data_args["diagnosis"]}/{PLANES}')


neptune_logger = pl_loggers.NeptuneLogger(
    api_key="",
    params={**model_args, **data_args},
    project_name='nclibz/mrknee',
    tags=[DIAGNOSIS] + PLANES
)

# MODEL

# TRAINER
trainer = pl.Trainer(gpus=1,
                     precision=16,
                     limit_train_batches=10,
                     # max_epochs = 2,
                     # overfit_batches = 10,
                     num_sanity_val_steps=0,
                     logger=neptune_logger,
                     log_every_n_steps=100,
                     callbacks=[lr_monitor, model_checkpoint],
                     deterministic=True)
# overfit_batches = 10
# max_epochs = 2


# %%
trainer.fit(model, dm)
# %%
