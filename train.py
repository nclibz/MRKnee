# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from model import MRKnee
from data import MRKneeDataModule

%load_ext autoreload
%autoreload 2


# TODO:

# lave argparser - gør det nemmere at køre i terminal mens jeg kan rette kode
# Tune hyperparams
# error analysis - find top losses - visualise!

# %%

dm = MRKneeDataModule(diagnosis="acl")
model = MRKnee()

tb_logger = pl_loggers.TensorBoardLogger('logs/')

trainer = pl.Trainer(gpus=1, fast_dev_run=True, logger=tb_logger)
trainer.fit(model, dm)

# %%
