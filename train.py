# %%
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from model import MRKnee
from data import MRKneeDataModule

# TODO:

# implement argparser? Behøver vel egentlig kun argparser til predict??
# Tune hyperparams
# error analysis - find top losses - visualise!

# %%

dm = MRKneeDataModule(plane="axial", diagnosis="acl")

model = MRKnee()

# %%
tb_logger = pl_loggers.TensorBoardLogger('logs/')

trainer = pl.Trainer(gpus=1, max_epochs=1, logger=tb_logger)
trainer.fit(model, dm)
