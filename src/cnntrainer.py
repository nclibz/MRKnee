from dataclasses import dataclass
from src.model import MRKnee
from src.data import MRKneeDataModule
import pytorch_lightning as pl

# Bruge .__dict__  til at passe cfg til neptune.
class CNNTrainer:
    def __init__(self, diagnosis, plane):
        self.diagnosis = diagnosis
        self.plane = plane

    def create_transforms(self):
        pass

    def create_datamodule(self, **kwargs):
        self.__dict__.update(kwargs)
        self.dm = MRKneeDataModule(diagnosis=self.diagnosis, plane=self.plane, **kwargs)

    def create_model(self, **kwargs):
        self.__dict__.update(kwargs)
        self.model = MRKnee(**kwargs)

    def create_callbacks(self):
        pass

    def neptune_logger(self):
        pass

    def create_pl_trainer(self, **kwargs):
        self.__dict__.update(kwargs)
        # TODO: callbaks -> list from self.callbacks
        # TODO: logger -> self.neptune_logger

        self.logger = self.neptune_logger()
        self.callbacks = self.create_callbacks()

        self.trainer = pl.Trainer(logger=self.logger, callbacks=self.callbacks, **kwargs)
