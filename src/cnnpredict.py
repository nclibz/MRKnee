# %%
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from src.augmentations import Augmentations
from src.data import MRKneeDataModule
from src.model import MRKnee
from sklearn.metrics import roc_auc_score


# %%


class CNNPredict:
    def __init__(
        self,
        diagnosis: str,
        plane: str,
        checkpoint: str,
        backbone: str,
        datadir: str,
        device: str = "cuda",
    ) -> None:
        self.diagnosis = diagnosis
        self.plane = plane
        self.checkpoint = checkpoint
        self.backbone = backbone
        self.device = device
        self.datadir = datadir
        self.model = MRKnee.load_from_checkpoint(
            self.checkpoint,
            backbone=self.backbone,
            drop_rate=0.5,
            learning_rate=1e-4,
            adam_wd=0.001,
            max_epochs=20,
            precision=32,
            log_auc=False,
            log_ind_loss=False,
        )
        self.augs = Augmentations(
            self.model,
            max_res_train=256,
            shift_limit=0,
            scale_limit=0,
            rotate_limit=0,
            reverse_p=0.0,
            ssr_p=1,
            clahe_p=1,
            indp_normalz=True,
        )

    def get_preds(self, stage):

        # TODO: Har slået clean fra. Ellers får jeg forskellige længde af preds

        dm = MRKneeDataModule(
            datadir=self.datadir,
            diagnosis=self.diagnosis,
            plane=self.plane,
            transforms=self.augs,
            clean=False,
            num_workers=2,
            pin_memory=True,
            trim_train=True,
            shuffle_train=False,
        )

        if stage == "train":
            dl = dm.train_dataloader()
        else:
            dl = dm.val_dataloader()
        trainer = pl.Trainer(gpus=1)

        preds_and_lbls = trainer.predict(self.model, dl)
        self.preds = torch.tensor([pred for pred, lbl in preds_and_lbls]).cpu().numpy()
        self.lbls = torch.tensor([lbl for pred, lbl in preds_and_lbls]).cpu().unsqueeze(1).numpy()
        return self

    def get_auc(self):
        return roc_auc_score(self.lbls, self.preds)

    def __repr__(self) -> str:
        return f"CNNPredictor({self.diagnosis}, {self.plane})"


# %%
