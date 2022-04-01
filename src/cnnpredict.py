# %%
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.augmentations import Augmentations
from src.data import MRNetDataModule
from src.model import MRKnee

# %%
# TODO: Pass in classes instead of instantiating inside CNNPredict
# Bliver jeg nød til for at kunne bruge andre datamodules


class CNNPredict:
    """Get predictions from model"""

    def __init__(
        self,
        model,
        dataloader,
    ) -> None:
        self.model = model
        self.dl = dataloader
        self.preds = []
        self.lbls = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def make_preds(self):
        """Calculate predictions from CNN"""
        preds_and_lbls = []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(iter(self.dl)):
                imgs, label = batch[0].to(self.device), batch[1].to(self.device)
                logit = self.model(imgs)
                preds_and_lbls.append((torch.sigmoid(logit), label))
        self.preds = torch.tensor([pred for pred, _ in preds_and_lbls]).cpu().numpy()
        self.lbls = torch.tensor([lbl for _, lbl in preds_and_lbls]).cpu().numpy()

    def get_preds(self):
        if not self.preds:
            self.make_preds()
        return self.preds, self.lbls

    def get_auc(self):
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.lbls, self.preds)
        return metrics.auc(self.fpr, self.tpr)

    def plot_roc(self):
        roc_auc = self.get_auc()
        display = metrics.RocCurveDisplay(
            fpr=self.fpr,
            tpr=self.tpr,
            roc_auc=roc_auc,
            estimator_name=f"{self.dl.dataset.diagnosis} {self.dl.dataset.plane}",
        )
        display.plot()
        plt.show()

    def __repr__(self) -> str:
        return f"CNNPredictor({self.dl.dataset.diagnosis}, {self.dl.dataset.plane})"


# %%
