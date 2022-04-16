# %%
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.augmentations import Augmentations
from torch.utils.data import DataLoader
from src.model import MRKnee

# %%


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
        return self

    def get_preds(self):
        if len(self.preds) == 0:
            self.make_preds()
        return self.preds

    def get_lbls(self):
        if len(self.lbls) == 0:
            self.make_preds()
        return self.lbls

    def get_auc(self):
        preds = self.get_preds()
        lbls = self.get_lbls()
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(lbls, preds)
        auc = metrics.auc(self.fpr, self.tpr)
        return auc

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
def collect_predictors(
    diagnosis: str, planes: List[str], stage: str, imgsize: int, ds_class, ckpt_dir: str
):

    augs = Augmentations(train_imgsize=imgsize, test_imgsize=imgsize)

    predictors = []

    for plane in planes:
        model = MRKnee.load_from_checkpoint(
            f"{ckpt_dir}/{diagnosis}_{plane}.ckpt",
            backbone="tf_efficientnetv2_s_in21k",
            drop_rate=0.5,
            learning_rate=1e-4,
            adam_wd=0.001,
            max_epochs=20,
            precision=32,
            log_auc=False,
            log_ind_loss=False,
        )

        ds = ds_class(
            stage=stage,
            diagnosis=diagnosis,
            plane=plane,
            clean=False,
            transforms=augs,
        )

        dl = DataLoader(
            ds,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        )

        predictors.append(CNNPredict(model, dl))
    return predictors
