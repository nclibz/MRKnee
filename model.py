
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# TODO:
# Efficientnet
#  køre forward på alle 3 planes samtidig?
#   kan vel egentlig outputte dem direkte i linear ? Backpropper den så til alle?
#       kan jeg backproppe fra unified loss til alle 3 models??
#       lave individuel checkpoint for hvert plane
# scheduler
# hvordan virker 3d conv??


class MRKnee(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

    def training_step(self, batch, batchidx):
        img, label, sample_path, weight = batch
        logit = self(img)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)

        self.epoch_pred.append(torch.sigmoid(logit).item())
        self.epoch_lbl.append(label.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=.01)

    def on_epoch_start(self):
        self.epoch_pred = []
        self.epoch_lbl = []

    def on_epoch_end(self):
        self.log('auc', auroc(torch.Tensor(
            self.epoch_pred), torch.Tensor(self.epoch_lbl), pos_label=1), prog_bar=True)

        img, label, sample_path, weight = batch
        logit = self(img)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)
