
# %%
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import timm


# TODO:
# efficientnet models tager forskellige størrelser - skal upscale / padde tilsvarende
# scheduler -> skal finde en måde at få total steps på
# hvorfor tager jeg torch.max??


# fine_tuning
# fine-tune efficientnet
#   for et par epochs freeze feature extraction.
#   discriminative learning rate - stigende lr fra tail to head.
# implementere en finetuning class ud fra fine-tuning eksemplet ??
# timm models har model.as_sequential
# kan slices som list og er defineret i blocks.
# finder den største model der kan fitte i mem på colab
# lære arkitekturen og speciallave en func i on_train_epoch der freezer

# def on_epoch_start(self):
#     if self.current_epoch == 0:
#         self.freeze()
#         self.trainer.lr_schedulers = ... # Define new scheduler

#     if self.current_epoch == N_FREEZE_EPOCHS:
#         self.unfreeze() # Or partially unfreeze
#         self.trainer.lr_schedulers = ... # Define new scheduler

# %%

class MRKnee(pl.LightningModule):
    def __init__(self, backbone='efficientnet_b1',
                 learning_rate=0.0001,
                 debug=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.debug = debug

        # layers

        if self.debug:
            self.bn_ax = nn.BatchNorm2d(3)
            self.model_ax = timm.create_model(
                backbone, pretrained=True, num_classes=0)
            self.clf = nn.Linear(self.model_ax.num_features, 1)

        else:
            self.bn_ax = nn.BatchNorm2d(3)
            self.model_ax = self.freeze(timm.create_model(
                backbone, pretrained=True, num_classes=0))
            self.bn_sag = nn.BatchNorm2d(3)
            self.model_sag = self.freeze(timm.create_model(
                backbone, pretrained=True, num_classes=0))
            self.bn_cor = nn.BatchNorm2d(3)
            self.model_cor = self.freeze(timm.create_model(
                backbone, pretrained=True, num_classes=0))  # set global_pool='' to return unpooled
            self.clf = nn.Linear(self.model_ax.num_features*3, 1)

    def run_model(self, model, bn, series):
        x = torch.squeeze(series, dim=0)
        x = bn(x)
        x = model(x)
        x = torch.max(x, 0, keepdim=True)[0]  # Hvad gør det?
        return x

    def forward(self, x):

        if self.debug:
            y = self.run_model(self.model_ax, self.bn_ax, x)
        else:
            ax, sag, cor = x
            ax = self.run_model(self.model_ax, self.bn_ax, ax)
            sag = self.run_model(self.model_sag, self.bn_sag, sag)
            cor = self.run_model(self.model_cor, self.bn_cor, cor)
            y = torch.cat((ax, sag, cor), 1)
        return self.clf(y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=.01)
        return optimizer

    def training_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch == 3:
            self.unfreeze(self.model_ax)
            self.unfreeze(self.model_cor)
            self.unfreeze(self.model_sag)

    def validation_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)

        self.preds.append(torch.sigmoid(logit).item())
        self.lbl.append(label.item())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_start(self):
        self.preds = []
        self.lbl = []

    def on_validation_epoch_end(self):
        self.log('val_auc', auroc(torch.Tensor(
            self.preds), torch.Tensor(self.lbl), pos_label=1), prog_bar=True)

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(module) -> None:
        for param in module.parameters():
            param.requires_grad = True

# %%
