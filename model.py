
# %%
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch.optim.lr_scheduler import OneCycleLR
import timm


# TODO:
# efficientnet models tager forskellige størrelser - skal upscale / padde tilsvarende
# scheduler -> skal finde en måde at få total steps på
# hvorfor tager jeg torch.max??


# fine_tuning
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


# NÅET TIL: HAR LAVET SIMPEL FREEZE MEN DEN LAVER FEJL??

# %%

class MRKnee(pl.LightningModule):
    def __init__(self, backbone='tf_efficientnet_b0_ns',
                 learning_rate=0.0001,
                 debug=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_models = 1 if debug == True else 3  # kan nok tage det som flag fra ds?

        self.backbones = [timm.create_model(
            backbone, pretrained=True, num_classes=0) for i in range(self.num_models)]
        self.num_features = self.backbones[0].num_features
        # self.backbones = ModuleList(self.backbones)
        # freeze backbones
        self.backbones = ModuleList([self.freeze(module.as_sequential())
                                     for module in self.backbones])
        self.bn_layers = ModuleList([nn.BatchNorm2d(3)
                                     for i in range(self.num_models)])
        self.clf = nn.Linear(self.num_features*self.num_models, 1)

    def run_model(self, model, bn, series):
        x = torch.squeeze(series, dim=0)
        x = bn(x)
        x = model(x)
        x = torch.max(x, 0, keepdim=True)[0]  # Hvad gør det?
        return x

    def forward(self, x):
        x = [self.run_model(model, bn, series)
             for model, bn, series in zip(self.backbones, self.bn_layers, x)]
        x = torch.cat(x, 1)
        x = self.clf(x)
        return x

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
            self.backbones = [self.unfreeze(module) for module in self.backbones]

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
        return module

    def unfreeze(self, module) -> None:
        for param in module.parameters():
            param.requires_grad = True
        return module

# %%
