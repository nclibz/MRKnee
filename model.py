
# %%
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList, Sequential
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, OneCycleLR
import timm


# %%

class MRKnee(pl.LightningModule):
    def __init__(self, backbone='tf_efficientnet_b0_ns',
                 pretrained=True,
                 n_chans=1,
                 drop_rate=0.0,
                 learning_rate=0.0001,
                 freeze_from=4,
                 unfreeze_epoch=5,  # -1 for not freezing any layers
                 planes=['axial', 'sagittal', 'coronal'],
                 log_auc=True,
                 log_ind_loss=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.freeze_from = freeze_from
        self.unfreeze_epoch = unfreeze_epoch
        self.log_auc = log_auc
        self.log_ind_loss = log_ind_loss
        self.n_planes = len(planes)

        self.backbones = [timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                            in_chans=n_chans, drop_rate=drop_rate, ) for i in range(self.n_planes)]
        self.num_features = self.backbones[0].num_features

        # freeze backbones
        self.backbones = ModuleList([self.freeze(module.as_sequential(), freeze_from)
                                     for module in self.backbones])
        self.clf = nn.Linear(self.num_features*self.n_planes, 1)
        self.t_sample_loss = {}
        self.v_sample_loss = {}

    def run_model(self, model, series):
        x = torch.squeeze(series, dim=0)
        x = model(x)
        x = torch.max(x, 0, keepdim=True)[0]  # Hvad gør det?
        return x

    def forward(self, x):
        x = [self.run_model(model, series)
             for model, series in zip(self.backbones, x)]
        x = torch.cat(x, 1)
        x = self.clf(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-4),
            'monitor': 'val_loss'}

        # {
        #     'optimizer': optimizer,
        #     'lr_scheduler': CyclicLR(optimizer, base_lr=1e-6, max_lr=self.learning_rate, step_size_up=len(self.train_dataloader())*2, mode="triangular2", cycle_momentum=False),
        #     'interval': 'step',
        #     'frequency': 1,
        # }

    def training_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)

        # logging
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        if self.log_ind_loss:
            self.t_sample_loss[sample_id] = loss.detach()
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch == self.unfreeze_epoch:
            self.backbones = ModuleList([self.unfreeze(module, self.freeze_from)
                                         for module in self.backbones])

    def validation_step(self, batch, batchidx):
        imgs, label, sample_id = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label)

        # logging

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        if self.log_ind_loss:
            self.v_sample_loss[sample_id] = loss.detach()
        if self.log_auc:
            self.preds.append(torch.sigmoid(logit).squeeze(0))
            self.lbl.append(label.squeeze(0))
        return loss

    def on_validation_epoch_start(self):
        if self.log_auc:
            self.preds = []
            self.lbl = []

    def on_validation_epoch_end(self):
        if self.log_auc:
            self.log('val_auc', auroc(torch.cat(self.preds), torch.cat(self.lbl), pos_label=1),
                     prog_bar=True, on_epoch=True)

    def unfreeze(self, module, idx):
        for param in module[idx:].parameters():
            param.requires_grad = True
        return module

    def freeze(self, module, idx):
        for param in module[idx:].parameters():
            param.requires_grad = False
        return module


# %%
