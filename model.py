
# %%
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList, Sequential
from torch.optim.lr_scheduler import OneCycleLR
import timm


# %%

class MRKnee(pl.LightningModule):
    def __init__(self, backbone='tf_efficientnet_b0_ns',
                 learning_rate=0.0001,
                 freeze_from=4,
                 unfreeze_epoch=5,  # -1 for not freezing any layers
                 debug=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.freeze_from = freeze_from
        self.unfreeze_epoch = unfreeze_epoch
        self.num_models = 1 if debug == True else 3  # kan nok tage det som flag fra ds?

        self.backbones = [timm.create_model(
            backbone, pretrained=True, num_classes=0) for i in range(self.num_models)]
        self.num_features = self.backbones[0].num_features
        # self.backbones = ModuleList(self.backbones)
        # freeze backbones
        self.backbones = ModuleList([self.freeze(module.as_sequential(), freeze_from)
                                     for module in self.backbones])
        self.bn_layers = ModuleList([nn.BatchNorm2d(3)
                                     for i in range(self.num_models)])
        # self.clf = Sequential(nn.Linear(self.num_features*self.num_models, 512),
        #                      nn.Linear(512, 1))
        self.clf = nn.Linear(self.num_features*self.num_models, 1)
        self.t_sample_loss = {}
        self.v_sample_loss = {}

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
            self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batchidx):
        imgs, label, sample_id = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label)

        # logging
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        #self.t_sample_loss[sample_id] = loss.item()
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
        #self.v_sample_loss[sample_id] = loss.item()
        self.preds.append(torch.sigmoid(logit).squeeze(0))
        self.lbl.append(label.squeeze(0))
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_start(self):
        self.preds = []
        self.lbl = []

    def on_validation_epoch_end(self):
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
