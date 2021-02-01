
# %%
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

import timm

# %%


class MRKnee(pl.LightningModule):
    def __init__(self,
                 backbone='efficientnet_b0',
                 pretrained=True,
                 n_chans=1,
                 drop_rate=0.0,
                 final_drop=0.0,
                 learning_rate=0.0001,
                 freeze_from=-1,
                 unfreeze_epoch=0,  # -1 for not freezing any layers
                 planes=['axial', 'sagittal', 'coronal'],
                 log_auc=True,
                 log_ind_loss=False,
                 final_pool='max',
                 lstm=True,
                 lstm_layers=1,
                 lstm_h_size=512,
                 adam_wd=0.01,
                 ** kwargs):
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
        self.final_pool = final_pool
        self.final_drop = nn.Dropout(p=final_drop)
        self.adam_wd = adam_wd

        # freeze backbones
        self.backbones = ModuleList([self._freeze(module.as_sequential(), freeze_from)
                                     for module in self.backbones])
        self.clf = nn.Linear(self.num_features*self.n_planes, 1)

        # lstm
        self.do_lstm = lstm
        self.lstm_layers = lstm_layers
        self.lstm_h_size = lstm_h_size
        if self.do_lstm:
            self.lstm = nn.LSTM(input_size=self.num_features,
                                hidden_size=lstm_h_size,
                                num_layers=lstm_layers,
                                batch_first=True, bidirectional=False
                                )
            self.clf = nn.Linear(lstm_h_size, 1)

        # logging
        self.t_sample_loss = {}
        self.v_sample_loss = {}
        self.best_val_loss = 20

    def forward(self, x):
        x = [self.run_model(model, series)
             for model, series in zip(self.backbones, x)]
        x = torch.cat(x, 1)
        x = self.final_drop(x)
        x = self.clf(x)
        return x

    def run_model(self, model, series):
        x = torch.squeeze(series, dim=0)
        x = model(x)  # (num_imgs, num_features)
        x = x.unsqueeze(0)  # (1, num_imgs, num_features)
        # lstm
        if self.do_lstm:
            x = self.final_drop(x)
            h0 = torch.zeros(self.lstm_layers, x.size(0),
                             self.lstm_h_size).to(self.device)
            c0 = torch.zeros(self.lstm_layers, x.size(0),
                             self.lstm_h_size).to(self.device)
            x, _ = self.lstm(x, (h0, c0))  # (1, num_imgs, lstm_h_size)

        # final pooling

        if self.final_pool == 'max':
            x = F.adaptive_max_pool2d(x, (1, x.size(-1)))
            x = x.squeeze(0)  # (1, num_features_out)
        elif self.final_pool == 'avg':
            x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))
            x = x.squeeze(0)
        elif self.final_pool == 'last_t_step':
            x = x[:, -1, :]  # ( 1, num_features_out)
        return x  # (1, num_features_out)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.adam_wd)

        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=1e-4),
            'monitor': 'val_loss'}

    def training_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)

        # logging
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        if self.log_ind_loss:
            self.t_sample_loss[sample_id] = (loss.detach(), label)
        return loss

    def on_train_epoch_start(self):
        if self.current_epoch == self.unfreeze_epoch:
            self.backbones = ModuleList([self._unfreeze(module, self.freeze_from)
                                         for module in self.backbones])

    def validation_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)

        # logging

        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        if self.log_ind_loss:
            self.v_sample_loss[sample_id] = loss
        if self.log_auc:
            self.preds.append(torch.sigmoid(logit).squeeze(0))
            self.lbl.append(label.squeeze(0))
        return loss

# log sample losses til neptune VIRKER IKKE LIGE NU - det er under validation step. Skal bruge mean loss
# kan bare bruge den fra neptune-contrib
#         if loss < self.best_val_loss:
#             self.trainer.logger.log_artifact(export_pickle(
#                 self.t_sample_loss), "t_sample_loss.pkl")
#             self.trainer.logger.log_artifact(export_pickle(
#                 self.v_sample_loss), "v_sample_loss.pkl")
#             self.best_val_loss = loss

    def on_validation_epoch_start(self):
        if self.log_auc:
            self.preds = []
            self.lbl = []

    def on_validation_epoch_end(self):
        if self.log_auc:
            self.log('val_auc', auroc(torch.cat(self.preds), torch.cat(self.lbl), pos_label=1),
                     prog_bar=True, on_epoch=True)

    def _unfreeze(self, module, idx):
        for param in module[idx:].parameters():
            param.requires_grad = True
        return module

    def _freeze(self, module, idx):
        for param in module[idx:].parameters():
            param.requires_grad = False
        return module


# %%
