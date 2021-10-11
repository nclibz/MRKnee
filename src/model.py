# %%
import pytorch_lightning as pl
from torchmetrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# %%


class MRKnee(pl.LightningModule):
    def __init__(
        self,
        backbone="efficientnet_b0",
        drop_rate=0.0,
        final_drop=0.0,
        learning_rate=0.0001,
        log_auc=True,
        log_ind_loss=False,
        adam_wd=0.01,
        precision=16,
        max_epochs=20,
    ):
        super().__init__()
        self.precision = precision
        self.learning_rate = learning_rate
        self.log_auc = log_auc
        self.log_ind_loss = log_ind_loss
        self.drop_rate = drop_rate
        self.final_drop = final_drop
        self.architecture = backbone

        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
            in_chans=1,
            drop_rate=drop_rate,
        )
        self.num_features = self.backbone.num_features
        self.final_drop = nn.Dropout(p=final_drop)
        self.adam_wd = adam_wd
        self.max_epochs = max_epochs

        self.clf = nn.Linear(self.num_features, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # -> (num_imgs, c, h, w)
        x = self.backbone(x)  # -> (num_imgs, num_features)
        x = x.unsqueeze(0)  # (1, num_imgs, num_features)

        # final pooling
        x = F.adaptive_max_pool2d(x, (1, x.size(-1)))
        x = x.squeeze(0)  # (1, num_features_out)
        x = self.clf(x)
        return x

    def training_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(logit, label, pos_weight=weight)

        # logging
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(logit, label, pos_weight=weight)

        # logging

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        if self.log_auc:
            self.preds.append(torch.sigmoid(logit).squeeze(0))
            self.lbl.append(label.squeeze(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.adam_wd
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, threshold=1e-4
            ),
            "monitor": "val_loss",
        }

    def on_validation_epoch_start(self):
        if self.log_auc:
            self.preds = []
            self.lbl = []

    def on_validation_epoch_end(self):
        if self.log_auc:
            preds = torch.cat(self.preds)
            lbls = torch.cat(self.lbl).to(dtype=torch.int32)
            self.log(
                "val_auc",
                auroc(preds, lbls, pos_label=1),
                prog_bar=True,
                on_epoch=True,
            )


# %%
