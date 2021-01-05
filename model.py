
# %%
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import auroc
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


# TODO:
# implementer Efficientnet
# løber tør for GPU mem
# Hvordan med optimizer??
#   kan vel egentlig outputte dem direkte i linear ? Backpropper den så til alle?
#       kan jeg backproppe fra unified loss til alle 3 models??
# scheduler
# hvorfor tager jeg torch.max??
# forstå view efter pool ordentligt

# efficientnet
#         # Convolution layers
#         x = self.extract_features(inputs)
#         # Pooling and final linear layer
#         x = self._avg_pooling(x)
#         if self._global_params.include_top:
#             x = x.flatten(start_dim=1)
#             x = self._dropout(x)
#             x = self._fc(x)
#         return x


#        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
#        self._dropout = nn.Dropout(self._global_params.dropout_rate)
#        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
# %%

class MRKnee(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model_ax = EfficientNet.from_pretrained('efficientnet-b0')
        self.model_sag = EfficientNet.from_pretrained('efficientnet-b0')
        self.model_cor = EfficientNet.from_pretrained('efficientnet-b0')
        self.clf = nn.Linear(1280*3, 1)

    def run_model(self, model, series):
        x = torch.squeeze(series, dim=0)  # only batch size 1 supported
        x = model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # Hvad gør de?
        x = torch.max(x, 0, keepdim=True)[0]  # Hvad gør de?
        return x

    def forward(self, x):
        axial, sagital, coronal = x
        ax = self.run_model(self.model_ax, axial)
        sag = self.run_model(self.model_sag, sagital)
        cor = self.run_model(self.model_cor, coronal)
        y = torch.cat((ax, sag, cor), 1)
        return self.clf(y)

    def training_step(self, batch, batchidx):
        imgs, label, sample_id, weight = batch
        logit = self(imgs)
        loss = F.binary_cross_entropy_with_logits(
            logit, label, pos_weight=weight)

        self.preds.append(torch.sigmoid(logit).item())
        self.lbl.append(label.item())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=.01)

    def on_epoch_start(self):
        self.preds = []
        self.lbl = []

    def on_epoch_end(self):
        self.log('auc', auroc(torch.Tensor(
            self.preds), torch.Tensor(self.lbl), pos_label=1), prog_bar=True)
