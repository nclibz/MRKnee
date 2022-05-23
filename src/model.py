# %%
# import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%


class VanillaMRKnee(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        drop_rate: float,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=1,
            drop_rate=self.drop_rate,
        )
        self.num_features = self.backbone.num_features
        self.clf = nn.Linear(self.num_features, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # -> (num_imgs, c, h, w)
        x = self.backbone(x)  # -> (num_imgs, num_features)
        x = x.unsqueeze(0)  # (1, num_imgs, num_features)
        x = F.adaptive_max_pool2d(x, (1, x.size(-1)))
        x = x.squeeze(0)  # (1, num_features_out)
        x = self.clf(x)
        return x


class MRKnee3D(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        drop_rate: float,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=1,
            drop_rate=self.drop_rate,
        )
        self.num_features = self.backbone.num_features
        self.clf = nn.Linear(self.num_features, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # -> (num_imgs, c, h, w)
        x = self.backbone(x)  # -> (num_imgs, num_features)
        x = x.unsqueeze(0)  # (1, num_imgs, num_features)
        x = F.adaptive_max_pool2d(x, (1, x.size(-1)))
        x = x.squeeze(0)  # (1, num_features_out)
        x = self.clf(x)
        return x
