import albumentations as A
import cv2
import torch
import torchvision
import torch.nn as nn
from layers import *

def backbone():
  backbone = torchvision.models.vgg11(pretrained=False).features
  backbone_module = torch.nn.Sequential(
      FeatureWrapper2(backbone),
      Sgate(),
      nn.Linear(512, 256)
      )
  return backbone_module

feature_module1 = FeatureNet1_1(backbone())
feature_module2 = FeatureNet1_1(backbone())
feature_module3 = FeatureNet1_1(backbone())

model = TriNet(feature_module1, feature_module2, feature_module3, indim=512)

img_aug = A.Compose(
    [
      A.Resize(160, 160),
      A.HorizontalFlip(p=0.5),
      A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12,
                rotate_limit=25, p=0.75, border_mode=cv2.BORDER_REPLICATE),
    ], p=1)

img_aug = A.Compose(
    [
      A.Resize(160, 160),
    ], p=1)


config = dict(
    seed=1234,
    use_gpu=False,
    learning_rate=1e-05,
    rgb=True,
    n_round=1,
    transform=img_aug,
    model=model,
    state_dict='model/exp9_5_4010.pth',
    im_type='all',
)
