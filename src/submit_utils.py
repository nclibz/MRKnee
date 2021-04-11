import numpy as np
import albumentations as A
import torch
from model import MRKnee


def do_aug(imgs, transf):
    img_dict = {}
    target_dict = {}
    for i in range(imgs.shape[0]):
        if i == 0:
            img_dict['image'] = imgs[i, :, :]
        else:
            img_name = 'image'+f'{i}'
            img_dict[img_name] = imgs[i, :, :]
            target_dict[img_name] = 'image'
    transf = A.Compose(transf)
    transf.add_targets(target_dict)
    out = transf(**img_dict)
    out = list(out.values())
    return out  # returns list of np arrays


class Model():
    def __init__(self, ckpt, plane, backbone, device='cuda'):
        self.plane = plane
        self.device = torch.device(device)
        model = MRKnee.load_from_checkpoint(ckpt, planes=[plane], backbone=backbone)
        model.freeze()
        self.model = model.to(device=self.device)
        self.preds = []
        if 'b0' in backbone:
            self.img_sz = 224
        elif 'b1' in backbone:
            self.img_sz = 240

    def prep_img(self, path, plane):
        imgs = np.load(path)
        imgs = do_aug(imgs, transf=[A.CenterCrop(self.img_sz, self.img_sz)])
        imgs = torch.as_tensor(imgs, dtype=torch.float32)
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255
        if plane == 'axial':
            MEAN, SD = 66.4869, 60.8146
        elif plane == 'sagittal':
            MEAN, SD = 60.0440, 48.3106
        elif plane == 'coronal':
            MEAN, SD = 61.9277, 64.2818
        imgs = (imgs - MEAN)/SD
        imgs = imgs.unsqueeze(1)  # create channel dim
        imgs = imgs.unsqueeze(0)  # create batch dim
        imgs = imgs.to(device=self.device)
        return imgs

    def predict_proba(self, X):
        proba = torch.sigmoid(self.model(X))
        self.preds.append(proba.item())
        return self


def gather_preds(df, plane):

    paths = df[df[0].str.contains(plane)][0].tolist()

    # cnns
    if plane == 'axial':
        ckpts = {'abnormal_axial': 'efficientnet_b1',
                 'acl_axial': 'efficientnet_b1',
                 'meniscus_axial': 'efficientnet_b1'}
    if plane == 'sagittal':
        ckpts = {'abnormal_sagittal': 'efficientnet_b1',
                 'acl_sagittal': 'efficientnet_b1',
                 'meniscus_sagittal': 'efficientnet_b0'}
    if plane == 'coronal':
        ckpts = {'abnormal_coronal': 'efficientnet_b1',
                 'acl_coronal': 'efficientnet_b1',
                 'meniscus_coronal': 'efficientnet_b0'}

    models = [Model(f'src/models/{ckpt}.ckpt', plane, backbone)
              for ckpt, backbone in ckpts.items()]

    for model in models:
        for path in paths:
            img = model.prep_img(path=path, plane=plane)
            model.predict_proba(img)
    probas = [model.preds for model in models]
    return probas
