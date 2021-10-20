import albumentations as A
from numpy.random import default_rng
import numpy as np
from typing import Any


class Augmentations:
    def __init__(
        self,
        model: Any,
        shift_limit: float,
        scale_limit: float,
        rotate_limit: float,
        max_res_train: int,
        reverse_p: float = 0.5,
        indp_normalz: bool = True,
    ):
        self.backbone_in = model.backbone.default_cfg["input_size"]
        self.backbone_test_in = model.backbone.default_cfg.get("test_input_size", self.backbone_in)
        self.max_res_train = max_res_train
        self.input_size = (
            self.max_res_train
            if self.backbone_in[-1] > self.max_res_train
            else self.backbone_in[-1]
        )
        self.test_input_size = (
            256 if self.backbone_test_in[-1] > 256 else self.backbone_test_in[-1]
        )
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.reverse_p = reverse_p
        self.indp_normalz = indp_normalz
        self.rng = default_rng()

    def set_transforms(self, stage, plane):
        transforms = []

        if stage == "train":
            transforms.append(
                A.ShiftScaleRotate(
                    always_apply=False,
                    p=1.0,
                    shift_limit=self.shift_limit,
                    scale_limit=self.scale_limit,
                    rotate_limit=self.rotate_limit,
                    border_mode=0,
                    value=(0, 0, 0),
                )
            )

            if plane != "sagittal":
                transforms.append(A.HorizontalFlip(p=0.5))

            transforms.append(A.CenterCrop(self.input_size, self.input_size))

        else:
            transforms.append(A.CenterCrop(self.test_input_size, self.test_input_size))

        self.transforms = A.Compose(transforms)

        return self

    def apply_transforms(self, imgs):
        img_dict = {}
        target_dict = {}
        for i in range(imgs.shape[0]):
            if i == 0:
                img_dict["image"] = imgs[i, :, :]
            else:
                img_name = "image" + f"{i}"
                img_dict[img_name] = imgs[i, :, :]
                target_dict[img_name] = "image"
        transf = self.transforms
        transf.add_targets(target_dict)
        out = transf(**img_dict)
        out = list(out.values())

        return np.array(out)

    def reverse_order(self, imgs):
        p = self.rng.random()
        if p < self.reverse_p:
            imgs = np.flipud(imgs)
        return imgs

    def standardize(self, imgs, plane):

        # Rescal range to be between 0 and 255
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255

        # normalize
        if self.indp_normalz:
            if plane == "axial":
                MEAN, SD = 66.4869, 60.8146
            elif plane == "sagittal":
                MEAN, SD = 60.0440, 48.3106
            elif plane == "coronal":
                MEAN, SD = 61.9277, 64.2818
        else:
            MEAN, SD = 58.09, 49.73

        return (imgs - MEAN) / SD

    def __call__(self, imgs, plane, stage):

        # apply albumentations
        res = self.apply_transforms(imgs)

        # apply reverse ordering for saggital
        if plane == "sagittal" and stage == "train":
            res = self.reverse_order(res)

        res = self.standardize(imgs=res, plane=plane)

        return res
