from typing import Any

import albumentations as A
import numpy as np
from numpy.random import default_rng

# TODO: Skal gøre så test size kan justeres


class Augmentations:
    """docstring"""

    def __init__(
        self,
        train_imgsize,
        test_imgsize,
        shift_limit: float,
        scale_limit: float,
        rotate_limit: float,
        ssr_p: float,
        clahe_p: float,
        reverse_p: float = 0.5,
        indp_normalz: bool = True,
    ):
        self.train_imgsize = train_imgsize
        self.test_imgsize = test_imgsize
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.clahe_p = clahe_p
        self.reverse_p = reverse_p
        self.indp_normalz = indp_normalz
        self.ssr_p = ssr_p
        self.rng = default_rng()

    def set_transforms(self, stage, plane):
        transforms = []

        if stage == "train":
            transforms.append(
                A.ShiftScaleRotate(
                    always_apply=False,
                    p=self.ssr_p,
                    shift_limit=self.shift_limit,
                    scale_limit=self.scale_limit,
                    rotate_limit=self.rotate_limit,
                    border_mode=0,
                    value=(0, 0, 0),
                )
            )

            transforms.append(A.CLAHE(p=self.clahe_p))

            if plane != "sagittal":
                transforms.append(A.HorizontalFlip(p=0.5))

            transforms.append(A.CenterCrop(self.train_imgsize[0], self.train_imgsize[1]))

        elif stage == "valid":
            transforms.append(A.CenterCrop(self.test_imgsize[0], self.test_imgsize[1]))

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

        # Rescale intensities to range between 0 and 255 -> tror ikke den gør noget!
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255
        imgs = imgs.astype(np.uint8)

        res = self.apply_transforms(imgs)

        # apply reverse ordering for saggital
        if plane == "sagittal" and stage == "train":
            res = self.reverse_order(res)

        res = self.standardize(imgs=res, plane=plane)

        return res
