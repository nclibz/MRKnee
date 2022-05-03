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
        trim_p: float = 0.0,
        shift_limit: float = 0,
        scale_limit: float = 0,
        rotate_limit: float = 0,
        ssr_p: float = 0,
        clahe_p: float = 0,
    ):
        self.train_imgsize = train_imgsize
        self.test_imgsize = test_imgsize
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.clahe_p = clahe_p
        self.ssr_p = ssr_p
        self.trim_p = trim_p
        self.rng = default_rng()
        self.plane = None
        self.stage = None

    def set_transforms(self, stage, plane, stats):
        self.plane = plane
        self.stage = stage
        self.mean, self.sd = stats[plane]
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

            transforms.append(A.CenterCrop(*self.train_imgsize))

        elif stage == "valid":
            transforms.append(A.CenterCrop(*self.test_imgsize))

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

    def trim_imgs(self, imgs, trim_p):
        """trims first and last 10% imgs"""
        remove_n = imgs.shape[0] // int(trim_p * 100)
        return imgs[remove_n:-remove_n, :, :]

    def standardize(self, imgs):
        return (imgs - self.mean) / self.sd

    def __call__(self, imgs):

        if self.trim_p > 0.0:
            imgs = self.trim_imgs(imgs, self.trim_p)

        # Rescale intensities to range between 0 and 255 -> tror ikke den gør noget!
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255
        imgs = imgs.astype(np.uint8)

        imgs = self.apply_transforms(imgs)

        imgs = self.standardize(imgs)

        return imgs
