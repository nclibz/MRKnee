from typing import Any, Tuple

import albumentations as A
import numpy as np
import torch
from timm.data.random_erasing import RandomErasing

from src.data import DataReader

# TODO: Kunne godt være en dataclass?


class Augmentations:
    """docstring"""

    def __init__(
        self,
        ssr_p: float = 0,
        shift_limit: float = 0,
        scale_limit: float = 0,
        rotate_limit: float = 0,
        bc_p: float = 0,
        brigthness_limit: float = 0,
        contrast_limit: float = 0,
        clahe_p: float = 0,
        re_p: float = 0,
        trim_p: float = 0.0,
    ):
        self.ssr_p = ssr_p
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.bc_p = bc_p
        self.brightness_limit = brigthness_limit
        self.contrast_limit = contrast_limit
        self.re_p = re_p
        self.clahe_p = clahe_p
        self.trim_p = trim_p
        self.transforms = None
        self.mean = None
        self.sd = None

    def set_transforms(self, datareader: DataReader):
        plane = datareader.plane
        stage = datareader.stage
        train_imgsize = datareader.train_imgsize
        test_imgsize = datareader.test_imgsize
        self.mean, self.sd = datareader.get_stats()
        transforms = []

        ## TRAIN TRANSFORMS
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

            transforms.append(
                A.RandomBrightnessContrast(
                    p=self.bc_p,
                    brightness_limit=self.brightness_limit,
                    contrast_limit=self.contrast_limit,
                    brightness_by_max=False,
                )
            )

            transforms.append(A.CLAHE(p=self.clahe_p))

            if plane != "sagittal":
                transforms.append(A.HorizontalFlip(p=0.5))

            transforms.append(A.CenterCrop(*train_imgsize))
        ## VALID TRANSFORMS
        elif stage == "valid":
            transforms.append(A.CenterCrop(*test_imgsize))

        ### COMMON TRANSFORMS

        transforms.append(
            A.Normalize(
                mean=self.mean, std=self.sd, always_apply=True, max_pixel_value=255
            )
        )

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
        """trims first and last % imgs equal to trim_p"""
        remove_n = imgs.shape[0] // int(trim_p * 100)
        return imgs[remove_n:-remove_n, :, :]

    def standardize(self, imgs):
        return (imgs - self.mean) / self.sd

    def __call__(self, imgs: np.ndarray) -> torch.Tensor:

        if self.trim_p > 0.0:
            imgs = self.trim_imgs(imgs, self.trim_p)

        # Rescale intensities to range between 0 and 255 -> tror ikke den gør noget!
        # imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255
        # imgs = imgs.astype(np.uint8)

        imgs = self.apply_transforms(imgs)

        # imgs = self.standardize(imgs) -> is done using albu now

        # Convert to tensor so randomeerase works
        imgs = torch.from_numpy(imgs).float()

        # randomerasing needs to be implemented after standardization
        re = RandomErasing(
            probability=self.re_p, max_area=0.15, mode="const", device="cpu"
        )

        imgs = re(imgs)

        return imgs
