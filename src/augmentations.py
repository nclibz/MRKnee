import albumentations as A
from numpy.random import default_rng
import numpy as np

# TODO: Set interpolation og border mode for shiftscalerotate


class Augmentations:
    def __init__(
        self,
        model,
        shift_limit,
        scale_limit,
        rotate_limit,
        reverse_p=0.5,
        indp_normalz=True,
    ):
        self.input_size = model.backbone.default_cfg["input_size"]
        self.test_input_size = model.backbone.default_cfg.get("test_input_size", self.input_size)
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
                )
            )

            if plane != "sagittal":
                transforms.append(A.HorizontalFlip(p=0.5))

            transforms.append(A.CenterCrop(self.input_size[1], self.input_size[2]))

        else:
            if self.test_input_size[1] > 256:
                test_input = 256
            else:
                test_input = self.test_input_size[1]

            transforms.append(A.CenterCrop(test_input, test_input))

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
