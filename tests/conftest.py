import pytest

from src.augmentations import Augmentations


@pytest.fixture
def augs():
    augs = {}
    augs["none"] = Augmentations(train_imgsize=(256, 256), test_imgsize=(256, 256))
    augs["all"] = Augmentations(
        train_imgsize=(256, 256),
        test_imgsize=(256, 256),
        trim_p=0.0,
        shift_limit=0.5,
        scale_limit=0.5,
        rotate_limit=0.5,
        ssr_p=0.5,
        clahe_p=0.5,
        reverse_p=0,
        indp_normalz=True,
    )
    return augs
