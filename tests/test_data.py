# %%

import random
from typing import List

import numpy as np
import pandas as pd
import pytest
from src.augmentations import Augmentations
from src.data import OAI, KneeMRI, MRNet

# %%

oai_params = [
    ("train", "meniscus", "coronal"),
    ("train", "meniscus", "sagittal"),
    ("valid", "meniscus", "coronal"),
    ("valid", "meniscus", "sagittal"),
    ("test", "meniscus", "coronal"),
    ("test", "meniscus", "sagittal"),
]


mrnet_params = [
    ("train", "meniscus", "coronal"),
    ("train", "meniscus", "sagittal"),
    ("train", "meniscus", "axial"),
    ("valid", "meniscus", "coronal"),
    ("valid", "meniscus", "sagittal"),
    ("valid", "meniscus", "axial"),
    ("test", "meniscus", "coronal"),
    ("test", "meniscus", "sagittal"),
    ("test", "meniscus", "axial"),
]


# TODO: Could add tests for each augment with a toy 4x4 array?


def test_kneemri_ds_no_augs(augs):
    augs = augs["none"]
    ds = KneeMRI(stage="train", diagnosis="acl", plane="sagittal", clean=False, transforms=augs)

    targets = pd.read_csv(f"data/kneemri/metadata.csv")

    assert len(ds) == len(targets)

    idx = random.randint(0, len(ds))

    ds_case = ds[idx]
    target_case = targets.iloc[idx, :]
    assert ds_case[2] == target_case.volumeFilename[:-4]

    assert ds_case[1].item() == target_case["aclDiagnosis"]


@pytest.mark.parametrize("stage, diagnosis, plane", oai_params)
def test_oai_ds_no_augs(stage, diagnosis, plane, augs):

    augs = augs["none"]

    ds = OAI(stage=stage, diagnosis=diagnosis, plane=plane, clean=False, transforms=augs)
    targets = pd.read_csv(f"data/oai/{stage}-{diagnosis}.csv")

    if plane == "coronal":
        targets = targets[targets.plane == "COR"]
    elif plane == "sagittal":
        targets = targets[targets.plane == "SAG"]

    assert len(ds) == len(targets)

    idx = random.randint(0, len(ds))

    ds_case = ds[idx]
    target_case = targets.iloc[idx, :]
    assert ds_case[2] == target_case.fname[:-4]

    assert ds_case[1].item() == target_case[diagnosis]


@pytest.mark.parametrize("stage, diagnosis, plane", oai_params)
def test_oai_ds_all_augs(stage, diagnosis, plane, augs):

    augs = augs["all"]

    ds = OAI(stage=stage, diagnosis=diagnosis, plane=plane, clean=False, transforms=augs)

    idx = random.randint(0, len(ds))

    ds[idx]


@pytest.mark.parametrize("stage, diagnosis, plane", mrnet_params)
def test_mrnet_ds_no_augs(stage, diagnosis, plane, augs):

    augs = augs["none"]

    ds = MRNet(stage=stage, diagnosis=diagnosis, plane=plane, clean=False, transforms=augs)
    targets = pd.read_csv(f"data/mrnet/{stage}-{diagnosis}.csv", header=None)

    assert len(ds) == len(targets)

    idx = random.randint(0, len(ds))

    batch = ds[idx]

    imgs = batch[0]
    img_mean = imgs.mean().item()
    img_std = imgs.std().item()
    assert -0.15 <= img_mean <= 0.15
    assert 0.9 <= img_std <= 1.1


@pytest.mark.parametrize(
    "stage, diagnosis, plane",
    [
        ("train", "meniscus", "axial"),
        ("train", "meniscus", "coronal"),
        ("train", "meniscus", "sagittal"),
    ],
)
def test_mrnet_stats(stage, diagnosis, plane, augs):

    augs = augs["none"]

    ds = MRNet(stage=stage, diagnosis=diagnosis, plane=plane, clean=False, transforms=augs)
    targets = pd.read_csv(f"data/mrnet/{stage}-{diagnosis}.csv", header=None)

    assert len(ds) == len(targets)

    idx = random.randint(0, len(ds))

    batch = ds[idx]

    imgs = batch[0]
    img_mean = imgs.mean().item()
    img_std = imgs.std().item()
    # assert -0.15 <= img_mean <= 0.15
    # assert 0.85 <= img_std <= 1.15


# %%
