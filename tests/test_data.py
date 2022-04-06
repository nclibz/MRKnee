# %%

from typing import List
import pytest
from src.augmentations import Augmentations
import pandas as pd
from src.data import OAI, KneeMRI
import random
import numpy as np

# %%

oai_params = [
    ("train", "acl", "coronal"),
    ("train", "acl", "sagittal"),
    ("valid", "acl", "coronal"),
    ("valid", "acl", "sagittal"),
    ("test", "acl", "coronal"),
    ("test", "acl", "sagittal"),
    ("train", "meniscus", "coronal"),
    ("train", "meniscus", "sagittal"),
    ("valid", "meniscus", "coronal"),
    ("valid", "meniscus", "sagittal"),
    ("test", "meniscus", "coronal"),
    ("test", "meniscus", "sagittal"),
]

# TODO: Could add tests for each augment with a toy 4x4 array?


def test_kneemri_ds_no_augs(augs):
    augs = augs["none"]
    ds = KneeMRI(
        stage="train", diagnosis="acl", plane="sagittal", clean=False, transforms=augs
    )

    targets = pd.read_csv(f"data/kneemri/metadata.csv")

    assert len(ds) == len(targets)

    idx = random.randint(0, len(ds))

    ds_case = ds[idx]
    target_case = targets.iloc[idx, :]
    assert ds_case[2] == target_case.volumeFilename[:-4]

    assert ds_case[1].item() == target_case["aclDiagnosis"]


def test_oai_ds_no_augs(stage, diagnosis, plane, augs):

    augs = augs["none"]

    ds = OAI(
        stage=stage, diagnosis=diagnosis, plane=plane, clean=False, transforms=augs
    )
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

    ds = OAI(
        stage=stage, diagnosis=diagnosis, plane=plane, clean=False, transforms=augs
    )

    idx = random.randint(0, len(ds))

    ds[idx]


def test_oai_ds_imgs_in_ram(augs):

    augs = augs["none"]

    ds = OAI(
        stage="valid",
        diagnosis="acl",
        plane="sagittal",
        clean=False,
        transforms=augs,
        imgs_in_ram=True,
    )

    ds[0]


# %%
# targets = pd.read_csv("data/oai/train-acl.csv")

# cor_fnames = targets[targets.plane == "COR"]["fname"].to_list()
# sag_fnames = targets[targets.plane == "SAG"]["fname"].to_list()


# imgs = [np.load("data/oai/imgs/" + fname) for fname in sag_fnames]
# # %%

# sum = 0
# meansq = 0
# count = 0

# for a in imgs:
#     mask = a != 0.0
#     data = a[mask]
#     sum += data.sum()
#     meansq = meansq + (data**2).sum()
#     count += data.shape[0]

# total_mean = sum / count
# total_var = (meansq / count) - (total_mean**2)
# total_std = np.sqrt(total_var)
# print("mean: " + str(total_mean))
# print("std: " + str(total_std))

# %%
