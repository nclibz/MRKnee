# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader

from src.cnnpredict import CNNPredict
from src.data import KneeMRI, MRNetDataModule, SkmTea, OAI, MRNet
from src.ensamble import Ensamble, collect_predictors
from src.model import MRKnee


# %% LOAD MODELS

acl_predictors_train = collect_predictors(
    "meniscus", ["sagittal"], "train", "data/mrnet", MRNet
)

acl_predictors_val = collect_predictors(
    "meniscus", ["sagittal"], "valid", "data/mrnet", MRNet
)

# %%
### INTERNAL VALIDATION OF ENSAMBLE
acl_ensamble = Ensamble()
acl_ensamble.train(acl_predictors_train)
acl_ensamble.validate(acl_predictors_val)
acl_ensamble.get_metrics()


# %%
predictor = acl_predictors_val[0].make_preds()
predictor.plot_roc()
# %%
### EXTERNAL VALIDATION STAJDUR

# TODO: test if using 2 as 0 gives better auc?

# %%

kneemri_ds = KneeMRI(
    datadir="data/kneemri",
    stage="valid",
    diagnosis="acl",
    plane="sagittal",
    clean=False,
    trim=False,
    transforms=augs,
)

kneemri_dl = DataLoader(
    kneemri_ds,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

predictions = CNNPredict(acl_sag_model, augs, kneemri_dl)
predictions.get_preds()
predictions.plot_roc()
# %%
### EXTERNAL VALIDATION TEASKM
#### ACL

# TODO: n_images er meget høj for det her dataset. Kan jeg trimme mere så jeg kan bruge større img size?
# TODO: SKAL JEG RESIEZ IMGS?
# TODO: normalize intensities ved at bruge histogram fra mrnet train set?

# %%
TRAIN_IMGSIZE, TEST_IMGSIZE = acl_sag_model.get_train_test_imgsize()
TEST_IMGSIZE = (256, 256)

augs = Augmentations(
    train_imgsize=TRAIN_IMGSIZE,
    test_imgsize=TEST_IMGSIZE,
    shift_limit=0.2,
    scale_limit=0.2,
    rotate_limit=0.2,
    ssr_p=0.2,
    clahe_p=0.2,
    reverse_p=0.0,
    indp_normalz=True,
)

skmtea_acl = SkmTea(
    datadir="data/skm-tea",
    stage="valid",
    diagnosis="acl",
    plane="sagittal",
    clean=False,
    trim=True,
    trim_p=0.40,  # TODO: OBS TRIM ER HØJ
    transforms=augs,
)

skmtea_acl_dl = DataLoader(
    skmtea_acl,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

### SKMTEA MENISCUS

skmtea_men = SkmTea(
    datadir="data/skm-tea",
    stage="valid",
    diagnosis="meniscus",
    plane="sagittal",
    clean=False,
    trim=True,
    trim_p=0.40,  # TODO: OBS TRIM ER HØJ
    transforms=augs,
)

skmtea_men_dl = DataLoader(
    skmtea_men,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


skmtea_acl_preds = CNNPredict(acl_sag_model, augs, skmtea_acl_dl)
skmtea_men_preds = CNNPredict(men_sag_model, augs, skmtea_men_dl)
# %%
# skmtea_acl_preds.get_preds()
skmtea_acl_preds.get_preds()
skmtea_acl_preds.plot_roc()

skmtea_men_preds.get_preds()
skmtea_men_preds.plot_roc()

# %%

#### OAI

TRAIN_IMGSIZE, TEST_IMGSIZE = men_cor_model.get_train_test_imgsize()
TEST_IMGSIZE = (256, 256)

augs = Augmentations(
    train_imgsize=TRAIN_IMGSIZE,
    test_imgsize=TEST_IMGSIZE,
    shift_limit=0.2,
    scale_limit=0.2,
    rotate_limit=0.2,
    ssr_p=0.2,
    clahe_p=0.2,
    reverse_p=0.0,
    indp_normalz=True,
)

## MENISCUS COR

oai_men_cor = OAI(
    datadir="data/oai",
    stage="valid",
    diagnosis="meniscus",
    plane="coronal",
    clean=False,
    trim=True,
    trim_p=0.10,
    transforms=augs,
    imgs_in_ram=False,
)

oai_men_cor_dl = DataLoader(
    oai_men_cor,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

oai_men_cor_preds = CNNPredict(men_cor_model, augs, oai_men_cor_dl)
oai_men_cor_preds.get_preds()
oai_men_cor_preds.plot_roc()

# %%
## OAI MENISCUS SAG

oai_men_sag = OAI(
    datadir="data/oai",
    stage="valid",
    diagnosis="meniscus",
    plane="sagittal",
    clean=False,
    trim=True,
    trim_p=0.10,
    transforms=augs,
)

oai_men_sag_dl = DataLoader(
    oai_men_sag,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

oai_men_sag_preds = CNNPredict(men_sag_model, augs, oai_men_sag_dl)
oai_men_sag_preds.get_preds()
oai_men_sag_preds.plot_roc()
# %%

# %%
## ACL SAG

oai_acl_sag = OAI(
    datadir="data/oai",
    stage="valid",
    diagnosis="acl",
    plane="sagittal",
    clean=False,
    trim=True,
    trim_p=0.10,
    transforms=augs,
)

oai_acl_sag_dl = DataLoader(
    oai_acl_sag,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

oai_acl_sag_preds = CNNPredict(acl_sag_model, augs, oai_acl_sag_dl)

# %%
oai_acl_sag_preds.get_preds()
oai_acl_sag_preds.plot_roc()
# %%

oai_acl_cor = OAI(
    datadir="data/oai",
    stage="valid",
    diagnosis="acl",
    plane="coronal",
    clean=False,
    trim=True,
    trim_p=0.10,
    transforms=augs,
)

oai_acl_cor_dl = DataLoader(
    oai_acl_cor,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

oai_acl_cor_preds = CNNPredict(acl_cor_model, augs, oai_acl_cor_dl)

# %%
oai_acl_cor_preds.get_preds()
oai_acl_cor_preds.plot_roc()

# %%
