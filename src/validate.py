# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader

from src.augmentations import Augmentations
from src.cnnpredict import CNNPredict
from src.data import KneeMRI, MRNetDataModule, SkmTea, OAI, MRNet
from src.ensamble import Ensamble
from src.model import MRKnee


# %% LOAD MODELS

diagnosis = "acl"
PLANES = ["sagittal", "coronal", "axial"]


acl_predictors_train = []

for plane in PLANES:
    model = MRKnee.load_from_checkpoint(
        f"src/models/v3/{diagnosis}_{plane}.ckpt",
        backbone="tf_efficientnetv2_s_in21k",
        drop_rate=0.5,
        learning_rate=1e-4,
        adam_wd=0.001,
        max_epochs=20,
        precision=32,
        log_auc=False,
        log_ind_loss=False,
    )

    TRAIN_IMGSIZE, TEST_IMGSIZE = model.get_train_test_imgsize()

    augs = Augmentations(
        train_imgsize=(256, 256),
        test_imgsize=(256, 256),
        shift_limit=0.2,
        scale_limit=0.2,
        rotate_limit=0.2,
        ssr_p=0.2,
        clahe_p=0.2,
        reverse_p=0.0,
        indp_normalz=True,
    )

    ds = MRNet(
        datadir="data/mrnet",
        stage="train",
        diagnosis=diagnosis,
        plane="sagittal",
        clean=False,
        trim=False,
        transforms=augs,
    )

    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    acl_predictors_train.append(CNNPredict(model, dl))


# %%
### INTERNAL VALIDATION OF ENSAMBLE
acl_ensamble = Ensamble()
acl_ensamble.train(acl_predictors_train)

# %%
acl_sag_model = MRKnee.load_from_checkpoint(
    "src/models/v3/acl_sagittal.ckpt",
    backbone="tf_efficientnetv2_s_in21k",
    drop_rate=0.5,
    learning_rate=1e-4,
    adam_wd=0.001,
    max_epochs=20,
    precision=32,
    log_auc=False,
    log_ind_loss=False,
)

acl_cor_model = MRKnee.load_from_checkpoint(
    "src/models/v3/acl_coronal.ckpt",
    backbone="tf_efficientnetv2_s_in21k",
    drop_rate=0.5,
    learning_rate=1e-4,
    adam_wd=0.001,
    max_epochs=20,
    precision=32,
    log_auc=False,
    log_ind_loss=False,
)


men_sag_model = MRKnee.load_from_checkpoint(
    "src/models/v3/meniscus_sagittal.ckpt",
    backbone="tf_efficientnetv2_s_in21k",
    drop_rate=0.5,
    learning_rate=1e-4,
    adam_wd=0.001,
    max_epochs=20,
    precision=32,
    log_auc=False,
    log_ind_loss=False,
)

men_cor_model = MRKnee.load_from_checkpoint(
    "src/models/v3/meniscus_coronal.ckpt",
    backbone="tf_efficientnetv2_s_in21k",
    drop_rate=0.5,
    learning_rate=1e-4,
    adam_wd=0.001,
    max_epochs=20,
    precision=32,
    log_auc=False,
    log_ind_loss=False,
)


# %%
### EXTERNAL VALIDATION STAJDUR

# TODO: test if using 2 as 0 gives better auc?

# %%
TRAIN_IMGSIZE, TEST_IMGSIZE = acl_sag_model.get_train_test_imgsize()
TEST_IMGSIZE = (
    288,
    288,
)  # TODO: bliver nød til at bruge den her size selvom jeg ville kunne brug 320 fordi nogle af billederne er 292 ?? -> ER KUN ET PAR FÅ. EKSKLUDERE?!

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
