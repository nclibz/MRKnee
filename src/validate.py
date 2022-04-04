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
    "meniscus", ["sagittal"], "train", (256, 256), MRNet
)
# %%
acl_predictors_val = collect_predictors(
    "meniscus", ["sagittal"], "valid", (256, 256), MRNet
)

# %%
### INTERNAL VALIDATION OF ENSAMBLE
acl_ensamble = Ensamble()
acl_ensamble.train(acl_predictors_train)
acl_ensamble.validate(acl_predictors_val)
acl_ensamble.get_metrics()
acl_ensamble.dump_model("data/models/acl_ensamble.pckl")

# %%


# %%
### EXTERNAL VALIDATION STAJDUR
# TODO: test if using 2 as 0 gives better auc?
# TODO: Kan bruge 320 men nogle få er 288 -> Ekskludere?

kneemri_predictors = collect_predictors(
    "acl", ["sagittal"], "valid", (288, 288), KneeMRI
)
kneemri_predictors[0].plot_roc()

# %%
### EXTERNAL VALIDATION TEASKM
# TODO: n_images er meget høj for det her dataset. Kan jeg trimme mere så jeg kan bruge større img size?
# TODO: SKAL JEG RESIEZ IMGS?
# TODO: normalize intensities ved at bruge histogram fra mrnet train set?
# TODO: IKKE KØRT FORDI OUT OF MEMORY. SKAL KUNNE LAVE TRIM FRA COLLECT PREDS

skmtea_acl_sag = collect_predictors("acl", ["sagittal"], "valid", (256, 256), SkmTea)

skmtea_men_sag = collect_predictors(
    "meniscus", ["sagittal"], "valid", (256, 256), SkmTea
)

skmtea_acl_sag[0].plot_roc()
skmtea_men_sag[0].plot_roc()


# %%

#### OAI

oai_men_sag = collect_predictors("meniscus", ["sagittal"], "valid", (256, 256), OAI)
oai_men_sag[0].plot_roc()

# %%
oai_acl_sag = collect_predictors("acl", ["sagittal"], "valid", (256, 256), OAI)
oai_acl_sag[0].plot_roc()


# %%
### TEST SAG-COR ENSAMBLE ON OAI
mrnet_acl_cor_sag_preds = collect_predictors(
    "acl", ["sagittal", "coronal"], "train", (256, 256), MRNet
)
oai_acl_cor_sag_preds = collect_predictors(
    "acl", ["sagittal", "coronal"], "valid", (256, 256), OAI
)

# %%
oai_acl_ensamble = Ensamble()
oai_acl_ensamble.train(mrnet_acl_cor_sag_preds)
oai_acl_ensamble.validate(oai_acl_cor_sag_preds)


# %%
oai_acl_ensamble.get_metrics()
# %%
oai_acl_ensamble.dump_model("src/models/v3/acl_oai_ensamble.joblib")
# %%
