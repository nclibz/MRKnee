# %%
from src.data import OAI
from src.cnnpredict import collect_predictors


oai_men_cor = collect_predictors(
    "meniscus", ["coronal"], "valid", (256, 256), OAI, ckpt_dir="src/models/oai"
)

# %%
oai_men_cor[0].plot_roc()
# %%


oai_men_cor[0].lbls
# %%
from sklearn import metrics 
fpr, tpr, thresholds = metrics.roc_curve(
oai_men_cor[0].lbls, 
oai_men_cor[0].preds)
metrics.auc(fpr, tpr)

# %%
import numpy as np
metrics.roc_auc_score(
oai_men_cor[0].lbls.astype(int), 
oai_men_cor[0].preds)
# %%

from torchmetrics.functional import auroc
import torch
auroc(torch.Tensor(oai_men_cor[0].preds), torch.Tensor(oai_men_cor[0].lbls.astype(int)).to(dtype=torch.long),  pos_label=1)

# %% 

torchmetrics.functional.auroc(preds, target, num_classes=None, pos_label=None, average='macro', max_fpr=None, sample_weights=None)