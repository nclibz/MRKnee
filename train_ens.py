# %%

import lightgbm
from utils import get_preds, compare_clfs

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

import numpy as np

# %%
# ACL

# %%
# training set
acl_train = get_preds('data',
                      'acl',
                      planes=[
                          'axial', 'sagittal', 'coronal'],
                      backbones=['efficientnet_b0',
                                 'efficientnet_b0', 'efficientnet_b1'],
                      stage='train')

X = acl_train.drop(['lbls', 'ids'], axis=1)
y = acl_train['lbls']

# %%
# validation set

acl_val = get_preds('data',
                    'acl',
                    planes=['axial', 'sagittal', 'coronal'],
                    backbones=['efficientnet_b0', 'efficientnet_b0', 'efficientnet_b1'],
                    stage='valid')

X_val = acl_val.drop(['lbls', 'ids'], axis=1)
y_val = acl_val['lbls']


# %%
# tune clfs


# %%
clfs = {"logr": LogisticRegression(), "lgbm": LGBMClassifier()}
compare_clfs(clfs, X, y, X_val, y_val)

# %%


# soft voting clf

X_val = X_val.assign(soft_vote=X_val.mean(axis=1))

X_val['soft_vote'] = np.where(X_val['soft_vote'] > 0.5, 1, 0)

roc_auc_score(y_val, X_val['soft_vote'].to_numpy())

# %%
