# %%

import lightgbm
from utils import get_preds, compare_clfs

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

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

X = acl_train.drop('lbls', axis=1)
y = acl_train['lbls']


# validation set

acl_val = get_preds('data',
                    'acl',
                    planes=['axial', 'sagittal', 'coronal'],
                    backbones=['efficientnet_b0',
                               'efficientnet_b0', 'efficientnet_b1'],
                    stage='valid')

X_val = acl_val.drop('lbls', axis=1)
y_val = acl_val['lbls']


# Calculate auc
# %%
clfs = {"logr": LogisticRegression(), "lgbm": LGBMClassifier()}
compare_clfs(clfs, X, y, X_val, y_val)

# %%
