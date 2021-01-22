# %%

import lightgbm
from utils import get_preds, compare_clfs, VotingCLF, fit_bcv

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from skopt.space import Categorical, Integer, Real
from skopt import BayesSearchCV
from skopt.callbacks import DeltaYStopper

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
# LGBM
pgrid_lgbm = {"n_estimators": Integer(1, 100),
              "min_child_samples": Integer(20, 200)}


bcv = BayesSearchCV(
    estimator=LGBMClassifier(),
    search_spaces=pgrid_lgbm,
    optimizer_kwargs={"initial_point_generator": "lhs"},
    scoring='roc_auc',
    n_jobs=-1,
    n_points=10,
    n_iter=100,
    error_score='raise',
    verbose=2)

# callbacks
callbacks = [DeltaYStopper(delta=0.01, n_best=5)]

# fit
bcv.fit(X_val, y_val, callback=callbacks)

# %%
clfs = {"logr": LogisticRegression(),
        "lgbm": LGBMClassifier(),
        "hard_vote": VotingCLF(),
        'soft_vote': VotingCLF('soft')}
compare_clfs(clfs, X, y, X_val, y_val)

# %%
# soft voting clf


acl_val.loc[(acl_val['axial'] > 0.5) & (acl_val['sagittal'] > 0.5)
            & (acl_val['coronal'] > 0.5) & (acl_val['lbls'] != 1)]
acl_val.loc[(acl_val['axial'] < 0.5) & (acl_val['sagittal'] < 0.5)
            & (acl_val['coronal'] < 0.5) & (acl_val['lbls'] != 0)]


# %%
