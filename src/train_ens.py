# %%

from sklearn.svm import SVC
import joblib
import pandas as pd
from utils import get_preds, compare_clfs, VotingCLF
from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from skopt.callbacks import DeltaYStopper, CheckpointSaver
from skopt.plots import plot_objective
from skopt.space import Categorical, Integer, Real
from skopt import BayesSearchCV

from joblib import dump

import numpy as np
import os
# %%
%load_ext autoreload
%autoreload 2


# %%
acl_train = pd.read_csv('preds/acl_train.csv')
acl_val = pd.read_csv('preds/acl_train.csv')
# %%
# ACL

# %%
acl_train = get_preds(diagnosis='acl', stage='train')
acl_val = get_preds(diagnosis='acl', stage='valid')

# %%
acl_X = acl_train[['axial', 'sagittal', 'coronal']]
acl_y = acl_train['lbls']

acl_X_val = acl_val[['axial', 'sagittal', 'coronal']]
acl_y_val = acl_val['lbls']


# %%
acl_clf = LogisticRegression()
acl_clf.fit(acl_X, acl_y)
acl_probas = acl_clf.predict_proba(acl_X_val)
acl_auc = roc_auc_score(acl_y_val, acl_probas[:, 1])
print(acl_auc)


# %%
men_train = pd.read_csv('preds/men_train.csv')
men_val = pd.read_csv('preds/men_val.csv')
# %%

# MENISCUS
men_train = get_preds(diagnosis='meniscus',
                      backbones=['efficientnet_b1'] + ['efficientnet_b0']*2,
                      stage='train')
men_val = get_preds(diagnosis='meniscus',
                    backbones=['efficientnet_b1'] + ['efficientnet_b0']*2,
                    stage='valid')

# %%
men_X = men_train[['axial', 'sagittal', 'coronal']]
men_y = men_train['lbls']
men_X_val = men_val[['axial', 'sagittal', 'coronal']]
men_y_val = men_val['lbls']

# %%
men_clf = LogisticRegression()
men_clf.fit(men_X, men_y)
men_probas = men_clf.predict_proba(men_X_val)
men_auc = roc_auc_score(men_y_val, men_probas[:, 1])
print(men_auc)


# %%
abn_train = pd.read_csv('preds/abn_train.csv')
abn_val = pd.read_csv('preds/abn_val.csv')

# %%
# ABNORMAL

abn_train = get_preds(diagnosis='abnormal', stage='train')
abn_val = get_preds(diagnosis='abnormal', stage='valid')

# %%
abn_X = abn_train[['axial', 'sagittal', 'coronal']]
abn_y = abn_train['lbls']
abn_X_val = abn_val[['axial', 'sagittal', 'coronal']]
abn_y_val = abn_val['lbls']

abn_clf = LogisticRegression()
abn_clf.fit(abn_X, abn_y)
abn_probas = abn_clf.predict_proba(abn_X_val)
abn_auc = roc_auc_score(abn_y_val, abn_probas[:, 1])
print(abn_auc)

# %%
log_aucs = [acl_auc, men_auc, abn_auc]
np.mean(log_aucs)

# %%
# SAVE CLFS
for name, clf in {'acl_clf': acl_clf, 'men_clf': men_clf, 'abn_clf': abn_clf}.items():
    dump(clf, f'src/models/{name}.joblib')

# %%
dfs = {
    'acl_train': acl_train,
    'acl_val': acl_val,
    'men_train': men_train,
    'men_val': men_val,
    'abn_train': abn_train,
    'abn_val': abn_val,
}
for name, df in dfs.items():
    df.to_csv(f'preds/{name}.csv')
# %%

# %%


############ TESTER BCV ######################


def fit_bcv(estimator,
            pgrid,
            X,
            y,
            scoring,
            n_iter,
            n_jobs=10,
            fit=True):

    bcv = BayesSearchCV(
        estimator=estimator,
        search_spaces=pgrid,
        scoring=scoring,
        cv=5,
        n_iter=n_iter,
        n_points=2,
        error_score='raise',
        optimizer_kwargs={"initial_point_generator": "lhs"},
        verbose=2,
        n_jobs=n_jobs)

    # callbacks
    callbacks = []
    callbacks.append(DeltaYStopper(delta=0.01, n_best=50))

    # fit
    if fit:
        bcv.fit(X, y, callback=callbacks)
    return bcv


# %%
pgrid_elnet = {
    "l1_ratio": Real(0, 1),
    "C": Real(0.0000001, 1, prior="log-uniform")}


# %%
acl_elnet = LogisticRegression(penalty="elasticnet", solver="saga")
acl_elnet_bcv = fit_bcv(acl_elnet, pgrid_elnet, X=acl_X,
                        y=acl_y, n_iter=100, scoring='roc_auc')

# %%
acl_elnet_bcv.best_score_
acl_elnet_bcv.best_params_
# %%
acl_elnet_probas = acl_elnet_bcv.predict_proba(acl_X_val)
acl_elnet_auc = roc_auc_score(acl_y_val, acl_elnet_probas[:, 1])
acl_elnet_auc
# %%
# MENISCUS

men_elnet = LogisticRegression(penalty="elasticnet", solver="saga")
men_elnet_bcv = fit_bcv(men_elnet, pgrid_elnet, X=men_X,
                        y=men_y, n_iter=100, scoring='roc_auc')

# %%
men_elnet_bcv.best_score_
men_elnet_bcv.best_params_
# %%
men_elnet_probas = men_elnet_bcv.predict_proba(men_X_val)
men_elnet_auc = roc_auc_score(men_y_val, men_elnet_probas[:, 1])
men_elnet_auc

# %%
abn_elnet = LogisticRegression(penalty="elasticnet", solver="saga")
abn_elnet_bcv = fit_bcv(abn_elnet, pgrid_elnet, X=abn_X,
                        y=abn_y, n_iter=100, scoring='roc_auc')

# %%
abn_elnet_bcv.best_score_
abn_elnet_bcv.best_params_
# %%
abn_elnet_probas = abn_elnet_bcv.predict_proba(abn_X_val)
abn_elnet_auc = roc_auc_score(abn_y_val, abn_elnet_probas[:, 1])
abn_elnet_auc

# %%
np.mean([acl_elnet_auc, men_elnet_auc, abn_elnet_auc])
# %%


pgrid_svm = {
    "C": Real(0.0000001, 1, prior="log-uniform")}

acl_svm = svc()

for x, y in zip()
