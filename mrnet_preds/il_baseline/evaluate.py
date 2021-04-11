# %%
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
# %%
# abnormality, ACL tear, and meniscal tear

preds = pd.read_csv('predictions.csv', header=None)
# %%

acl_truth = pd.read_csv('/home/nicolai/Desktop/MRKnee/data/valid-acl.csv', header=None)
abn_truth = pd.read_csv(
    '/home/nicolai/Desktop/MRKnee/data/valid-abnormal.csv', header=None)
men_truth = pd.read_csv(
    '/home/nicolai/Desktop/MRKnee/data/valid-meniscus.csv', header=None)

auc_dict = {}
for i, truth in enumerate([abn_truth, acl_truth, men_truth]):
    auc_dict[i] = roc_auc_score(truth[1].to_numpy(), preds[i].to_numpy())

# %%
avg_auc = np.mean(list(auc_dict.values()))
avg_auc
# %%

# %%
