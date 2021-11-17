# %%
from src.ensamble import Ensamble
import pandas as pd

# %%
acl_chkpts = {
    "axial": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
    "sagittal": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
    "coronal": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
}

men_chkpts = {
    "axial": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
    "sagittal": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
    "coronal": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
}

abn_chkpts = {
    "axial": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
    "sagittal": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
    "coronal": "src/models/ensamble_test/epoch=04-val_loss=0.27-val_auc=0.96.ckpt",
}

all_checkpoints = {"acl": acl_chkpts, "men": men_chkpts, "abn": abn_chkpts}


# %%

ensambles = [Ensamble(diagnosis, chkpts) for diagnosis, chkpts in all_checkpoints.items()]

trained_ensambles = [ensamble.evaluate() for ensamble in ensambles]
# %%

metric_dfs = [ensamble.get_metrics() for ensamble in trained_ensambles]

all_metrics = pd.concat(metric_dfs)
