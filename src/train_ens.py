# %%
from src.ensamble import Ensamble
import pandas as pd

# %%
models_dir = "src/models/"
model_name = "baseline"
# %%
acl_chkpts = {
    "axial": models_dir + model_name + "/acl_axial.ckpt",
    "sagittal": models_dir + model_name + "/acl_sagittal.ckpt",
    "coronal": models_dir + model_name + "/acl_coronal.ckpt",
}

men_chkpts = {
    "axial": models_dir + model_name + "/meniscus_axial.ckpt",
    "sagittal": models_dir + model_name + "/meniscus_sagittal.ckpt",
    "coronal": models_dir + model_name + "/meniscus_coronal.ckpt",
}

abn_chkpts = {
    "axial": models_dir + model_name + "/abnormal_axial.ckpt",
    "sagittal": models_dir + model_name + "/abnormal_sagittal.ckpt",
    "coronal": models_dir + model_name + "/abnormal_coronal.ckpt",
}

all_checkpoints = {"acl": acl_chkpts, "meniscus": men_chkpts, "abnormal": abn_chkpts}


# %%
ensambles = [Ensamble(diagnosis, chkpts) for diagnosis, chkpts in all_checkpoints.items()]
# %%
for ensamble in ensambles:
    ensamble.evaluate()
# %%

metric_dfs = [ensamble.get_metrics() for ensamble in ensambles]


all_metrics = pd.concat(metric_dfs)

all_metrics.to_csv("out/metrics/" + model_name + ".csv")

# %%
