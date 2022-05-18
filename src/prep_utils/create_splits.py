# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_SIZE = 0.15

# ER DET FOR HØJT? Giver mig 170 i mrnet val sæt

#### OAI #####
oai = pd.read_csv("data/oai/targets.csv")

# ## Delete patients with
# nuniq = oai.groupby(["id", "side"]).fname.nunique()
# drops = nuniq[nuniq < 2].index.get_level_values(0)
# oai[~oai.id.isin(drops)].to_csv("data/oai/targets.csv", index=False)


# Drop duplicate ids before splitting
men = oai.sort_values("meniscus", ascending=False).drop_duplicates("id")[["id", "meniscus"]]

# Get ids for train, val and test splits
train, test = train_test_split(men, test_size=SPLIT_SIZE, stratify=men.meniscus)
train, val = train_test_split(train, test_size=SPLIT_SIZE, stratify=train.meniscus)

# Create dataframes including all fnames
train_oai = oai[oai.id.isin(train.id)]
val_oai = oai[oai.id.isin(val.id)]
test_oai = oai[oai.id.isin(test.id)]

# %%


# TESTS
train_oai.value_counts("meniscus", normalize=True)
val_oai.value_counts("meniscus", normalize=True)
test_oai.value_counts("meniscus", normalize=True)

id_leakage = any(
    (
        train_oai.id.isin(val_oai.id).any(),
        train_oai.id.isin(test_oai.id).any(),
        test_oai.id.isin(val_oai.id).any(),
    )
)
assert id_leakage == False


# Write to csv
train_oai.to_csv("data/oai/train-meniscus.csv", index=False)
val_oai.to_csv("data/oai/valid-meniscus.csv", index=False)
test_oai.to_csv("data/oai/test-meniscus.csv", index=False)


# %%
#### MRNet ####
mrnet = pd.read_csv(
    "data/mrnet/orig_splits/train-meniscus.csv",
    header=None,
    names=["id", "lbl"],
    dtype={"id": str, "lbl": np.int64},
)

mrn_train, mrn_val = train_test_split(mrnet, test_size=SPLIT_SIZE, stratify=mrnet["lbl"])


# %%
mrn_train.to_csv("data/mrnet/train-meniscus.csv", header=None, index=False)
mrn_val.to_csv("data/mrnet/valid-meniscus.csv", header=None, index=False)

# %%
