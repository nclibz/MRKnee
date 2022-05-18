# %%
import pathlib

import numpy as np
import pandas as pd

moaks_1k = "kmri_sq_moaks_bicl00"
moaks_600 = "kmri_fnih_sq_moaks_bicl00"

metadata_path = (
    pathlib.Path("data/oai/metadata/MR Image Assessment_SAS/Semi-Quant Scoring_SAS/")
    / f"{moaks_600}.sas7bdat"
)

df = pd.read_sas(metadata_path)


men_vars = ["V00MMTMA", "V00MMTMB", "V00MMTMP", "V00MMTLA", "V00MMTLB", "V00MMTLP"]
men_tear_vars = ["V00MMRTM", "V00MMRTL"]
acl_vars = ["V00MACLTR"]
id_vars = ["ID", "SIDE", "VERSION", "READPRJ"]
cols = id_vars + men_vars + men_tear_vars + acl_vars

targets = df[cols].assign(
    meniscus=np.select(
        [
            df[men_vars].isin([2, 3, 4, 5]).any(axis=1),
            df[men_tear_vars].eq(1).any(axis=1),
        ],
        [1, 1],
        0,
    ),
    acl=np.select([df[acl_vars].isin([1, 2]).any(axis=1)], [1], 0),
    ID=df.ID.str.decode("utf-8"),
    VERSION=df.VERSION.str.decode("utf-8"),
    READPRJ=df.READPRJ.str.decode("utf-8"),
)

targets[["ID", "meniscus", "acl"]].to_csv("data/oai/metadata/extracted_lbls.csv", index=False)


# -> UPLOAD TO ERDA FOR EXTRACTING IMGS

# %%

# data codebook
# V00MMTMA MOAKS: medial meniscal morphology - anterior horn
# V00MMTMB MOAKS: medial meniscal morphology - body
# V00MMTMP MOAKS: medial meniscal morphology - posterior horn
# V00MMTLA MOAKS: lateral meniscal morphology - anterior horn
# V00MMTLB MOAKS: lateral meniscal morphology - body
# V00MMTLP MOAKS: lateral meniscal morphology - posterior horn
# V00MMRTM MOAKS: medial meniscal morphology - posterior root tear
# V00MMRTL MOAKS: lateral meniscal morphology - posterior root tear

# MOAKS
# 2.4.4 Meniscus
# Abnormalities of the meniscus are scores as follows
# 0: normal meniscus
# 1: signal abnormality that is not severe enough to be considered a meniscal tear
# 2: radial tear
# 3: horizontal tear
# 4: vertical tear
# 5: complex tear
# 6: partial maceration
# 7: progressive partial maceration (only used for follow-up visit scores)
# 8: complete maceration


# %%
