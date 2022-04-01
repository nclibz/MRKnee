# %%
import optuna
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.augmentations import Augmentations
from src.callbacks import Callbacks
from src.data import MRNet, MRNetDataModule
from src.model import MRKnee

pl.seed_everything(123)

# %%

DIAGNOSIS = "acl"
PLANE = "sagittal"
BACKBONE = "tf_mobilenetv3_small_minimal_100"
DATADIR = "data/mrnet"

# %%

model = MRKnee(
    backbone=BACKBONE,
    drop_rate=0.0,
    learning_rate=0.0001,
    log_auc=True,
    log_ind_loss=False,
    adam_wd=0.01,
    max_epochs=20,
    precision=32,
)

TRAIN_IMGSIZE, TEST_IMGSIZE = model.get_train_test_imgsize()


augs = Augmentations(
    train_imgsize=TRAIN_IMGSIZE,
    test_imgsize=TEST_IMGSIZE,
    shift_limit=0.2,
    scale_limit=0.2,
    rotate_limit=0.2,
    ssr_p=0.2,
    clahe_p=0.2,
    reverse_p=0.0,
    indp_normalz=True,
)

dm = MRNetDataModule(
    datadir=DATADIR,
    diagnosis=DIAGNOSIS,
    plane=PLANE,
    transforms=augs,
    clean=True,
    num_workers=1,
    pin_memory=True,
    trim_train=True,
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=2,
    log_every_n_steps=100,
    num_sanity_val_steps=0,
    progress_bar_refresh_rate=20,
)

trainer.fit(model, dm)


# %%
