# %%
import torch
from madgrad import MADGRAD
from torch.utils.data import DataLoader

from src.augmentations import Augmentations
from src.data import DS, OAI, MRNet, get_dataloader
from src.effnet3d import EfficientNetBN
from src.metrics import AUC, Loss, MetricLogger
from src.model import VanillaMRKnee
from src.model_checkpoint import SaveModelCheckpoint
from src.trainer import Trainer
from src.utils import seed_everything

seed_everything(123)


CFG = {
    "dataset": "mrnet",
    "plane": "coronal",
    "protocol": "TSE",
    "backbone": "efficientnet-b0",  # "tf_efficientnetv2_s_in21k" or "tf_mobilenetv3_small_minimal_100"
    "n_epochs": 3,
    "batch_size": 1,
    "n_trials": 1,
    "use_3d": True,
    "wandb_project": "mrknee",
    "dev_run_samples": 2,
}


augs = Augmentations(
    ssr_p=0,
    shift_limit=0,
    scale_limit=0,
    rotate_limit=0,
    bc_p=0.00,
    brigthness_limit=0.10,
    contrast_limit=0.10,
    re_p=0,
    clahe_p=0,
    trim_p=0.0,
)

if CFG["dataset"] == "oai":
    DATAREADER = OAI
elif CFG["dataset"] == "mrnet":
    DATAREADER = MRNet

### DATASETS AND LOADERS

train_dr = DATAREADER(
    stage="train",
    diagnosis="meniscus",
    plane=CFG["plane"],
    protocol=CFG["protocol"],
    clean=True,
)

val_dr = DATAREADER(
    stage="valid",
    diagnosis="meniscus",
    plane=CFG["plane"],
    protocol=CFG["protocol"],
    clean=False,
)

train_ds = DS(
    train_dr, augs, use_3d=CFG["use_3d"], dev_run_samples=CFG["dev_run_samples"]
)
val_ds = DS(val_dr, augs, use_3d=CFG["use_3d"])

train_dl = DataLoader(
    train_ds,
    batch_size=CFG["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

val_dl = DataLoader(
    val_ds,
    batch_size=CFG["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

### MODELS

DROP_RATE = 0.2

if CFG["use_3d"]:
    model = EfficientNetBN(
        CFG["backbone"],
        spatial_dims=3,
        in_channels=1,
        num_classes=1,
        drop_rate=DROP_RATE,
    )
else:
    model = VanillaMRKnee(CFG["backbone"], pretrained=True, drop_rate=DROP_RATE)


### OPTIMIZERS

LR = 0.001
WD = 0.01
OPTIM_NAME = "adamw"

if OPTIM_NAME == "madgrad":
    optimizer = MADGRAD(model.parameters(), lr=LR, weight_decay=WD)
elif OPTIM_NAME == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

### SCHEDULER
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "min",
    patience=4,
)

## HELPER CLASSES
metriclogger = MetricLogger(
    train_metrics={"train_loss": Loss(), "train_auc": AUC()},
    val_metrics={"val_loss": Loss(), "val_auc": AUC()},
)

chpkt = SaveModelCheckpoint("checkpoint")

trainer = Trainer(
    model,
    optimizer,
    scheduler,
    metriclogger,
    label_smoothing=0,
    progressbar=True,
)


### TRAINING

for epoch in range(CFG["n_epochs"]):
    trainer.train(train_dl)
    trainer.validate(val_dl)

    metrics = {k: metriclogger.get_metric(k, epoch) for k in metriclogger.all_metrics}

    trainer.print_metrics(epoch)


# %%

metriclogger.get_min("val_loss")
# %%
metriclogger.plot_metrics(["train_loss", "train_auc", "val_loss", "val_auc"])

# %%
metrics["val_loss"]
# %%
metriclogger.get_min("val_loss")

# %%


t = torch.Tensor([0.0034, 0.3392])

preds = [t, t]

torch.cat(preds)
# %%
