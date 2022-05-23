# %%
import torch
from dotenv import dotenv_values
from madgrad import MADGRAD
from tqdm import tqdm

import wandb
from src.augmentations import Augmentations
from src.data import OAI, MRNet, get_dataloader
from src.effnet3d import EfficientNetBN
from src.metrics import AUC, Loss, MetricLogger
from src.model import VanillaMRKnee
from src.model_checkpoint import SaveModelCheckpoint
from src.trainer import Trainer
from src.utils import seed_everything

ENV = dotenv_values()
seed_everything(123)


# %%

from src.effnet3d import EfficientNetBN

# %%
wandb.login(key=ENV["WANDB_API_KEY"])

#%%
##### CONFIG

CFG = {
    "plane": "sagittal",
    "backbone": "tf_mobilenetv3_small_minimal_100",
    "protocol": "DESS",
    "dataset": "oai",
    "n_epochs": 15,
}
# %%%

augs = Augmentations(
    ssr_p=0.50,
    shift_limit=0.05,
    scale_limit=0.05,
    rotate_limit=0.05,
    bc_p=0.00,
    brigthness_limit=0.10,
    contrast_limit=0.10,
    re_p=0.0,
    clahe_p=0.50,
    trim_p=0.0,
)

if CFG["dataset"] == "oai":
    DATAREADER = OAI
elif CFG["dataset"] == "mrnet":
    DATAREADER = MRNet

# TODO: flytte dr loading ind i get_dataloader

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

train_dl = get_dataloader(train_dr, augs)
val_dl = get_dataloader(val_dr, augs)

model = VanillaMRKnee(CFG["backbone"], pretrained=True, drop_rate=0.7)

optimizer = MADGRAD(model.parameters(), lr=1e-4, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    "min",
    patience=4,
)

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
    label_smoothing=0.05,
    progressbar=True,
)

wandb.init(project="my-test-project", entity="nclibz", config=CFG)

### TRAINING

for epoch in range(CFG["n_epochs"]):
    trainer.train(train_dl)
    trainer.validate(val_dl)

    metrics = {k: metriclogger.get_metric(k, epoch) for k in metriclogger.all_metrics}

    wandb.log(metrics)

    is_best = chpkt.check(metrics["val_loss"], model, optimizer, scheduler, epoch)
    if is_best:
        wandb.save(chpkt.get_checkpoint_path())

    # TODO: Flytte print af metrics ind i metriclogger?
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
