# %% 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations import Augmentations
from src.data import OAI, MRNet
from src.metrics import AUC, Loss, MetricLogger
from src.model import VanillaMRKnee
from src.trainer import Trainer
from src.utils import seed_everything

seed_everything(123)
PLANE = "sagittal"
BACKBONE = 'tf_mobilenetv3_small_minimal_100'
DATASET_CLASS = OAI
%load_ext autoreload
%autoreload 2

augs = Augmentations(
    train_imgsize=(256, 256),
    test_imgsize=(256, 256),
    shift_limit=.10,
    scale_limit=.10,
    rotate_limit=.10,
    ssr_p=.50,
    clahe_p=0.50,
    trim_p=0.0,
)

train_ds = DATASET_CLASS(
    stage="train",
    diagnosis="meniscus",
    plane=PLANE,
    clean=True,
    transforms=augs,
)

train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

val_ds = DATASET_CLASS(
    stage="valid",
    diagnosis="meniscus",
    plane=PLANE,
    clean=True,
    transforms=augs,)

val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

model = VanillaMRKnee(
    BACKBONE,
    drop_rate=0.60,
    )
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        "min",
                                                        patience=4,
                                                        )

metriclogger = MetricLogger(
    train_metrics={
        "train_loss": Loss(),
        "train_auc": AUC(),
    },
    val_metrics={
        "val_loss": Loss(),
        "val_auc": AUC()
    },
)

trainer = Trainer(model, optimizer, scheduler, metriclogger, progressbar = True,)

for epoch in tqdm(range(4), desc='Epochs', disable=True):
    trainer.train(train_dl)
    trainer.validate(val_dl)
    trn_loss = metriclogger.train_loss.epoch_values[-1]
    trn_auc = metriclogger.train_auc.epoch_values[-1]
    val_loss = metriclogger.val_loss.epoch_values[-1]
    val_auc = metriclogger.val_auc.epoch_values[-1]
    print(f'EPOCH: {epoch} train_loss: {trn_loss} train_auc: {trn_auc} val_auc: {val_auc} val_loss: {val_loss}')


# %% 
metriclogger.plot_metrics(['train_loss', 'train_auc', 'val_loss', 'val_auc'])

# %%
