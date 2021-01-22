# %%
from optuna.integration import PyTorchLightningPruningCallback
import optuna
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import MRKnee
from data import MRKneeDataModule
import albumentations as A
from pytorch_lightning import Callback
pl.seed_everything(123)


# %%
%load_ext autoreload
%autoreload 2


# %%
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)
# %%


def objective(trial):

    IMG_SZ = 224  # b0 = 224, b1 = 240,

    cfg = {
        # DATA
        'datadir': 'data',
        'diagnosis': 'meniscus',
        'planes': ['axial'],  # , 'sagittal', 'coronal', 'axial',
        'n_chans': 1,
        'num_workers': 4,
        'pin_memory': True,
        'upsample': False,
        'w_loss': True,
        'indp_normalz': False,
        'transf': {
            'train': [A.Rotate(limit=25, p=1),
                      A.HorizontalFlip(p=0.5),
                      A.RandomCrop(IMG_SZ, IMG_SZ)],
            'valid': [A.CenterCrop(IMG_SZ, IMG_SZ)]
        },
        # MODEL
        'backbone': 'efficientnet_b0',
        'pretrained': True,
        'learning_rate': trial.suggest_loguniform('lr', 1e-6, 1e-2),
        'drop_rate': trial.suggest_float('dropout', 0., 0.8),
        'freeze_from': -1,
        'unfreeze_epoch': 0,
        'log_auc': True,
        'log_ind_loss': True,
        'final_pool': 'max',
        # Trainer
        'precision': 16,
        'max_epochs': 5,
    }

    # LOGGER
    neptune_logger = pl_loggers.NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNDI5ODUwMzQtOTM0Mi00YTY2LWExYWQtMDNlZDZhY2NlYjUzIn0=",
        params=cfg,
        project_name='nclibz/optuna-test',
        tags=[cfg['diagnosis']] + cfg['planes']
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(dirpath=f'checkpoints/trial{trial.number}/',
                                       filename='{epoch:02d}-{val_loss:.2f}-{val_auc:.2f}',
                                       verbose=True,
                                       save_top_k=2,
                                       monitor='val_loss',
                                       mode='min',
                                       period=1)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    metrics_callback = MetricsCallback()

    prune_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    # DM AND MODEL
    dm = MRKneeDataModule(**cfg)
    model = MRKnee(**cfg)
    trainer = pl.Trainer(gpus=1,
                         precision=cfg['precision'],
                         max_epochs=cfg['max_epochs'],
                         logger=neptune_logger,
                         log_every_n_steps=100,
                         num_sanity_val_steps=0,
                         callbacks=[lr_monitor,
                                    model_checkpoint,
                                    metrics_callback,
                                    prune_callback],
                         progress_bar_refresh_rate=20,
                         limit_train_batches=0.10,  # HUSK AT SLETTE
                         deterministic=True)

    trainer.fit(model, dm)

    return metrics_callback.metrics[-1]["val_loss"].item()

# %%

# hvis jeg skal bruge hyperband skal jeg kunne rapportere metrics imens jeg trainer?


pruner = optuna.pruners.MedianPruner()
# skal vel også bruge en TPE sampler?
study = optuna.create_study(direction="minimize", pruner=pruner)

study.optimize(objective, n_trials=10, timeout=600)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# %%
