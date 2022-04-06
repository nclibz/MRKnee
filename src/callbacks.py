# %%
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


# %%


class Callbacks:
    def __init__(self, cfg, trial=None, neptune_name: str = None):
        self.cfg = cfg
        self.trial = trial
        self.neptune_name = neptune_name
        self.neptune_logger = None
        self.model_checkpoint = None

    def get_neptune_logger(self):

        self.neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNDI5ODUwMzQtOTM0Mi00YTY2LWExYWQtMDNlZDZhY2NlYjUzIn0=",
            project="nclibz/" + self.neptune_name,
            tags=[self.cfg["diagnosis"], self.cfg["plane"]],
            log_model_checkpoints=True,
        )

        self.neptune_logger.log_hyperparams(params=self.cfg)

        return self.neptune_logger

    def get_callbacks(self):
        callbacks = []
        # Callbacks
        self.model_checkpoint = ModelCheckpoint(
            dirpath=f"checkpoints/",
            filename="{epoch:02d}-{val_loss:.2f}-{val_auc:.2f}",
            verbose=True,
            save_top_k=2,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
            save_weights_only=True,
        )
        callbacks.append(self.model_checkpoint)

        self.lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(self.lr_monitor)

        if self.trial:
            self.prune_callback = PyTorchLightningPruningCallback(
                self.trial, monitor="val_loss"
            )
            callbacks.append(self.prune_callback)

        return callbacks

    # def upload_best_checkpoints(self):
    #     self.neptune_logger.experiment.set_property(
    #         "best_val_loss", self.model_checkpoint.best_model_score.tolist()
    #     )

    #     for k in self.model_checkpoint.best_k_models.keys():
    #         model_name = "checkpoints/" + k.split("/")[-1]
    #         self.neptune_logger.experiment.log_artifact(k, model_name)
