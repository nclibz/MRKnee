# %%
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback


# %%


# class MetricsCallback(Callback):
#     """PyTorch Lightning metric callback."""

#     def __init__(self):
#         super().__init__()
#         self.metrics = []

#     def on_validation_end(self, trainer, pl_module):
#         self.metrics.append(trainer.callback_metrics)


class Callbacks:
    def __init__(self, cfg, trial, neptune_name: str):
        self.cfg = cfg
        self.trial = trial
        self.neptune_name = neptune_name

    def get_neptune_logger(self):

        self.neptune_logger = NeptuneLogger(
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNDI5ODUwMzQtOTM0Mi00YTY2LWExYWQtMDNlZDZhY2NlYjUzIn0=",
            params=self.cfg,
            project_name="nclibz/" + self.neptune_name,
            tags=[self.cfg["diagnosis"], self.cfg["plane"]],
        )
        return self.neptune_logger

    def get_callbacks(self):

        # Callbacks
        self.model_checkpoint = ModelCheckpoint(
            dirpath=f"checkpoints/trial{self.trial.number}/",
            filename="{epoch:02d}-{val_loss:.2f}-{val_auc:.2f}",
            verbose=True,
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
            save_weights_only=True,
        )

        self.lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # self.metrics_callback = MetricsCallback()

        self.prune_callback = PyTorchLightningPruningCallback(self.trial, monitor="val_loss")

        return [self.model_checkpoint, self.prune_callback, self.lr_monitor]

    def upload_best_checkpoints(self):
        self.neptune_logger.experiment.set_property(
            "best_val_loss", self.model_checkpoint.best_model_score.tolist()
        )

        for k in self.model_checkpoint.best_k_models.keys():
            model_name = "checkpoints/" + k.split("/")[-1]
            self.neptune_logger.experiment.log_artifact(k, model_name)
