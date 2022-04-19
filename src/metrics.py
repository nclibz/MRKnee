from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from typing import Dict, List
import torch
from torchmetrics.functional import auroc


@dataclass
class Metric(ABC):
    step_values: list[torch.Tensor] = field(default_factory=list)
    epoch_values: list[torch.Tensor] = field(default_factory=list)

    @abstractmethod
    def log_step(self, preds, targets, loss):
        pass

    @abstractmethod
    def log_epoch(self):
        pass


@dataclass
class MetricLogger:
    train_metrics: Dict[str, Metric]
    val_metrics: Dict[str, Metric]

    def __post_init__(self):
        self.set_attributes()

    def set_attributes(self):
        all_metrics = self.train_metrics | self.val_metrics
        for k, v in all_metrics.items():
            setattr(self, k, v)

    def log_step(self, stage, preds, targets, loss):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        _ = [m.log_step(preds, targets, loss) for m in metrics.values()]

    def log_epoch(self, stage):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        _ = [m.log_epoch() for m in metrics.values()]


@dataclass
class AUC(Metric):
    step_preds: List[torch.Tensor] = field(default_factory=list)
    step_targets: List[torch.Tensor] = field(default_factory=list)

    def log_step(self, preds, targets, loss):
        self.step_preds.append(preds)
        self.step_targets.append(targets)

    def log_epoch(self):
        preds = torch.Tensor(self.step_preds)
        targets = torch.Tensor(self.step_targets).long()
        auc = auroc(preds, targets)
        self.epoch_values.append(auc)
        self.step_preds = []
        self.step_targets = []


@dataclass
class Loss(Metric):
    def log_step(self, preds, targets, loss):
        self.step_values.append(loss)

    def log_epoch(self):
        loss = torch.Tensor(self.step_values)
        mean_loss = torch.mean(loss)
        self.epoch_values.append(mean_loss)
        self.step_values = []
