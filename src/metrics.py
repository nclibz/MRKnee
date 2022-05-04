from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import auroc

# TODO: Implement sensitivity, specificity etc


@dataclass
class Metric(ABC):
    step_values: List[torch.Tensor] = field(default_factory=list)
    epoch_values: List[torch.Tensor] = field(default_factory=list)

    @abstractmethod
    def log_step(self, preds, targets, loss):
        pass

    @abstractmethod
    def log_epoch(self):
        pass


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
        self.df = None


@dataclass
class MetricLogger:
    train_metrics: Dict[str, Metric]
    val_metrics: Dict[str, Metric]

    def __post_init__(self):
        self.all_metrics = {**self.train_metrics, **self.val_metrics}
        self.set_attributes(self.all_metrics)

    def set_attributes(self, metrics: Dict[str, Metric]):
        for k, v in metrics.items():
            setattr(self, k, v)

    def log_step(self, stage, preds, targets, loss):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        _ = [m.log_step(preds, targets, loss) for m in metrics.values()]

    def log_epoch(self, stage):
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        _ = [m.log_epoch() for m in metrics.values()]

    def get_metricsdf(self):
        series = []
        for k, v in self.all_metrics.items():
            series.append(pd.Series(np.array(v.epoch_values), name=k))

        df = pd.concat(series, axis=1).round(3)
        return df

    def plot_metrics(self, metrics: List[str]):
        df = self.get_metricsdf()

        fig, ax = plt.subplots()

        for metric in metrics:
            ax.plot(df.index.to_list(), df[metric], label=metric)

        ax.legend(loc="best")
