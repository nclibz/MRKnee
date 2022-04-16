from dataclasses import dataclass
import torch
import torch.nn.functional as F
from src.metrics import AUC, Loss, Metric
from tqdm import tqdm
from src.metrics import MetricLogger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metriclogger: MetricLogger,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metriclogger = metriclogger

    def train(self, dataloader):
        self.model.train()
        for imgs, target, sample_id, weight in dataloader:
            imgs, target, weight = (
                imgs.to(self.device),
                target.to(self.device),
                weight.to(self.device),
            )
            output = self.model(imgs)
            loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            pred = torch.sigmoid(output).squeeze(0)
            self.metriclogger.log_step("train", pred, target, loss)
        self.metriclogger.log_epoch("train")

    def test(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for imgs, target, sample_id, weight in dataloader:
                imgs, target, weight = (
                    imgs.to(self.device),
                    target.to(self.device),
                    weight.to(self.device),
                )

                output = self.model(imgs)
                loss = F.binary_cross_entropy_with_logits(
                    output, target, pos_weight=weight
                )
                self.scheduler.step(loss)
                pred = torch.sigmoid(output).squeeze(0)
                self.metriclogger.log_step("val", pred, target, loss)
            self.metriclogger.log_epoch("val")

    def fit(self, epochs, train_dataloader, val_dataloader):
        for _ in tqdm(range(epochs)):
            self.train(train_dataloader)
            self.test(val_dataloader)
