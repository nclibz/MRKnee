import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.metrics import MetricLogger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metriclogger: MetricLogger,
        label_smoothing: float,
        device: str = "cuda",
        progressbar: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metriclogger = metriclogger
        self.progressbar = progressbar
        self.label_smoothing = label_smoothing
        self.scaler = GradScaler() # 16bit

    def train(self, dataloader):
        self.model.train()
        for imgs, target, sample_id, weight in tqdm(
            dataloader,
            desc="Training",
            disable=not self.progressbar,
        ):
            imgs, target, weight = (
                imgs.to(self.device),
                target.to(self.device),
                weight.to(self.device),
            )
            output = self.model(imgs)
            loss = self.calculate_loss(output, target, weight)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            pred = torch.sigmoid(output).squeeze(0)
            self.metriclogger.log_step("train", pred, target, loss)
        self.metriclogger.log_epoch("train")

    def validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for imgs, target, sample_id, weight in tqdm(
                dataloader,
                desc="Validating",
                disable=not self.progressbar,
            ):
                imgs, target, weight = (
                    imgs.to(self.device),
                    target.to(self.device),
                    weight.to(self.device),
                )

                output = self.model(imgs)
                loss = self.calculate_loss(output, target, weight)
                self.scheduler.step(loss)
                pred = torch.sigmoid(output).squeeze(0)
                self.metriclogger.log_step("val", pred, target, loss)
            self.metriclogger.log_epoch("val")

    def fit(self, epochs, train_dataloader, val_dataloader):
        for _ in tqdm(range(epochs), desc="Epochs", disable=not self.progressbar):
            self.train(train_dataloader)
            self.validate(val_dataloader)

    def smooth_labels(self, target):
        return target * (1 - 2 * self.label_smoothing) + self.label_smoothing

    def calculate_loss(self, output, target, weight):
        if self.label_smoothing > 0.0:
            target = self.smooth_labels(target)
        loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
        return loss
