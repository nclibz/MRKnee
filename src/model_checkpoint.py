#%%
import pathlib

import torch

# TODO: implement max / min directions


class SaveModelCheckpoint:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_fpath = self.save_dir / "checkpoint.pt"
        self.best_metric = 10

    def check(self, metric, model, optimizer, scheduler, epoch) -> bool:
        is_best = metric < self.best_metric

        if is_best:
            print(f"New best: {metric} vs. {self.best_metric}. Saving model.")
            state_dict = {
                "epoch": epoch,
                "model_wts": model.state_dict(),
                "loss": metric,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(state_dict, self.save_fpath)
            self.best_metric = metric
        return is_best

    def get_checkpoint_path(self) -> str:
        return self.save_fpath.as_posix()
