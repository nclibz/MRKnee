# %%
import torch
import pytest

from src.data import MRNet
from src.augmentations import Augmentations
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import MRKnee
from sklearn.metrics import roc_auc_score


# # %%


def predict(model, dl):
    preds_and_lbls = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iter(dl)):
            imgs, label = batch[0].to(device), batch[1].to(device)
            logit = model(imgs)
            preds_and_lbls.append((torch.sigmoid(logit), label))
    preds = torch.tensor([pred for pred, lbl in preds_and_lbls]).cpu().numpy()
    lbls = (
        torch.tensor([lbl for pred, lbl in preds_and_lbls]).cpu().unsqueeze(1).numpy()
    )
    return preds, lbls


model = MRKnee.load_from_checkpoint(
    "src/models/v3/acl_sagittal.ckpt",
    backbone="tf_efficientnetv2_s_in21k",
    drop_rate=0.5,
    learning_rate=1e-4,
    adam_wd=0.001,
    max_epochs=20,
    precision=32,
    log_auc=False,
    log_ind_loss=False,
)

augs = Augmentations(train_imgsize=(256, 256), test_imgsize=(256, 256))

ds = MRNet(
    stage="valid",
    diagnosis="acl",
    plane="sagittal",
    clean=False,
    transforms=augs,
)


dl = DataLoader(
    ds,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
)

# %%
# TEST
def test_acl_sagittal():
    preds, lbls = predict(model, dl)
    assert roc_auc_score(lbls, preds) > 0.96


# %%
