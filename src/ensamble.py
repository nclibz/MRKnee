# %%
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.cnnpredict import CNNPredict
from src.model import MRKnee
from torch.utils.data import DataLoader


class Ensamble:
    def __init__(
        self,
        clf=LogisticRegression(),
    ) -> None:
        self.clf = clf
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def get_data(self, predictors: List[CNNPredict]):
        predictors = [predictor.make_preds() for predictor in predictors]

        X = np.stack([predictor.preds for predictor in predictors], axis=1)

        y = predictors[0].lbls.ravel()
        return X, y

    def train(self, predictors):
        self.X_train, self.y_train = self.get_data(predictors)
        self.clf.fit(self.X_train, self.y_train)

    def validate(self, predictors):
        self.X_val, self.y_val = self.get_data(predictors)
        self.probas = self.clf.predict_proba(self.X_val)
        self.preds = np.argmax(self.probas, axis=-1)

    def get_metrics(self):
        self.auc = roc_auc_score(self.y_val, self.probas[:, 1])
        tn, fp, fn, tp = confusion_matrix(self.y_val, self.preds).ravel()
        self.spec = tn / (tn + fp)
        self.sens = tp / (tp + fn)
        metrics = pd.DataFrame(
            {
                "sens": self.sens,
                "spec": self.spec,
                "auc": self.auc,
            },
            index=[0],
        )
        return metrics

    def dump_model(self, fname: str):
        dump(self.clf, f"out/models/{fname}.joblib")

    def load_model(self, path: str):
        self.clf = load(path)


# %%


def collect_predictors(
    diagnosis: str, planes: List[str], stage: str, datadir: str, ds_class
):

    predictors = []

    for plane in planes:
        model = MRKnee.load_from_checkpoint(
            f"src/models/v3/{diagnosis}_{plane}.ckpt",
            backbone="tf_efficientnetv2_s_in21k",
            drop_rate=0.5,
            learning_rate=1e-4,
            adam_wd=0.001,
            max_epochs=20,
            precision=32,
            log_auc=False,
            log_ind_loss=False,
        )

        ds = ds_class(
            datadir=datadir,
            stage=stage,
            diagnosis=diagnosis,
            plane=plane,
            clean=False,
            trim=False,
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=0.2,
            ssr_p=0.2,
            clahe_p=0.2,
            indp_normalz=True,
            no_augments=True,
        )

        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        predictors.append(CNNPredict(model, dl))
    return predictors
