# %%
from typing import Dict
from src.cnnpredict import CNNPredict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd
from joblib import dump


class Ensamble:
    def __init__(
        self,
        diagnosis,
        chkpts: Dict[str, str],
        backbone: str = "tf_efficientnetv2_s_in21k",
        clf=LogisticRegression(),
    ) -> None:
        self.diagnosis = diagnosis
        self.chkpts = chkpts
        self.backbone = backbone
        self.clf = clf
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def get_data(self, stage: str):
        predictors = [
            CNNPredict(self.diagnosis, plane, path, self.backbone, "data").get_preds(stage)
            for plane, path in self.chkpts.items()
        ]

        X = np.stack([predictor.preds for predictor in predictors], axis=1)

        y = predictors[0].lbls.ravel()
        return X, y

    def evaluate(self):
        print(f"Evaluating: {self.diagnosis}")
        self.X_train, self.y_train = self.get_data("train")
        self.clf.fit(self.X_train, self.y_train)
        self.X_val, self.y_val = self.get_data("valid")
        self.probas = self.clf.predict_proba(self.X_val)
        self.preds = np.argmax(self.probas, axis=-1)
        self.auc = roc_auc_score(self.y_val, self.probas[:, 1])
        tn, fp, fn, tp = confusion_matrix(self.y_val, self.preds).ravel()
        self.spec = tn / (tn + fp)
        self.sens = tp / (tp + fn)

    def get_metrics(self):
        metrics = pd.DataFrame(
            {"diagnosis": self.diagnosis, "sens": self.sens, "spec": self.spec, "auc": self.auc},
            index=[0],
        )
        return metrics

    def dump_model(self, fname: str = None):
        fname = fname if fname is not None else self.diagnosis

        dump(self.clf, f"out/models/{fname}.joblib")

    def __repr__(self) -> str:
        return f"Ensamble({self.diagnosis}, {self.backbone}, {str(self.clf)})"


# %%
