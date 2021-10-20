import optuna


class Study:
    def __init__(
        self, diagnosis: str, plane: str, backbone: str, n_warmup_steps: int, threshold: float
    ) -> None:
        self.study_name = f"{diagnosis}_{plane}_{backbone}"
        self.pruner = optuna.pruners.ThresholdPruner(
            lower=None, upper=threshold, n_warmup_steps=n_warmup_steps, interval_steps=1
        )
        self.sampler = optuna.samplers.TPESampler(multivariate=True)
        self.storage = optuna.storages.RDBStorage(
            url="mysql+pymysql://admin:Testuser1234@database-1.c17p2riuxscm.us-east-2.rds.amazonaws.com/optuna",
            heartbeat_interval=120,
            grace_period=360,
        )
        self.study = optuna.create_study(
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            sampler=self.sampler,
            pruner=self.pruner,
            direction="minimize",
        )

    def enqueue(self, dict):
        self.study.enqueue_trial(dict)

    def optimize(self, objective, n_trials):
        self.study.optimize(objective, n_trials=n_trials)
