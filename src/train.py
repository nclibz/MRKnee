# %%
from src.model import MRKnee
from src.data import MRKneeDataModule
from src.augmentations import Augmentations
from src.callbacks import Callbacks
import pytorch_lightning as pl
import optuna

pl.seed_everything(123)

# %%

DIAGNOSIS = "acl"
PLANE = "sagittal"
BACKBONE = "tf_mobilenetv3_small_minimal_100"
DATADIR = "data"

# %%
def objective(trial, diagnosis=DIAGNOSIS, plane=PLANE, backbone=BACKBONE, datadir=DATADIR):

    model = MRKnee(
        backbone=backbone,
        drop_rate=0.0,
        final_drop=0.0,
        learning_rate=0.0001,
        log_auc=True,
        log_ind_loss=False,
        adam_wd=0.01,
        max_epochs=20,
        precision=32,
    )

    augs = Augmentations(
        model,
        shift_limit=0.20,
        scale_limit=0.20,
        rotate_limit=30,
        reverse_p=0.5,
        same_range=True,
        indp_normalz=True,
    )

    dm = MRKneeDataModule(
        datadir=datadir,
        diagnosis=diagnosis,
        plane=plane,
        transforms=augs,
        clean=True,
        num_workers=1,
        pin_memory=True,
        trim_train=True,
    )

    # TODO: Lave cfg class?
    cfg = dict()
    cfg.update(model.__dict__)
    cfg.update(augs.__dict__)
    cfg.update(dm.__dict__)

    callbacks = Callbacks(cfg, trial, neptune_name="tester")

    trainer = pl.Trainer(
        gpus=1,
        precision=cfg["precision"],
        max_epochs=cfg["max_epochs"],
        logger=callbacks.get_neptune_logger(),
        log_every_n_steps=100,
        num_sanity_val_steps=0,
        callbacks=callbacks.get_callbacks(),
        progress_bar_refresh_rate=20,
        deterministic=True,
    )

    trainer.fit(model, dm)

    ## UPLOAD BEST CHECKPOINTS TO LOG
    callbacks.upload_best_checkpoints()

    return callbacks.metrics_callback.metrics[-1]["val_loss"].item()


# %%

pruner = optuna.pruners.HyperbandPruner(min_resource=10)
sampler = optuna.samplers.TPESampler(multivariate=True)
storage = optuna.storages.RDBStorage(
    url="mysql+pymysql://admin:Testuser1234@database-1.c17p2riuxscm.us-east-2.rds.amazonaws.com/optuna",
    heartbeat_interval=120,
    grace_period=360,
)
study_name = f"{DIAGNOSIS}_{PLANE}_{BACKBONE}"

study = optuna.create_study(
    storage=storage,
    study_name=study_name,
    load_if_exists=True,
    sampler=sampler,
    pruner=pruner,
    direction="minimize",
)

# study.enqueue_trial({
#    'dropout': 55,
#    'lr': 3.e-4,
#    'rotate': 25,
#    'scale': 8,
#    'shift': 10,
#    'adam_wd': 0.0900
#    })

# %%

study.optimize(objective, n_trials=40, timeout=8 * 60 * 60)

# %%
