import os
from pathlib import Path
from typing import Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichProgressBar

from classy.data.data_modules import ClassyDataModule
from classy.utils.log import get_project_logger

python_logger = get_project_logger(__name__)


def train(conf: DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.training.seed)

    # data module declaration
    pl_data_module: ClassyDataModule = hydra.utils.instantiate(
        conf.data.datamodule,
        external_vocabulary_path=getattr(conf.data, "vocabulary_dir", None),
        _recursive_=False,
    )
    pl_data_module.prepare_data()

    # main module declaration
    pl_module_init = {"_recursive_": False}
    if pl_data_module.vocabulary is not None:
        pl_module_init["vocabulary"] = pl_data_module.vocabulary
    pl_module = hydra.utils.instantiate(conf.model, **pl_module_init)

    # callbacks declaration
    callbacks_store = [RichProgressBar()]

    # lightning callbacks
    if conf.training.early_stopping_callback is not None:
        early_stopping = hydra.utils.instantiate(conf.training.early_stopping_callback)
        callbacks_store.append(early_stopping)

    if conf.training.model_checkpoint_callback is not None:
        model_checkpoint = hydra.utils.instantiate(
            conf.training.model_checkpoint_callback,
            filename="{epoch:02d}-{" + conf.callbacks_monitor + ":.2f}",
        )
        callbacks_store.append(model_checkpoint)

    # model callbacks
    for callback in conf.callbacks:
        if (
            callback["_target_"]
            == "classy.pl_callbacks.prediction.PredictionPLCallback"
            and callback.get("path", None) is None
        ):
            validation_bundle = pl_data_module.train_coordinates.validation_bundle
            dataset_path = list(validation_bundle.items())[0][0]

            python_logger.info(
                f"Callback dataset path automatically set to: {dataset_path}"
            )

            callbacks_store.append(
                hydra.utils.instantiate(
                    callback, path=validation_bundle, _recursive_=False
                )
            )
        else:
            callbacks_store.append(hydra.utils.instantiate(callback, _recursive_=False))

    # logging
    logger = None

    # wandb
    if conf.logging.wandb.use_wandb:
        from pytorch_lightning.loggers import WandbLogger

        wandb_params = dict(
            project=conf.logging.wandb.project_name,
            name=conf.logging.wandb.experiment_name,
            resume="allow",
            id=conf.logging.wandb.run_id,
        )

        allowed_additional_params = {"entity", "group", "tags"}
        wandb_params.update(
            {
                k: v
                for k, v in conf.logging.wandb.items()
                if k in allowed_additional_params
            }
        )

        if conf.logging.wandb.anonymous is not None:
            wandb_params["anonymous"] = "allow"

        logger = WandbLogger(**wandb_params)

        if conf.logging.wandb.run_id is None:
            conf.logging.wandb.run_id = logger.experiment.id

        # learning rate monitor
        learning_rate_monitor = pl.callbacks.LearningRateMonitor(
            logging_interval="step"
        )
        callbacks_store.append(learning_rate_monitor)

    # trainer
    resume_from_checkpoint: Optional[Union[Path, str]] = None
    if conf.training.resume_from is not None:
        resume_from_checkpoint = conf.training.resume_from

    trainer: pl.trainer.Trainer = hydra.utils.instantiate(
        conf.training.pl_trainer,
        callbacks=callbacks_store,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # save resources
    pl_module.save_resources_and_update_config(
        conf=conf,
        working_folder=hydra.utils.get_original_cwd(),
        experiment_folder=os.getcwd(),
        data_module=pl_data_module,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)
