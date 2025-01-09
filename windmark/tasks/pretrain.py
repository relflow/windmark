# Copyright Grantham Taylor.

import flytekit as fl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import NeptuneLogger

from core.architecture.encoders import SequenceModule
from core.architecture.callbacks import FrozenPretraining
from core.constructs.managers import SystemManager
from core.constructs.general import Hyperparameters
from orchestration import environments


@environments.lab
def pretrain_sequence_encoder(
    lifestreams: fl.FlyteDirectory,
    params: Hyperparameters,
    manager: SystemManager,
    label: str,
) -> fl.FlyteFile:
    """
    Pretrains a sequence encoder model using the provided lifestreams, hyperparameters, and system manager.

    Args:
        lifestreams (fl.FlyteDirectory): The directory containing the lifestreams data.
        params (Hyperparameters): The hyperparameters for pretraining.
        manager (SystemManager): The system state manager.
        label (str): The name for the experiment.

    Returns:
        fl.FlyteFile: The path to the best model checkpoint file.
    """

    lifestreams.download()

    key = fl.current_context().secrets.get(key="NEPTUNE_API_KEY")

    torch.set_float32_matmul_precision("medium")
    torch.multiprocessing.set_sharing_strategy("file_system")

    module = SequenceModule(datapath=str(lifestreams.path), params=params, manager=manager, mode="pretrain")

    checkpointer = ModelCheckpoint(
        dirpath=fl.current_context().working_directory, monitor="pretrain-total-validate/loss", filename=label
    )

    trainer = Trainer(
        logger=NeptuneLogger(project="grantham/windmark", name=label, api_key=key),
        precision="bf16-mixed",
        gradient_clip_val=params.gradient_clip_val,
        max_epochs=params.max_pretrain_epochs,
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor="pretrain-total-validate/loss", patience=params.patience),
            FrozenPretraining(),
            checkpointer,
        ],
        limit_train_batches=params.n_pretrain_steps,
    )

    trainer.fit(module)

    module = SequenceModule.load_from_checkpoint(
        checkpoint_path=str(checkpointer.best_model_path),
        datapath=str(lifestreams.path),
        params=params,
        manager=manager,
        mode="pretrain",
    )

    trainer.test(module)

    return fl.FlyteFile(checkpointer.best_model_path)
