# Copyright Grantham Taylor.

from typing import Annotated

import flytekit as fl

from core.constructs.general import Hyperparameters
from core.constructs.managers import SystemManager, StashManager
from orchestration.artifact import StashArtifact
from orchestration import environments

# TODO also add model results

@environments.lite
def stash_model_state(
    checkpoint: fl.FlyteFile,
    manager: SystemManager,
    params: Hyperparameters,
    label: str,
) -> Annotated[list[StashManager], StashArtifact]:
    """
    Stash state required to reproduce a model.

    Args:
        checkpoint (fl.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.
        label (str): The name for the experiment.
    """

    checkpoint.download()

    stash = StashManager(checkpoint=checkpoint, manager=manager, params=params, label=label)

    # FIXME for whatever reason I need to use a list here
    return StashArtifact.create_from([stash], label=label)
