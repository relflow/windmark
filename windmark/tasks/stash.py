# Copyright Grantham Taylor.

from typing import Annotated

import union
import flytekit as fl

from core.constructs.general import Hyperparameters
from core.constructs.managers import SystemManager, StashManager
from orchestration import environments


StashArtifact = union.Artifact(name="Windmark Model State")


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
        model (fl.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.
        label (str): The name for the experiment.
    """

    checkpoint.download()

    stash = StashManager(checkpoint=checkpoint, manager=manager, params=params, label=label)

    return StashArtifact.create_from([stash])
