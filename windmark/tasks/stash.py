# Copyright Grantham Taylor.

from typing import Annotated
import json

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
) -> Annotated[str, StashArtifact]:
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

    serialized = json.dumps(stash.model_dump(mode="json"))

    print(serialized)

    return StashArtifact.create_from(serialized, label=label)
