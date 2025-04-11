# Copyright Grantham Taylor.

from typing import Annotated
import json

import flytekit as fl

from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.managers import SystemManager
from windmark.orchestration.artifact import StashArtifact, StashManager
from windmark.orchestration import environments

# TODO also add model results


@environments.lite
def stash_model_state(
    checkpoint: fl.FlyteFile,
    manager: SystemManager,
    params: Hyperparameters,
    label: str,
) -> Annotated[fl.FlyteFile, StashArtifact]:
    """
    Stash state required to reproduce a model.

    Args:
        checkpoint (fl.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.
        label (str): The name for the experiment.
    """

    stash = StashManager(checkpoint=checkpoint, manager=manager, params=params, label=label)

    serialized = json.dumps(stash.model_dump(mode="json"))
    
    ff = fl.FlyteFile.new("stash.json")
    
    with open(ff.path, "w") as file:
        file.write(serialized)

    return StashArtifact.create_from(ff, label=label)
