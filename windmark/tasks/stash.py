# Copyright Grantham Taylor.

import json

import flytekit as fl

from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.managers import SystemManager, StashManager
from windmark.orchestration.environments import context


@context.default(cache=False)
def stash_model_state(checkpoint: fl.FlyteFile, manager: SystemManager, params: Hyperparameters, label: str):
    """
    Stash state required to reproduce a model.

    Args:
        model (fl.FlyteFile): The model file to pretrain.
        manager (SystemManager): The system state manager.
        params (Hyperparameters): The hyperparameters for pretraining.
        label (str): The name for the experiment.
    """

    stash = StashManager(checkpoint=str(checkpoint), manager=manager, params=params, label=label)

    with open(f"stash/{label}.json", "w") as file:
        json.dump(stash.dict(), file)
