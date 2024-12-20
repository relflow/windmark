# Copyright Grantham Taylor.

from pathlib import Path
from collections import Counter
from functools import reduce

import flytekit as fl

from core.constructs.managers import StructureManager, SupervisedTaskManager, BalanceManager
from core.data.processors import multithread, count
from orchestration import environments


@environments.proc
def create_task_manager(
    lifestreams: fl.FlyteDirectory, schema: StructureManager, interpolation_rate: float
) -> SupervisedTaskManager:
    """
    Creates a task manager for supervised machine learning tasks.

    Args:
        lifestreams (fl.FlyteDirectory): The directory containing the lifestreams.
        schema (StructureManager): The schema manager for the task.
        interpolation_rate (float): interpolation rate.

    Returns:
        SupervisedTaskManager: The created task manager.
    """

    lifestreams.download()

    path = Path(lifestreams.path)

    results: list[Counter] = multithread(process=count, key=schema.target_id, path=path)

    counter = reduce(lambda a, b: a + b, results)

    targets: dict[str, int] = dict(counter)

    if None in targets.keys():
        unlabeled: int = targets.pop(None)
    else:
        unlabeled: int = 0

    labels: list[str] = list(targets.keys())
    counts: list[int] = list(targets.values())

    balancer = BalanceManager(labels=labels, counts=counts, interpolation_rate=interpolation_rate, unlabeled=unlabeled)

    return SupervisedTaskManager(task="classification", n_targets=len(targets), balancer=balancer)
