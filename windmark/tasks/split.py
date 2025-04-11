# Copyright Grantham Taylor.

from pathlib import Path
from collections import Counter
from functools import reduce

import flytekit as fl

from windmark.core.constructs.managers import StructureManager, SplitManager
from windmark.core.data.processors import multithread, count
from windmark.orchestration import environments


@environments.proc
def create_split_manager(lifestreams: fl.FlyteDirectory, schema: StructureManager) -> SplitManager:
    """
    Create a SplitManager to count number of events in each strata.

    Args:
        lifestreams (fl.FlyteDirectory): The directory containing the lifestreams.
        schema (StructureManager): The schema manager object.

    Returns:
        SplitManager: The created SplitManager object.

    Raises:
        ValueError: If any of the splits in the resulting SplitManager are not in ["train", "test", "validate"].
    """

    lifestreams.download()

    path = Path(lifestreams.path)

    results: list[Counter] = multithread(process=count, key=schema.split_id, path=path)

    counter = reduce(lambda a, b: a + b, results)

    splits: dict[str, int] = dict(counter)

    for split in list(splits.keys()):
        if split not in ["train", "test", "validate"]:
            raise ValueError("split must be one of ['train', 'test', 'validate']")

    return SplitManager(
        training=splits["train"],
        validating=splits["validate"],
        testing=splits["test"],
    )
