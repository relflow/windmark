# Copyright Grantham Taylor.

from windmark.orchestration.environments import context
from windmark.core.constructs.general import Centroid, LevelSet
from windmark.core.constructs.managers import (
    StructureManager,
    SupervisedTaskManager,
    SampleManager,
    SplitManager,
    SystemManager,
    CentroidManager,
    LevelManager,
)


@context.default
def create_system_manager(
    schema: StructureManager,
    task: SupervisedTaskManager,
    sample: SampleManager,
    split: SplitManager,
    centroids: list[Centroid | None],
    levelsets: list[LevelSet | None],
) -> SystemManager:
    """
    Creates a system state manager to contain multiple child state managers.

    Args:
        structure (StructureManager): The schema manager object.
        task (SupervisedTaskManager): The supervised task manager object.
        sample (SampleManager): The sample manager object.
        split (SplitManager): The split manager object.
        centroids (list[Centroid | None]): The centroid objects.
        levelsets (list[LevelSet | None]): The levelset objects.

    Returns:
        SystemManager: The created system state manager.
    """
    manager = SystemManager(
        structure=schema,
        task=task,
        sample=sample,
        split=split,
        centroids=CentroidManager(centroids=centroids),
        levelsets=LevelManager(levelsets=levelsets),
    )

    manager.show()

    return manager
