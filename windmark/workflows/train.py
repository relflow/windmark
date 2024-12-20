# Copyright Grantham Taylor.

from functools import partial

import flytekit as fl

import tasks as lib
from core.constructs.general import Hyperparameters
from core.constructs.managers import StructureManager
from orchestration import environments


@environments.lite
def fetch() -> fl.FlyteDirectory:
    return fl.FlyteDirectory.from_source("s3://union-oc-production-demo/windmark/trxns/")


@fl.workflow
def train(
    # lifestreams: fl.FlyteDirectory,
    schema: StructureManager,
    params: Hyperparameters,
) -> fl.FlyteFile:
    """
    Trains a model using the provided data and hyperparameters.

    Args:
        schema (StructureManager): The schema manager object.
        params (Hyperparameters): The hyperparameters object.
    """

    lifestreams = fetch()

    label = lib.label()

    fields = lib.fan(schema=schema)

    compare = partial(lib.compare, lifestreams=lifestreams, sequence_id=schema.sequence_id, event_id=schema.event_id)

    fl.map_task(compare)(field=fields)

    split = lib.split(lifestreams=lifestreams, schema=schema)

    task = lib.task(lifestreams=lifestreams, schema=schema, interpolation_rate=params.interpolation_rate)

    sample = lib.sample(task=task, split=split, params=params)

    centroids = fl.map_task(partial(lib.digest, lifestreams=lifestreams))(field=fields)

    levelsets = fl.map_task(partial(lib.levels, lifestreams=lifestreams))(field=fields)

    system = lib.system(schema=schema, task=task, sample=sample, split=split, centroids=centroids, levelsets=levelsets)

    pretrained = lib.pretrain(lifestreams=lifestreams, params=params, manager=system, label=label)

    finetuned = lib.finetune(checkpoint=pretrained, lifestreams=lifestreams, params=params, manager=system, label=label)

    predictions = lib.predict(checkpoint=finetuned, params=params, manager=system, lifestreams=lifestreams, label=label)

    lib.stash(checkpoint=finetuned, manager=system, params=params, label=label)

    return predictions
