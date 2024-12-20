# Copyright Grantham Taylor.

import os
from pathlib import Path
import random
from functools import partial
from typing import TypeAlias, Any, Iterator

import msgspec
import torch
from tensordict import TensorDict

from core.constructs.managers import SystemManager
from core.constructs.general import Hyperparameters
from core.architecture.embedders import FieldInterface
from core.constructs.packages import SupervisedData, PretrainingData, SequenceData

from core.data.samplers import Sampler, sample as sample_fn


AnnotationType: TypeAlias = tuple[str, str, int]
FieldType: TypeAlias = dict[str, list[Any] | Any]


def transform(
    fields: FieldType,
    params: Hyperparameters,
    manager: SystemManager,
) -> TensorDict:
    """
    Convert the fields in the observation into tensors using the specified parameters and manager.

    Args:
        observation (FieldType): The observation containing annotations and fields.
        params (Hyperparameters): The hyperparameters to be used for tensor conversion.
        manager (SystemManager): The system manager for accessing the schema.

    Returns:
        tuple[AnnotationType, TensorDict]: The converted annotations and tensor dictionary.

    """

    output = {}

    for field in manager.structure.fields:
        values = fields[field.name]
        tensorfield = FieldInterface.tensorfield(field)
        output[field.name] = tensorfield.new(values=values, field=field, params=params, manager=manager)

    return TensorDict(output, batch_size=1)


def package(
    fields: TensorDict,
    annotation: AnnotationType,
    params: Hyperparameters,
    manager: SystemManager,
    mode: str,
) -> SequenceData:
    """
    Packages the observation data into a SequenceData object based on the specified mode.

    Args:
        observation (tuple[AnnotationType, TensorDict]): The observation data to be packaged.
        params (Hyperparameters): The hyperparameters for packaging.
        manager (SystemManager): The system manager.
        mode (str): The packaging mode. Must be one of "pretrain", "finetune", or "inference".

    Returns:
        SequenceData: The packaged data.

    Raises:
        AssertionError: If the mode is not one of "pretrain", "finetune", or "inference".
    """

    (*meta, label) = annotation

    assert mode in ("pretrain", "finetune", "inference")

    if mode in ["finetune", "inference"]:
        return SupervisedData.new(inputs=fields, targets=torch.tensor(label), meta=tuple(meta))

    N, L = (1, params.n_context)

    targets = {}

    is_event_masked = torch.rand(N, L).lt(params.p_mask_event)

    for field in manager.structure.fields:
        targets[field.name] = fields[field.name].mask(is_event_masked, params=params)

    # pruned_fields = []
    # for pruned_field in pruned_fields:
    #     assert pruned_field in fields.keys(), f'pruned field "{pruned_field}" not found'
    #     fields[pruned_field].prune()

    targets = TensorDict(targets, batch_size=1)

    return PretrainingData.new(inputs=fields, targets=targets, meta=tuple(meta))


def mock(params: Hyperparameters, manager: SystemManager) -> TensorDict:
    """
    Generate mock observations based on the given hyperparameters and system manager.

    Args:
        params (Hyperparameters): The hyperparameters for generating the mock observations.
        manager (SystemManager): The system manager containing the schema and other necessary information.

    Returns:
        TensorDict: A dictionary of mock observations, where each observation is represented as a tensor.

    """
    observations = []

    for _ in range(params.batch_size):
        tree = {}

        for field in manager.structure.fields:
            tensorfield = FieldInterface.tensorfield(field)
            tree[field.name] = tensorfield.mock(field=field, params=params, manager=manager)

        observations.append(TensorDict(tree, batch_size=[1]))

    return torch.stack(observations, dim=0).squeeze(1)


class SequenceDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        datapath: str,
        mode: str,
        params: Hyperparameters,
        manager: SystemManager,
        split: str,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.params = params
        self.manager = manager
        self.split = split

        root = Path(datapath)

        self.filenames = [root / filename for filename in os.listdir(root) if filename.endswith(".ndjson")]
        self.sampler = partial(sample_fn, manager=manager, params=params, split=split, sampler=Sampler[mode].value)

    def __iter__(self):
        return iter(self.sample())

    def sample(self) -> Iterator[SequenceData]:
        n_workers: int = torch.utils.data.get_worker_info().num_workers
        worker: int = torch.utils.data.get_worker_info().id

        random.shuffle(self.filenames)

        for filename in self.filenames:
            if hash(filename) % n_workers != worker:
                continue

            with open(filename, "r") as file:
                shard = file.readlines()

            for line in shard:
                sequence = msgspec.json.decode(line)

                if sequence[self.manager.structure.split_id] != self.split:
                    continue

                observations = self.sampler(sequence=sequence)

                for annotation, fields in observations:
                    tensorfields = transform(fields, self.params, self.manager)

                    yield package(
                        fields=tensorfields,
                        annotation=annotation,
                        params=self.params,
                        manager=self.manager,
                        mode=self.mode,
                    )


def collate(batch: list[SequenceData]) -> SequenceData:
    """
    Collates a batch of SequenceData objects into a single SequenceData object.

    Args:
        batch (list[SequenceData]): A list of SequenceData objects to be collated.

    Returns:
        SequenceData: The collated SequenceData object.

    """
    stacked = torch.stack(batch, dim=0).squeeze(1)
    stacked.meta = [observation.meta for observation in batch]

    return stacked
