# Copyright Grantham Taylor.

from pathlib import Path

import flytekit as fl

from windmark.core.constructs.general import FieldRequest
from windmark.core.data.processors import multithread, compare
from windmark.orchestration import environments

# FIXME this should be extended to include other validation checks


@environments.proc
def compare_sequence_lengths(
    lifestreams: fl.FlyteDirectory,
    field: FieldRequest,
    sequence_id: str,
    event_id: str,
):
    """
    Compares the lengths of the inputs and indices in the given lifestreams.

    Args:
        lifestreams (fl.FlyteDirectory): The directory containing the lifestreams.
        field (FieldRequest): The field to compare with the index.
        sequence_id (str): The sequence ID.
        event_id (str): The sequence ID.
    """

    lifestreams.download()

    print(f'- validating field "{field.name}"')
    print(f'- event_id "{event_id}"')
    print(f'- sequence_id "{sequence_id}"')

    path = Path(lifestreams.path)

    multithread(
        process=compare,
        path=path,
        key=field.name,
        is_static=field.is_static,
        sequence_id=sequence_id,
        event_id=event_id,
    )
