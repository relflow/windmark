# Copyright Grantham Taylor.

import time
from pathlib import Path
import asyncio
import random
import os
from typing import Iterator
from typing_extensions import TypedDict

import flytekit as fl
import aiohttp
from hydra import compose, initialize
import polars as pl

from core.constructs.managers import StructureManager


class Request(TypedDict):
    name: str
    fields: dict[str, str | int | float | None]


def sample(schema: StructureManager, lifestreams: fl.FlyteFile) -> Iterator[Request]:

    shards = [file for file in os.listdir(lifestreams) if file.endswith("ndjson")]

    names = [field.name for field in schema.fields]

    sequences = pl.read_ndjson(lifestreams / random.choice(shards)).iter_slices(1)

    for sequence in sequences:
        index = sequence.get_column(schema.sequence_id).item()

        events = sequence.select(*names).explode(pl.all()).iter_slices(1)

        for event in events:

            values, *_ = event.to_dicts()
            yield dict(name=index, fields=values)


async def fetch(session, url, request):
    async with session.post(url, json=request) as response:
        _ = await response.text()
        # print(f"Response: {text}")


async def request(
    schema: StructureManager,
    lifestreams: fl.FlyteFile,
    deployment: str
):

    count = 0
    start = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for request in sample(schema=schema, lifestreams=lifestreams):
            count += 1
            task = fetch(session, f"{deployment}/predict", request)
            tasks.append(task)

        await asyncio.gather(*tasks)

    finish = time.time()

    print(f"Total Requests: {count}")
    print(f"Total Time: {finish - start} seconds")
    print(f"Seconds / Request: {(finish - start) / count}")


if __name__ == "__main__":
    asyncio.run(request())
