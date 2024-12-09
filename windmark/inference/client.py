# Copyright Grantham Taylor.

import time
from pathlib import Path
import asyncio
import random
import os
from typing import Iterator
from typing_extensions import TypedDict

import aiohttp
from omegaconf import DictConfig
from hydra import compose, initialize
import polars as pl

from windmark.core.constructs.managers import SchemaManager


class Request(TypedDict):
    name: str
    fields: dict[str, str | int | float]


def sample(manifest: DictConfig) -> Iterator[Request]:
    datapath = Path(manifest.data.path)

    shards = [file for file in os.listdir(datapath) if file.endswith("ndjson")]

    schema = SchemaManager.new(**manifest.data.structure)

    names = [field.name for field in schema.fields]

    sequences = pl.read_ndjson(datapath / random.choice(shards)).iter_slices(1)

    for sequence in sequences:
        index = sequence.get_column(schema.sequence_id).item()

        events = sequence.select(*names).explode(pl.all()).iter_slices(1)

        for event in events:
            values, *_ = event.to_dicts()

            for key, value in values.items():
                fieldtype = schema[key].fieldtype
                if (fieldtype.lower() in ["categories", "entities"]) and (value is None):
                    values[key] = "[UNK]"

            yield dict(name=index, fields=values)


async def fetch(session, url, request):
    async with session.post(url, json=request) as response:
        _ = await response.text()
        # print(f"Response: {text}")


async def main():
    config = Path(os.getcwd()) / "windmark" / "config"
    cwd = Path(os.path.realpath(__file__)).parent

    with initialize(version_base=None, config_path=os.path.relpath(config, cwd)):
        manifest = compose(config_name="config")

    count = 0
    start = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for request in sample(manifest):
            count += 1
            task = fetch(session, "http://127.0.0.1:1243/predict", request)
            tasks.append(task)

        await asyncio.gather(*tasks)

    finish = time.time()

    print(f"Seconds / Request: {(finish - start) / count}")


if __name__ == "__main__":
    asyncio.run(main())
