# Copyright Grantham Taylor.

import hydra
import union

from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.managers import StructureManager
from windmark.workflows.train import train

EXECUTE_LOCALLY = False


@hydra.main(version_base=None, config_path="config", config_name="config")
def windmark(config) -> None:
    remote = union.UnionRemote.for_endpoint(
        endpoint="demo.hosted.unionai.cloud", default_domain="development", default_project="windmark"
    )

    inputs = dict(
        # lifestreams=fl.FlyteDirectory(config.data.path),
        schema=StructureManager(**config.data.structure),
        params=Hyperparameters(**config.model),
    )

    if EXECUTE_LOCALLY:
        train(**inputs)

    else:
        registration = remote.fast_register_workflow(train)
        run = remote.execute(registration, inputs=inputs)

        print(run.execution_url)


if __name__ == "__main__":
    windmark()
