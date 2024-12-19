# Copyright Grantham Taylor.

from types import SimpleNamespace

import flytekit as fl

from windmark.orchestration.context import TaskEnvironment

base = TaskEnvironment(
    container_image=fl.ImageSpec(builder="union", requirements="requirements.txt", apt_packages=["build-essential"]),
    secret_requests=[fl.Secret(key="WANDB_API_KEY")],
    cache_version="v0.1.0",
    cache=True,
    retries=3,
)

context = SimpleNamespace(
    default=base,
    lab=base.extend(
        requests=fl.Resources(gpu="1", cpu="32", mem="64Gi"),
        cache_ignore_input_vars=tuple(["label"]),
    ),
)
