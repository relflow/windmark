# Copyright Grantham Taylor.

import flytekit as fl

from windmark.orchestration.context import TaskEnvironment

lite = TaskEnvironment(
    container_image=fl.ImageSpec(builder="union", requirements="requirements.txt", apt_packages=["build-essential"]),
    secret_requests=[fl.Secret(key="WANDB_API_KEY")],
    cache_version="v0.0.1",
    cache=True,
    retries=3,
)

proc = lite.extend(requests=fl.Resources(cpu="4", mem="16Gi"))

lab = lite.extend(cache_ignore_input_vars=tuple(["label"]), requests=fl.Resources(gpu="1", cpu="24", mem="48Gi"))
