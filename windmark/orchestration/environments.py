# Copyright Grantham Taylor.

import flytekit as fl
from kubernetes.client.models import (
    V1PodSpec,
    V1Container,
    V1Volume,
    V1EmptyDirVolumeSource,
    V1VolumeMount,
)

from orchestration.context import TaskEnvironment

NAME = "windmark"

image = fl.ImageSpec(name=NAME, builder="union", requirements="requirements.txt")

lite = TaskEnvironment(
    container_image=image,
    requests=fl.Resources(mem="2Gi"),
    retries=3,
    cache=True,
    cache_version="v0.0.3",
    secret_requests=[fl.Secret(key="WANDB_API_KEY")],
    environment={
        "POLARS_ALLOW_FORKING_THREAD": "1",
        "PYTHONUNBUFFERED": "1"
    },
)


proc = lite.extend(
    requests=fl.Resources(cpu="8", mem="16Gi"),
    cache_serialize=True,
)


lab = proc.extend(
    retries=1,
    cache_ignore_input_vars=("label",),
    requests=fl.Resources(gpu="8", cpu="92", mem="256Gi"),
    
    pod_template=fl.PodTemplate(
        primary_container_name=NAME,
        pod_spec=V1PodSpec(
            containers=[V1Container(name=NAME, volume_mounts=[V1VolumeMount(mount_path="/dev/shm", name="dshm")])],
            volumes=[V1Volume(name="dshm", empty_dir=V1EmptyDirVolumeSource(medium="", size_limit="256Gi"))],
        ),
    ),
)
