import flytekit as fl
from union.app import App, Input

from core.orchestration.artifacts import StashArtifact
from core.orchestration.environments import image
from core.orchestration import constants


db = App(
    name="Windmark Context Store",
    container_image=fl.ImageSpec(
        builder="union",
        base_image="valkey/valkey:8.0.1",
        packages=["union-runtime==0.1.6"],
    ),
    port=constants.DB_PORT,
    command="valkey-server",
    limits=fl.Resources(cpu="1", mem="4Gi"),
    min_replicas=1,
)

model = App(
    name="Windmark Model Deployment",
    container_image=image,
    include=("./*"),
    inputs=[
        Input(name="stash", value=StashArtifact.query(), auto_download=True),
        Input(name="valkey-server", value=db.query_endpoint(public=False), env_name="VALKEY_HOST"),
        Input(name="port", value=constants.MODEL_PORT, env_name="MODEL_PORT"),
        Input(name="max_batch_size", value=constants.MAX_BATCH_SIZE, env_name="MAX_BATCH_SIZE"),
        Input(name="n_workers", value=constants.N_WORKERS, env_name="N_WORKERS"),
    ],
    limits=fl.Resources(gpu="1", cpu="4", mem="16Gi"),
    command=["python", "windmark/inference/serve.py"],
    port=constants.MODEL_PORT,
)