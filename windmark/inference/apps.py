import flytekit as fl
from union.app import App, Input

import union


NAME = "windmark"

DB_HOST: str = "localhost"
DB_PORT: int = 6379
DB_PASSWORD: str = "Aa44AAIjcDE5MDBiZjU1MTdlMWQ0OTdkODRjNWQzYjg5MmQ4Yzc5OHAxMA"

MODEL_PORT: int = 1243
MAX_BATCH_SIZE: int = 4
N_WORKERS: int = 2

StashArtifact = union.Artifact(name="Windmark Model State", partition_keys=["label"])

model = App(
    name='model',
    container_image=fl.ImageSpec(name=NAME, builder="union", requirements="requirements.txt"),
    include=[
        "../inference/*",
        "../orchestration/*",
        "../core/*",
    ],
    port=MODEL_PORT,
    inputs=[
        Input(name="stash", value=StashArtifact.query(label="brett-locks", project="windmark", domain="development"), auto_download=True),

        Input(name="valkey-host", value=DB_HOST, env_name="VALKEY_HOST"),
        Input(name="valkey-port", value=DB_PORT, env_name="VALKEY_PORT"),
        Input(name="valkey-password", value=DB_PASSWORD, env_name="VALKEY_PASSWORD"),

        Input(name="port", value=MODEL_PORT, env_name="MODEL_PORT"),
        Input(name="max_batch_size", value=MAX_BATCH_SIZE, env_name="MAX_BATCH_SIZE"),
        Input(name="n_workers", value=N_WORKERS, env_name="N_WORKERS"),
    ],
    limits=fl.Resources(gpu="1", cpu="4", mem="16Gi"),
    command=["python", "inference/serve.py"],
)