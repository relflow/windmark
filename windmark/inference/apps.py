import json

import flytekit as fl
from union.app import App, Input
import union

NAME = "windmark"

MODEL_PORT = 1243
MAX_BATCH_SIZE = 4
N_WORKERS = 2
CONNECTION = json.dumps({
    "host": "expert-flounder-44600.upstash.io", 
    "port": 6379,
    "password": "Aa44AAIjcDE5MDBiZjU1MTdlMWQ0OTdkODRjNWQzYjg5MmQ4Yzc5OHAxMA",
})

StashArtifact = union.Artifact(name="Windmark Model State", partition_keys=["label"])
artifact = StashArtifact.query(label="gregory-isle", project="windmark", domain="development")


model = App(
    name="model",
    container_image=fl.ImageSpec(name=NAME, builder="union", requirements="requirements.txt"),
    include=["windmark/*"],
    port=MODEL_PORT,
    inputs=[
        Input(name="stash", value=artifact, auto_download=True, env_name="STASH_FILENAME"),
        Input(name="connection", value=CONNECTION),
        Input(name="port", value=str(MODEL_PORT)),
        Input(name="max_batch_size", value=str(MAX_BATCH_SIZE)),
        Input(name="n_workers", value=str(N_WORKERS)),
    ],
    limits=fl.Resources(cpu="4", mem="16Gi"),
    command=["python", "windmark/inference/serve.py"],
)


StashArtifact = union.Artifact(name="Windmark Model State", partition_keys=["label"])
artifact = StashArtifact.query(label="gregory-isle", project="windmark", domain="development")


example = App(
    name="example",
    container_image="356633062068.dkr.ecr.us-east-2.amazonaws.com/union/demo:windmark-_y2t7mUzEhWa8WSFa8mYvw",
    include=["./example.py"],
    port=8000,
    inputs=[
        Input(name="stash", value=artifact, auto_download=True, env_name="STASH_FILENAME"),
    ],
    limits=fl.Resources(cpu="1", mem="4Gi"),
    command=["fastapi", "dev"],
)
