import flytekit as fl
from union.app import App, Input

from windmark.orchestration.artifact import StashArtifact

NAME = "windmark"

MODEL_PORT = 1243
MAX_BATCH_SIZE = 4
N_WORKERS = 1
CONNECTION = "on-collie-11005.upstash.io"


artifact = StashArtifact.query(label="walker-lodge", project="windmark", domain="development")

model = App(
    name="model",
    container_image=fl.ImageSpec(
        name="image",
        builder="union",
        requirements="requirements.txt",
        apt_packages=['build-essential'],
    ),
    port=MODEL_PORT,
    secrets=[
        fl.Secret(key="UPSTASH_PASSWORD", env_var="PASSWORD"),
    ],
    inputs=[
        Input(name="stash", value=artifact, download=True, env_var="STASH_FILENAME"),
        Input(name="connection", value=CONNECTION, env_var="CONNECTION"),
        Input(name="port", value=str(MODEL_PORT), env_var="MODEL_PORT"),
        Input(name="max_batch_size", value=str(MAX_BATCH_SIZE), env_var="MAX_BATCH_SIZE"),
        Input(name="n_workers", value=str(N_WORKERS), env_var="N_WORKERS"),
    ],
    limits=fl.Resources(cpu="4", mem="16Gi"),
    args=["python", "windmark/inference/serve.py"],
    include=["windmark/*"],
)
