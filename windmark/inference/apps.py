import flytekit as fl
from union.app import App, Input
import union

NAME = "windmark"

MODEL_PORT = 1243
MAX_BATCH_SIZE = 4
N_WORKERS = 2
CONNECTION = "expert-flounder-44600.upstash.io:6379"


StashArtifact = union.Artifact(name="windmark", partition_keys=["label"])
artifact = StashArtifact.query(label="stewart-river", project="windmark", domain="development")

model = App(
    name="model",
    container_image=fl.ImageSpec(
        name="image",
        builder="union",
        requirements="uv.lock",
        apt_packages=['build-essential'],
    ),
    include=["windmark/*"],
    port=MODEL_PORT,
    inputs=[
        Input(name="stash", value=artifact, download=True),
        Input(name="connection", value=CONNECTION, env_var="CONNECTION"),
        Input(name="port", value=str(MODEL_PORT), env_var="MODEL_PORT"),
        Input(name="max_batch_size", value=str(MAX_BATCH_SIZE), env_var="MAX_BATCH_SIZE"),
        Input(name="n_workers", value=str(N_WORKERS), env_var="N_WORKERS"),
    ],
    limits=fl.Resources(cpu="4", mem="16Gi"),
    args=["python", "windmark/inference/serve.py"],
)
