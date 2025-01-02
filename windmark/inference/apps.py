import flytekit as fl
from union.app import App, Input
import union

NAME = "windmark"
DB_HOST = "localhost"
DB_PORT = "6379"
DB_PASSWORD = "Aa44AAIjcDE5MDBiZjU1MTdlMWQ0OTdkODRjNWQzYjg5MmQ4Yzc5OHAxMA"
MODEL_PORT = "1243"
MAX_BATCH_SIZE = "4"
N_WORKERS = "2"

StashArtifact = union.Artifact(name="Windmark Model State", partition_keys=["label"])
artifact = StashArtifact.query(label="brett-locks", project="windmark", domain="development")
image = fl.ImageSpec(name=NAME, builder="union", requirements="requirements.txt")

inputs = [
    Input(name="stash", value=artifact),
    Input(name="valkey-host", value=DB_HOST),
    Input(name="valkey-port", value=DB_PORT),
    Input(name="valkey-password", value=DB_PASSWORD),
    Input(name="port", value=MODEL_PORT),
    Input(name="max_batch_size", value=MAX_BATCH_SIZE),
    Input(name="n_workers", value=N_WORKERS),
]

model = App(
    name="model",
    container_image=image,
    include=["windmark/*"],
    port=int(MODEL_PORT),
    inputs=inputs,
    limits=fl.Resources(cpu="4", mem="16Gi"),
    command=["python", "windmark/inference/serve.py"],
)
