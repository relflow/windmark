from typing import TypeVar, Iterator
import json
import os

import pydantic
import valkey
import litserve as ls
from tensordict import TensorDict
import torch
from beartype import beartype

from core.architecture.embedders import FieldInterface
from core.architecture.encoders import SequenceModule
from core.constructs.managers import StashManager
from core.data.operators import transform

NULL = "[UNK]"

Context = dict[str, list[int|float|None] | list[str|None]]

@beartype
def schema(stash: StashManager) -> type:
    """Create a pydantic schema from the structure of the stash manager."""

    annotations: dict[str, TypeVar] = {}
    for field in stash.manager.structure.fields:
        tensorfield = FieldInterface.tensorfield(field)
        annotations[field.name] = tensorfield.get_annotation(manager=stash.manager, field=field)

    return pydantic.create_model("Fields", **annotations)


def query(request: "Request", deployment: "Deployment") -> Context:
    """Store a request in the database and retrieve the context."""

    # replace None values with NULL
    # it is annoying that redis does not support None values
    request.fields = {key: (value if value is not None else NULL) for key, value in request.fields.items()}
    
    # add the dynamic fields to the database
    deployment.db.xadd(**request.dict(), maxlen=deployment.limit, limit=deployment.limit)

    # retrieve the most recent records from the database
    records: list[dict[str, list]] = deployment.db.xrevrange(name=request.name, count=deployment.limit)

    # forcibly cap the number of records to the limit, and then reverse the context
    # in theory the stream will cap the number of records but we should be safe
    events: Iterator[dict[str, list]] = reversed(records[: min(deployment.limit, len(records))])

    # FIXME this would be improved by using a Redis Pipeline, such that we can batch the requests
    # that is, instead of returning a list of dictionaries, we can return a dictionary of lists
    # we still need to truncate, reverse, and convert to NULL, but that will be a little easier
    # the pipeline should be a separate function

    # TODO additionally we need to add support for static fields, which are not stored in the database
    # presumably they're constant. perhaps we require the request to include them nonetheless.
    
    # convert list of events to dictionary of fields
    context: Context = {}

    for fieldname in deployment.fieldnames:
        context[fieldname] = []
        for event in events:
            value = event[fieldname]
            context[fieldname].append(value if value != NULL else None)
    
    return context


class Deployment(ls.LitAPI):

    """A deployment of the model."""

    def __init__(self, stash: StashManager, connection: valkey.Valkey) -> None:
        """Initialize the deployment."""

        self.stash: StashManager = stash
        self.db: valkey.Valkey = connection
        self.limit: int = stash.params.n_context
        self.fieldnames: list[str] = [field.name for field in stash.manager.structure.fields]
        self.device: torch.device | None = None

    @beartype
    def setup(self, device: torch.device) -> None:
        """Setup the deployment."""

        self.device = device

        self.model = SequenceModule.load_from_checkpoint(
            checkpoint_path=self.stash.checkpoint,
            params=self.stash.params,
            manager=self.stash.manager,
            mode="inference",
        )

        self.model.to(device)

    @beartype
    def decode_request(self, request: "Request") -> TensorDict:
        """Decode a request to the model."""

        # store the request in the database and query the context
        context = query(request, self)

        # encode context to tensordict of tensorfields
        # this same function powers the training pipeline
        return transform(context, self.stash.params, self.stash.manager).to(self.device)

    @torch.no_grad
    @beartype
    def predict(self, data: TensorDict) -> torch.Tensor:
        """Pass the data through the model."""
        output = self.model.forward(data)
        return torch.nn.functional.softmax(output.predictions, dim=1)

    @beartype
    def encode_response(self, response: torch.Tensor) -> list[float]:
        """Encode the response from the model output."""
        return response.squeeze().tolist()

    @beartype
    def batch(self, inputs: list[TensorDict]) -> TensorDict:
        """Batch the inputs."""
        return torch.stack(inputs, dim=0).squeeze(1)

with open(os.getenv("STASH_FILENAME"), "r") as file:

    stash = StashManager(json.load(file))

Fields = schema(stash)

class Request(pydantic.BaseModel):
    """A request to the model."""
    name: str
    fields: Fields # type: ignore


if __name__ == "__main__":
    
    
    connection = valkey.Valkey(**json.loads(os.getenv("CONNECTION")))
    
    deployment = Deployment(stash=stash, connection=connection)

    stash.checkpoint.download()

    port = int(os.getenv("MODEL_PORT"))
    n_workers = int(os.getenv("N_WORKERS"))
    max_batch_size = int(os.getenv("MAX_BATCH_SIZE"))

    server = ls.LitServer(
        deployment,
        accelerator="auto",
        track_requests=True,
        max_batch_size=max_batch_size,
        workers_per_device=n_workers,
    )

    server.run(port=port, generate_client_file=False)
