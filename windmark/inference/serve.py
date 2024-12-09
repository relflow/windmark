import json
from pydantic.dataclasses import dataclass

from hydra.utils import instantiate
import valkey
import litserve as ls
from tensordict import TensorDict
import torch
from beartype import beartype

from windmark.core.architecture.encoders import SequenceModule
from windmark.core.constructs.managers import StashManager
from windmark.core.data.operators import transform


@dataclass
class Connection:
    host: str
    port: int


@dataclass
class Request:
    name: str
    fields: dict[str, str | int | float]


class API(ls.LitAPI):
    def __init__(self, stash: StashManager, connection: Connection) -> None:
        self.stash = stash

        self.limit: int = stash.params.n_context
        self.fieldnames: list[str] = [field.name for field in stash.manager.schema.fields]
        self.connection: Connection = connection

    def setup(self, device: torch.device) -> None:
        self.device = device

        self.model = SequenceModule.load_from_checkpoint(
            checkpoint_path=self.stash.checkpoint,
            params=self.stash.params,
            manager=self.stash.manager,
            mode="inference",
        )

        self.model.to(device)

        self.db = valkey.Valkey(host=self.connection.host, port=self.connection.port, decode_responses=True)

    @beartype
    def decode_request(self, request: Request) -> TensorDict:
        # add the record to the database
        self.db.xadd(**vars(request), maxlen=self.limit, limit=self.limit)

        # retrieve the most recent records from the database
        records = self.db.xrevrange(name=request.name, count=self.limit)

        # forcibly cap the number of records to the limit, and then reverse the context
        events = reversed(records[: min(self.limit, len(records))])

        # convert list of events to dictionary of fields
        fields = {name: [event[name] for _, event in events] for name in self.fieldnames}

        # encode fields to tensordict of tensorfields
        tensorfields = transform(fields, self.stash.params, self.stash.manager)

        return tensorfields.to(self.device)

    @torch.no_grad
    @beartype
    def predict(self, data: TensorDict) -> torch.Tensor:
        output = self.model.forward(data)

        return torch.nn.functional.softmax(output.predictions, dim=1)

    @beartype
    def encode_response(self, response: torch.Tensor) -> list[float]:
        return response.tolist()

    @beartype
    def batch(self, inputs: list[TensorDict]) -> TensorDict:
        return torch.stack(inputs, dim=0).squeeze(1)


if __name__ == "__main__":
    connection = Connection(host="localhost", port=7000)

    with open("stash/barrett-overpass.json", "r") as file:
        stash = instantiate(json.load(file), _convert_="object", _target_=StashManager)

    api = API(stash=stash, connection=connection)

    server = ls.LitServer(
        lit_api=api,
        accelerator="auto",
        track_requests=True,
        max_batch_size=32,
        workers_per_device=16,
    )

    server.run(port=1243, generate_client_file=False)
