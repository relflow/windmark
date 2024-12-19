import json
from typing import TypeVar

import pydantic
from hydra.utils import instantiate
import valkey
import litserve as ls
from tensordict import TensorDict
import torch
from beartype import beartype

from windmark.core.architecture.embedders import FieldInterface
from windmark.core.architecture.encoders import SequenceModule
from windmark.core.constructs.managers import StashManager
from windmark.core.data.operators import transform


def schema(stash: StashManager) -> TypeVar:
    annotations: dict[str, TypeVar] = {}
    for field in stash.manager.structure.fields:
        tensorfield = FieldInterface.tensorfield(field)
        annotations[field.name] = tensorfield.get_annotation(manager=stash.manager, field=field)

    return pydantic.create_model("Fields", **annotations)


# FIXME kind of gross, but I need to load the stash to instantiate the API schema
with open("stash/alison-island.json", "r") as file:
    stash = instantiate(json.load(file), _convert_="object", _target_=StashManager)

Fields = schema(stash)


class Request(pydantic.BaseModel):
    name: str
    fields: Fields


class API(ls.LitAPI):
    def __init__(self, stash: StashManager, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.stash = stash
        self.limit: int = stash.params.n_context
        self.fieldnames: list[str] = [field.name for field in stash.manager.structure.fields]

    def setup(self, device: torch.device) -> None:
        self.device = device

        self.model = SequenceModule.load_from_checkpoint(
            checkpoint_path=self.stash.checkpoint,
            params=self.stash.params,
            manager=self.stash.manager,
            mode="inference",
        )

        self.model.to(device)

        self.db = valkey.Valkey(host=self.host, port=self.port, decode_responses=True)

    @beartype
    def decode_request(self, request: Request) -> TensorDict:
        # add the record to the database
        self.db.xadd(**request.dict(), maxlen=self.limit, limit=self.limit)

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
        return response.squeeze().tolist()

    @beartype
    def batch(self, inputs: list[TensorDict]) -> TensorDict:
        return torch.stack(inputs, dim=0).squeeze(1)


if __name__ == "__main__":
    api = API(stash=stash, host="localhost", port=7000)

    server = ls.LitServer(lit_api=api, accelerator="auto", track_requests=True, max_batch_size=2, workers_per_device=4)

    server.run(port=1243, generate_client_file=False)
