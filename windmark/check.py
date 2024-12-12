import json
from typing import TypeVar

import pydantic
from hydra.utils import instantiate

from windmark.core.architecture.embedders import FieldInterface
from windmark.core.constructs.managers import StashManager


with open("stash/alison-island.json", "r") as file:
    stash = instantiate(json.load(file), _convert_="object", _target_=StashManager)

annotations: dict[str, TypeVar] = {}

for field in stash.manager.schema.fields:
    tensorfield = FieldInterface.tensorfield(field)
    annotations[field.name] = tensorfield.get_annotation(manager=stash.manager, field=field)

FieldType: TypeVar = pydantic.create_model("Field", **annotations)
# RequestType: TypeVar = pydantic.create_model("Request", name=(str, ...), fields=(FieldType, ...))

print(FieldType.model_json_schema())
