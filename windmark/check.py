import json
from typing import TypeVar

import pydantic
from hydra.utils import instantiate

from core.architecture.embedders import FieldInterface
from core.constructs.managers import StashManager


with open("stash/alison-island.json", "r") as file:
    stash = instantiate(json.load(file), _convert_="object", _target_=StashManager)

annotations: dict[str, TypeVar] = {}

for field in stash.manager.structure.fields:
    tensorfield = FieldInterface.tensorfield(field)
    annotations[field.name] = tensorfield.get_annotation(manager=stash.manager, field=field)

FieldType: TypeVar = pydantic.create_model("Field", **annotations)
# RequestType: TypeVar = pydantic.create_model("Request", name=(str, ...), fields=(FieldType, ...))

print(FieldType.model_json_schema())
