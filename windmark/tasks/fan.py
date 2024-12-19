# Copyright Grantham Taylor.

from windmark.core.constructs.general import FieldRequest
from windmark.core.constructs.managers import StructureManager

from windmark.orchestration.environments import context


# FIXME this should be simplified by flytekit soon
@context.default
def fan_out_schema_fields(schema: StructureManager) -> list[FieldRequest]:
    return schema.fields
