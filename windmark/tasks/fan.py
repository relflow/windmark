# Copyright Grantham Taylor.

from core.constructs.general import FieldRequest
from core.constructs.managers import StructureManager

from orchestration import environments


# FIXME this should be simplified by flytekit soon
@environments.lite
def fan_out_schema_fields(schema: StructureManager) -> list[FieldRequest]:
    return schema.fields
