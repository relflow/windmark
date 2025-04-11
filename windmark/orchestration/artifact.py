import flytekit as fl
import pydantic
import union

from windmark.core.constructs.general import Hyperparameters
from windmark.core.constructs.managers import SystemManager

class StashManager(pydantic.BaseModel):
    label: str
    checkpoint: fl.FlyteFile
    manager: SystemManager
    params: Hyperparameters


StashArtifact = union.Artifact(name="windmark", partition_keys=["label"])
