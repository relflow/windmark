# Copyright Grantham Taylor.

from windmark.tasks.compare import compare_sequence_lengths as compare
from windmark.tasks.digest import create_digest_centroids_from_lifestream as digest
from windmark.tasks.finetune import finetune_sequence_encoder as finetune
from windmark.tasks.label import create_experiment_label as label
from windmark.tasks.levels import create_unique_levels_from_lifestream as levels
from windmark.tasks.predict import predict_sequence_encoder as predict
from windmark.tasks.pretrain import pretrain_sequence_encoder as pretrain
from windmark.tasks.sample import create_sample_manager as sample
from windmark.tasks.split import create_split_manager as split
from windmark.tasks.stash import stash_model_state as stash
from windmark.tasks.system import create_system_manager as system
from windmark.tasks.task import create_task_manager as task


__all__ = [
    compare,
    digest,
    finetune,
    label,
    levels,
    predict,
    pretrain,
    sample,
    split,
    stash,
    system,
    task,
]
