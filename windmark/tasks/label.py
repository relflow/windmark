# Copyright Grantham Taylor.

from faker import Faker
from windmark.orchestration import environments


@environments.lite(cache=False)
def create_experiment_label() -> str:
    """
    Generates a new experiment label.

    Returns:
        str: The generated experiment label in the format.
    """

    fake = Faker()

    return fake.street_name().replace(" ", "-").lower()
