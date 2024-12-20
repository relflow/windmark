from union import Resources
from union.app import App

PORT = 7000

app = App(
    name="Windmark Context State",
    container_image="bitnami/valkey:latest",
    command=[
        "VALKEY_PORT_NUMBER",
        str(PORT),
        "ALLOW_EMPTY_PASSWORD",
        "yes",
    ],
    port=PORT,
    limits=Resources(cpu="4", mem="4Gi"),
)
