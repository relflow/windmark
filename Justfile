_help:
  @just --list

# run pre-commit checks
check:
  @echo "JUST running pre-commit checks"
  @git add .
  @uv run pre-commit

# compile whitepaper
publish:
  @echo "JUST publishing whitepaper"
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/windmark.pdf  --root=docs

# run training pipeline
train: && clean
  @echo "JUST running model training pipeline"
  @uv run windmark/__main__.py

# reset pyflyte cache
reset:
  @echo "JUST resetting flyte local cache"
  @uv run pyflyte local-cache clear

# run model explorer TUI
explore:
  @echo "JUST running data explorer TUI"
  @uv run windmark/core/dev/explorer.py

# clean up temporary files
clean:
  @echo "JUST cleaning up wandb and hydra outputs"
  @rm -rf wandb
  @rm -rf outputs

# sync pyproject.toml and freeze dependencies
sync:
  @echo "JUST syncing pyproject.toml and freezing dependencies"
  @uv sync
  @uv pip compile pyproject.toml > requirements.txt --python-platform linux

# start valkey database
valkey:
  @echo "JUST starting up local Valkey database"
  @sudo docker pull bitnami/valkey:latest
  @sudo docker run -e VALKEY_PORT_NUMBER=7000 -e ALLOW_EMPTY_PASSWORD=yes -p 7000:7000 bitnami/valkey:latest

# start model deployment
serve:
  @echo "JUST serving model deployment"
  @uv run windmark/inference/serve.py

# submit events to deployment
client:
  @echo "JUST sending mock events to model deployment"
  @uv run windmark/inference/client.py

# run model training in debug mode
debug: && clean
  @echo "JUST debugging model training workflow"
  @uv run windmark data=debug model=debug

upload:
  @echo "JUST uploading ledger to S3"
  @uv run union run --remote windmark/workflows/upload.py upload
