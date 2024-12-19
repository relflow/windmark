_help:
  @just --list

# run pre-commit checks
check: publish
  @echo "JUST running pre-commit checks"
  @git add .
  @uv run pre-commit

# compile whitepaper
publish:
  @echo "JUST publishing whitepaper"
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/windmark.pdf  --root=docs

# run training pipeline
train: freeze && clean
  @echo "JUST running model training pipeline"
  @uv run -m windmark

# reset pyflyte cache
reset:
  @echo "JUST resetting flyte local cache"
  @uv run pyflyte local-cache clear

# run model explorer TUI
explore:
  @echo "JUST running data explorer TUI"
  @uv run -m windmark.core.dev.explorer

# clean up temporary files
clean:
  @echo "JUST cleaning up wandb and hydra outputs"
  @rm -rf wandb
  @rm -rf outputs

# freeze python dependencies to requirements.txt
freeze:
  @echo "JUST freezing dependencies to requirements.txt"
  @uv pip compile pyproject.toml > requirements.txt --python-platform linux

# start valkey database
valkey:
  @echo "JUST starting up local Valkey database"
  @sudo docker pull bitnami/valkey:latest
  @sudo docker run -e VALKEY_PORT_NUMBER=7000 -e ALLOW_EMPTY_PASSWORD=yes -p 7000:7000 bitnami/valkey:latest

# start model deployment
serve:
  @echo "JUST serving model deployment"
  @uv run -m windmark.inference.serve

# submit events to deployment
client:
  @echo "JUST sending mock events to model deployment"
  @uv run -m windmark.inference.client

# run model training in debug mode
debug: freeze && clean
  @echo "JUST debugging model training workflow"
  @uv run -m windmark data=debug model=debug
