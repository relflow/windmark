_help:
  @just --list

# run pre-commit checks
check: publish
  @git add .
  @uv run pre-commit

# compile whitepaper
publish:
  @typst compile docs/whitepaper/whitepaper.typ docs/whitepaper/windmark.pdf  --root=docs

# run training pipeline
train: freeze && clean
  @uv run -m windmark

# reset pyflyte cache
reset:
  @uv run pyflyte local-cache clear

# run model explorer TUI
explore:
  @uv run -m windmark.core.dev.explorer

# clean up temporary files
clean:
  @rm -rf wandb
  @rm -rf outputs

# freeze python dependencies to requirements.txt
freeze:
  @uv pip compile pyproject.toml > requirements.txt --python-platform linux

# start valkey database
valkey:
  @sudo docker pull bitnami/valkey:latest
  @sudo docker run -e VALKEY_PORT_NUMBER=7000 -e ALLOW_EMPTY_PASSWORD=yes -p 7000:7000 bitnami/valkey:latest

# start model deployment
serve:
  @uv run -m windmark.inference.serve

# submit events to deployment
client:
  @uv run -m windmark.inference.client

# run model training in debug mode
debug: freeze && clean
  @uv run -m windmark data=debug model=debug
