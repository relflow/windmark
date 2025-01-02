export ENABLE_UNION_SERVING := "1"

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
  @uv run windmark

# run model explorer TUI
explore:
  @echo "JUST running data explorer TUI"
  @uv run windmark/core/dev/explorer.py

# clean up temporary files
clean:
  @echo "JUST cleaning up wandb and hydra outputs"
  @rm -rf outputs

# sync pyproject.toml and freeze dependencies
sync:
  @echo "JUST syncing pyproject.toml and freezing dependencies"
  @uv sync
  @uv pip compile pyproject.toml > requirements.txt --python-platform linux

# start model deployment
serve:
  @echo "JUST serving model deployment"
  @uv run union deploy apps --project windmark --domain development windmark/inference/apps.py model

# submit events to deployment
# client:
#   @echo "JUST sending mock events to model deployment"
#   @uv run windmark/inference/client.py

# run model training in debug mode
debug: && clean
  @echo "JUST debugging model training workflow"
  @uv run windmark data=debug model=debug
