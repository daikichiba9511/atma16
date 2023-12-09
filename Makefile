.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPE := child-mind-institute-detect-sleep-states
PYTHONPATH := $(shell pwd)
LINT_SRC := scripts src

.PHONY: setup
setup: ## setup install packages
	@# bootstrap of rye
	@if ! command -v rye > /dev/null 2>&1; then \
		curl -sSf https://rye-up.com/get | bash; \
		echo "source ${HOME}/.rye/env" > ~/.profile; \
	fi;
	@rye sync


.PHONY: lint
lint: ## lint code
	@rye run ruff check --config pyproject.toml $(LINT_SRC)

.PHONY: mypy
mypy: ## typing check
	@rye run mypy --config pyproject.toml $(LINT_SRC)

.PHONY: fmt
fmt: ## auto format
	@rye run ruff --fix --config pyproject.toml $(LINT_SRC)
	@rye run ruff format --config pyproject.toml $(LINT_SRC)

.PHONY: test
test: ## run test with pytest
	@rye run pytest -c tests

.PHONY: clean
clean: ## clean outputs
	@rm -rf ./output/*
	@rm -rf ./wandb
	@rm -rf ./debug
	@rm -rf ./.venv

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
