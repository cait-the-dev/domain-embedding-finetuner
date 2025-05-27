VENV_DIR ?= .venv
PY      := $(VENV_DIR)/Scripts/python.exe
PIP     := $(VENV_DIR)/Scripts/pip.exe          

CFG     ?= config.yaml

.PHONY: help venv deps all smoke clean format lint

help:  
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?##"}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv:  
	@python -m venv $(VENV_DIR)
	@$(PIP) install -U pip >/dev/null

deps: venv  
	@$(PIP) install -r requirements.txt >/dev/null

all: deps  
	@$(PY) -m src.ingest   --config $(CFG)
	@$(PY) -m src.generate --config $(CFG)
	@$(PY) -m src.finetune --config $(CFG)
	@$(PY) -m src.evaluate --config $(CFG)

smoke: deps 
	@pytest -q tests

format: deps 
	@$(PY) -m black src tests
	@$(PY) -m isort  src tests

lint: deps  
	@$(PY) -m flake8 src tests

clean:      
	@rm -rf $(VENV_DIR) models/ data/pdf_chunks.jsonl data/train.jsonl
	@echo "clean."
