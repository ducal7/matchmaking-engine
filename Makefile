.PHONY: sim plots test lint all venv

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	python3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

sim:
	$(PY) -m matchmaking.simulate

plots: sim

test:
	$(VENV)/bin/pytest

lint:
	$(VENV)/bin/ruff check

all: lint test sim
