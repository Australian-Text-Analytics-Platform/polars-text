PYTHON ?= python

.PHONY: build test

build:
	maturin develop --release

test:
	pytest -q
