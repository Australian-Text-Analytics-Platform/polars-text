PYTHON ?= python
JOBS ?= 24

.PHONY: build test timings-full timings-lean

build:
	maturin develop --release

test:
	pytest -q

timings-full:
	maturin build --release --features full --timings=html -j $(JOBS)

timings-lean:
	maturin build --no-default-features --timings=html -j $(JOBS)
