PYTHON ?= python
JOBS ?=
JOBS_ARG := $(if $(strip $(JOBS)),--jobs $(JOBS),)

.PHONY: build build-full build-basic build-tokenization build-embedding build-topic check-basic check-tokenization check-embedding check-topic test timings-full timings-basic timings-tokenization timings-embedding timings-topic

build: build-full

build-full:
	maturin develop --release $(JOBS_ARG)

build-basic:
	maturin develop --no-default-features $(JOBS_ARG)

build-tokenization:
	maturin develop --no-default-features --features tokenization $(JOBS_ARG)

build-embedding:
	maturin develop --no-default-features --features embedding $(JOBS_ARG)

build-topic:
	maturin develop --no-default-features --features topic-modeling $(JOBS_ARG)

check-basic:
	cargo check --no-default-features $(JOBS_ARG)

check-tokenization:
	cargo check --no-default-features --features tokenization $(JOBS_ARG)

check-embedding:
	cargo check --no-default-features --features embedding $(JOBS_ARG)

check-topic:
	cargo check --no-default-features --features topic-modeling $(JOBS_ARG)

test:
	pytest -q

timings-full:
	maturin build --release --timings $(JOBS_ARG)

timings-basic:
	maturin build --no-default-features --timings $(JOBS_ARG)

timings-tokenization:
	maturin build --no-default-features --features tokenization --timings $(JOBS_ARG)

timings-embedding:
	maturin build --no-default-features --features embedding --timings $(JOBS_ARG)

timings-topic:
	maturin build --no-default-features --features topic-modeling --timings $(JOBS_ARG)
