JOBS ?=
JOBS_ARG := $(if $(strip $(JOBS)),--jobs $(JOBS),)

.PHONY: build build-full build-basic build-tokenization build-embedding build-tokenization-embedding build-topic check-basic check-tokenization check-embedding check-tokenization-embedding check-topic check-full test

build: build-full

build-full:
	maturin develop --release --locked $(JOBS_ARG)

build-basic:
	maturin develop --no-default-features --locked $(JOBS_ARG)

build-tokenization:
	maturin develop --no-default-features --features tokenization --locked $(JOBS_ARG)

build-embedding:
	maturin develop --no-default-features --features embedding --locked $(JOBS_ARG)

build-tokenization-embedding:
	maturin develop --no-default-features --features tokenization,embedding --locked $(JOBS_ARG)

build-topic:
	maturin develop --no-default-features --features topic-modeling --locked $(JOBS_ARG)

check-basic:
	cargo check --no-default-features --locked $(JOBS_ARG)

check-tokenization:
	cargo check --no-default-features --features tokenization --locked $(JOBS_ARG)

check-embedding:
	cargo check --no-default-features --features embedding --locked $(JOBS_ARG)

check-tokenization-embedding:
	cargo check --no-default-features --features tokenization,embedding --locked $(JOBS_ARG)

check-topic:
	cargo check --no-default-features --features topic-modeling --locked $(JOBS_ARG)

check-full:
	cargo check --all-targets --all-features --locked $(JOBS_ARG)

test:
	pytest -q
