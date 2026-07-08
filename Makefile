.PHONY: build install test test-py test-rust phase0 clean
VENV := .venv/bin

build:            ## build the Rust data generator (release)
	cargo build --release --manifest-path data-gen/Cargo.toml

install: build    ## create the venv and install the JAX stack
	uv venv $(if $(wildcard .venv),,--python-preference only-managed) .venv
	uv pip install --python $(VENV)/python -e ".[dev]"

test: test-py test-rust  ## run all tests

test-py:          ## python tests (model, halting, gradients, data labels)
	$(VENV)/python -m pytest -q

test-rust:        ## rust tests (determinism, BFS-verified labels)
	cargo test --release --manifest-path data-gen/Cargo.toml

phase0:           ## reproduce the Phase-0 calibration + ablation results
	bash scripts/phase0.sh
	$(VENV)/python scripts/ablation_table.py

demo:             ## train one Reverie model on multi-hop reachability (~8 min CPU)
	$(VENV)/python scripts/run.py --method reverie --steps 1000 --hops-mix 2,3,4 \
	  --branch 0 --trap-depth 0 --d-model 128 --layers 2 --out runs/demo.json

clean:
	rm -rf runs/*.json runs/*.log data-gen/target/release/reverie-datagen

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
