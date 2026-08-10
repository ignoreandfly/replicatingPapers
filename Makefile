PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.DEFAULT_GOAL := help

.PHONY: help venv data chance test test-harness test-rung1 overfit plots clean

help:
	@echo "toy-vlm targets"
	@echo "  make venv          create .venv and install pinned deps (cu118)"
	@echo "  make data          generate the rung-0 synthetic dataset (deterministic)"
	@echo "  make chance        print per-axis chance level from the blind baselines"
	@echo "  make test          run the full suite"
	@echo "  make test-harness  run only rung-0 harness tests (these should pass)"
	@echo "  make test-rung1    run only rung-1 tests (these should fail until you write clip.py)"
	@echo "  make overfit       overfit 8 examples with your CLIP (rung-1 definition of done)"
	@echo "  make clean         remove generated data, caches, run artifacts"

# Pins duplicated from pyproject.toml on purpose: `uv pip install .` does not
# apply [tool.uv.sources], and a cu12x torch silently fails to init on driver
# 515. Explicit index + explicit +cu118 tags is the only form that is safe here.
venv:
	uv venv --python 3.11 .venv
	VIRTUAL_ENV=.venv uv pip install --index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cu118 \
		"torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "numpy==2.1.3" \
		"einops==0.8.1" "matplotlib==3.10.0" "pytest==8.3.4" "pyyaml==6.0.2"
	@$(PY) -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

data:
	$(PY) -m src.data.build --out data/shapes

chance: data
	$(PY) -m src.eval.chance --data data/shapes

test:
	$(PYTEST) -q

test-harness:
	$(PYTEST) -q -m "not rung1"

test-rung1:
	$(PYTEST) -q -m rung1

overfit:
	$(PY) -m src.train.overfit8 --data data/shapes --steps 400

plots:
	$(PY) -m src.eval.chance --data data/shapes --plot runs/chance.png

clean:
	rm -rf data/shapes runs .pytest_cache
	find . -path ./.venv -prune -o -name '__pycache__' -type d -print0 | xargs -0 rm -rf
