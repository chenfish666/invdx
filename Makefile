# Convenience targets. uv owns fdtdx/JAX (see README Environment); the meep
# env is only ever reached through the subprocess bridge (engines/meep_bridge.py).
PY  ?= uv run python
GPU ?= $(if $(INVDX_GPU),$(INVDX_GPU),0)  # CUDA_VISIBLE_DEVICES for GPU-using targets

check:            ## gate G0 only: pure-python unit tests (seconds)
	$(PY) scripts/00_check.py --only unit

gates:            ## all validation gates in order (G0..G5)
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/00_check.py

test:             ## raw pytest (same tests G0 wraps)
	$(PY) -m pytest tests/ -q

smoke:            ## tiny forward fdtdx sim on GPU through config/cli/runio
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/01_smoke_fdtdx.py --tag smoke

smoke-meep:       ## round-trip of the meep-env subprocess bridge
	$(PY) scripts/02_smoke_meep_bridge.py

phc-bend:         ## literature benchmark, default stages (toy engine, CPU)
	$(PY) scripts/06_phc_bend.py --tag make

pvgc-opt:         ## M1: one inverse-design round on the pvgc coupler (~13 h, run detached)
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/15_pvgc_optimize.py --tag m1 --gradcheck

help:
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/  -/'

viz:              ## render figures from a run dir: make viz RUN=runs/<dir>
	$(PY) -m invdx.viz $(RUN) --pdf
