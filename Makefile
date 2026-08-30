# Convenience targets. uv owns fdtdx/JAX (see README Environment); the meep
# env is only ever reached through the subprocess bridge (engines/meep_bridge.py).
# Every target here is a command, not a file. Without this, `make runs`
# silently does nothing because runs/ is a real directory and make decides it
# is already up to date -- the same trap waits for any target whose name
# collides with a path.
.PHONY: check gates test smoke smoke-meep phc-bend coupler-opt-smoke coupler-opt verify tolerance handoff runs help viz

PY  ?= uv run python
GPU ?= $(if $(INVDX_GPU),$(INVDX_GPU),0)  # CUDA_VISIBLE_DEVICES for GPU-using targets

# Targets taking RUN= fail inside Python with a path traceback when it is
# missing, which tells the reader nothing about what to type. Say it here.
define need_run
	@test -n "$(RUN)" || { \
	  echo "$@ needs a run directory:  make $@ RUN=runs/<dir>" >&2; \
	  echo "try 'make runs' to see which directories qualify" >&2; \
	  exit 2; }
endef

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

coupler-opt-smoke:   ## ~10 min: 4 iterations, proves the whole path runs
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/15_grating_coupler_optimize.py \
	    --tag smoke --iters 4 --set sim_time_s=0.3e-12

coupler-opt:         ## one inverse-design round on the grating coupler (~13 h, run detached)
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/15_grating_coupler_optimize.py --tag opt --gradcheck

verify:           ## second-engine check of a finished design: make verify RUN=runs/<dir>
	$(call need_run)
	CUDA_VISIBLE_DEVICES=$(GPU) $(PY) scripts/07_grating_coupler_verify_design.py --run $(RUN)

tolerance:        ## fabrication tolerance report: make tolerance RUN=runs/<dir> [LAMS=1.27,1.35,9]
	$(call need_run)
	$(PY) scripts/16_tolerance_report.py $(RUN) $(if $(LAMS),--lams $(LAMS))

handoff:          ## tool-neutral export for another solver: make handoff RUN=runs/<dir>
	$(call need_run)
	$(PY) -m invdx.export.handoff $(RUN)

runs:             ## which run directories can be fed to verify/tolerance/handoff
	@$(PY) tools/list_runs.py

help:
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/  -/'

viz:              ## render figures from a run dir: make viz RUN=runs/<dir>
	$(call need_run)
	$(PY) -m invdx.viz $(RUN) --pdf
