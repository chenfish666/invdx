"""Configuration — single source of truth for all scripts.

BaseConfig holds only problem-agnostic fields; a concrete design problem
subclasses it and adds its geometry/wavelengths/FOM weights. Scripts never
hardcode numbers (pvgc doctrine): everything tweakable lives here and is
overridable per run via `--set KEY=VAL` (see cli.py).

Units: lengths in um unless a field says otherwise (fdtdx itself uses meters;
engines/fdtdx_engine.py converts at the boundary).
"""

from dataclasses import dataclass


@dataclass
class BaseConfig:
    # ---- Numerics ----
    resolution: int = 80           # Meep-side resolution (pixels/um). pvgc lesson:
                                   # must be >= design_grid_per_um or adjoint
                                   # gradients are systematically small (see
                                   # engines.conventions.assert_resolution_covers_design_grid)
    spacing_um: float = 0.0125     # fdtdx-side uniform grid spacing (um); 1/resolution
                                   # equivalence is not enforced, engines differ
    dft_decay_tol: float = 1e-6    # Meep stop_when_dft_decayed threshold; the worker
                                   # always passes it explicitly (default 1e-11 is
                                   # ~3.4x slower at equal accuracy)
    dtype: str = "float32"         # fdtdx array dtype; only
                                   # engines.fdtdx_engine.make_sim_config reads
                                   # this. problems/pvgc.py build_scene and
                                   # build_scene_3d hardcode float32 and raise
                                   # if this is set to anything else (float64
                                   # there is untested, see
                                   # pvgc._require_float32_dtype)
    # Not an optimisation seed: the design initialisation is deterministic, so
    # two runs differing only in `seed` are bit-identical in their trajectory.
    # It selects which voxels the gradcheck samples (and nothing else), which
    # is why re-running a failed gradcheck reproduces the same voxels.
    seed: int = 0

    # ---- Design parameterization / fabrication ----
    design_grid_per_um: int = 100  # design variable grid density (10 nm pixels)
    min_feature: float = 0.130     # minimum feature size (um) — 193 nm DUV rule
    eta_i: float = 0.5             # nominal projection threshold
    eta_e: float = 0.75            # eroded threshold  (three-field robust opt)
    eta_d: float = 0.25            # dilated threshold (three-field robust opt)
    beta_schedule: tuple = (8, 16, 32, 64, 128)
    softmin_beta: float = 30.0     # LogSumExp sharpness for smooth-min aggregation

    # ---- Bookkeeping ----
    runs_root: str = "runs"

    # ---- Derived helpers ----
    @property
    def filter_radius(self):
        # WARNING (2026-08-21 audit): the identity below is TRUE but its
        # PRECONDITION IS NOT MET by this repo's optimisation loop.
        #   R = b / (2 - 2*sqrt(1 - eta_e)) = b  when eta_e = 0.75
        # is the guaranteed solid/void length scale for a conic filter under
        # the THREE-FIELD ROBUST formulation (optimise eroded eta_e / nominal
        # eta_i / dilated eta_d simultaneously).  This repo optimises the
        # NOMINAL FIELD ONLY: eta_e and eta_d appear nowhere in optimize.py,
        # problems/pvgc.py or scripts/15 -- only in scripts/16, a post-hoc
        # tolerance report.  Consequently R = min_feature guarantees NOTHING
        # about the produced design: measured runs have contained
        # features down to 40 nm against a 130 nm rule.  Keeping R =
        # min_feature is still a reasonable default, but it is a HEURISTIC
        # here, not a guarantee.  docs/tolerance.md states the same limitation
        # from the other direction: its V8 linewidth check is report-only,
        # never pass/fail, because a single nominal field carries no length-
        # scale guarantee.  The fix is the three-field robust formulation
        # (optimise all three thresholds and take the worst case), which costs
        # 3x forward solves per iteration.
        return self.min_feature
