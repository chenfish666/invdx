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
    dtype: str = "float32"         # fdtdx array dtype
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
        # For a conic filter with three-field projection at eta_e = 0.75, the
        # guaranteed solid/void length scale equals the filter radius:
        #   R = b / (2 - 2*sqrt(1 - eta_e)) = b  when eta_e = 0.75.
        # (Relation from length-scale analysis used by Meep's
        #  get_conic_radius_from_eta_e.)  We therefore set R = min_feature.
        return self.min_feature
