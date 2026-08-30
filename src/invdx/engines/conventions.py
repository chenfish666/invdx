"""Cross-engine convention contracts — hard-won engine lessons as code.

Every rule here cost real debugging time to find, and every one of them is a
silent-wrong-answer trap rather than a crash. Gates and engine adapters import
from this module instead of re-deriving conventions ad hoc; that is the whole
point.

Gate-ordering rationale (why unit -> API -> gradcheck -> physics ->
reciprocity -> cross-engine): each gate assumes everything before it is
trustworthy, so failures localize. gradcheck before physics because a wrong
gradient silently corrupts every optimization even when forward physics looks
fine; reciprocity last among single-engine gates because it is the only check
that catches *normalization* errors end-to-end (it caught the 2x CE bug);
cross-engine last because it presumes both engines individually validated.
"""

import numpy as np

# Lesson 1 — Meep power convention.
# Meep DFT fields, |alpha|^2 eigenmode coefficients and fluxes omit the
# physical 1/2 (P = 1/2 Re E x H*). invdx.modes uses the physical convention.
# Missing the bridge shows up as a factor-2 error in coupling efficiency; the
# fiber-side reciprocity gate is what exposes it.
MEEP_POWER_OMITS_HALF = True


def meep_to_physical_power(overlap_power_value):
    """Bridge an invdx.modes.overlap_power result into Meep's power convention
    so it can be divided by a Meep-side input power: multiply by 2."""
    return 2.0 * overlap_power_value


# Lesson 2 — meep.adjoint multi-frequency gradient shape.
def collapse_multifreq_gradient(dJ):
    """meep.adjoint returns dJ with shape (Nx, nf) for multi-frequency
    objectives; the design gradient is the frequency sum, NOT a ravel.
    (A ravel here silently produces a wrong-length gradient.)"""
    dJ = np.asarray(dJ)
    if dJ.ndim == 2:
        return dJ.sum(axis=1)
    return dJ


# Lesson 3 — resolution vs design-grid density.
def assert_resolution_covers_design_grid(cfg):
    """Adjoint gradients are systematically small when the simulation
    resolution is below the design grid density (measured on the grating_coupler problem:
    5-8% error at res 40, exact at res 80 with design_grid_per_um=100).
    Fail loudly, don't warn."""
    if cfg.resolution < cfg.design_grid_per_um:
        raise ValueError(
            f"resolution={cfg.resolution} < design_grid_per_um="
            f"{cfg.design_grid_per_um}: adjoint gradients will be "
            f"systematically underestimated (5-8% measured at res 40). "
            f"Raise resolution or lower the design grid density.")


# Lesson 4 — Meep DFT decay tolerance.
# mpa.OptimizationProblem / stop_when_dft_decayed default to decay_by=1e-11,
# ~3.4x slower than 1e-6 with FOM unchanged to 4 decimals (measured).
# meep_worker must always pass cfg.dft_decay_tol explicitly.
MEEP_DECAY_BY_DEFAULT_TOO_SLOW = 1e-11


# Lesson 5 — FOM spectral sampling vs validation density.
# A minimax FOM over sparse wavelength samples lets the optimizer pump the
# sampled points while the spectrum collapses between them (seen with three
# samples spread over a 100 nm band: the sampled points improved while a deep
# valley opened between them and the peak migrated to the band edge, so the
# optimizer's own readout was meaningless). Guards: (a) FOM sample spacing must
# be finer than the device's expected spectral feature width (grating:
# ~bandwidth/4); (b) never claim a bandwidth wider than the physics supports;
# (c) always validate on a dense spectrum before trusting an optimized design —
# that validation belongs in the problem's post-optimization gate.
def assert_fom_sampling_covers_band(sample_spacing_nm, expected_feature_nm):
    if sample_spacing_nm > expected_feature_nm / 2:
        raise ValueError(
            f"FOM wavelength sampling ({sample_spacing_nm} nm) is coarser than "
            f"half the expected spectral feature width ({expected_feature_nm} nm): "
            f"the optimizer can exploit the gaps between samples. "
            f"Densify samples or narrow the band.")


# Lesson 6 — cross-engine comparisons of resonant structures must compare
# spectra, not single-wavelength values.
# Engines discretize the same nominal geometry into different EFFECTIVE
# geometry: crisp cell-fill (fdtdx UniformMaterialObject) vs subpixel-averaged
# edges (Meep Block) differ by up to half a cell per edge. A grating's coupling
# ridge is very sensitive to tooth width, so half a cell of effective edge
# difference (6.25 nm at res 80) is enough to move the ridge by more than its
# own width: readings taken at one fixed wavelength can then disagree by tens
# of dB even though both engines put the ridge PEAK at the same value.
# Corollaries: (a) validate cross-engine on ridge position+peak, with the
# effective-geometry uncertainty stated; (b) the same sensitivity means a
# design's tolerance to geometry error has to be evaluated explicitly rather
# than assumed.
CROSS_ENGINE_COMPARE_SPECTRA = True
