"""Gate 4 — reciprocity: forward vs reciprocal excitation must give the same
coupling efficiency.

This is the gate that caught a factor-2 normalization bug in this codebase's
own mode-overlap path — it is the only single-engine check that validates
*normalization* end-to-end, which is why it outranks the cross-engine gate
for trust in absolute numbers. A convention error that scales both directions
equally is invisible to every other check; one that scales only one of them
shows up here immediately.

Contract (to implement once both excitation directions exist):
    1. forward: problem's nominal source -> CE into the target mode
    2. reciprocal: excite from the target mode -> CE back into the source mode
    3. |CE_fwd - CE_rev| within a tight bound (0.2 dB) at the nominal design;
       grayscale high-Q intermediates seen mid-optimization ring longer and
       are noisier, so they need a looser bound (~1.6 dB) or a longer run

Implementation (grating_coupler problem, cheap settings — 25nm grid, uniform grating,
theta=10 where CE is strong):
    forward:  wg-side beam -> forward-TE0-normalized CE into the tilted
              upward Gaussian (wg_side_characterize; injection impurity is
              filtered by the forward mode overlap)
    reverse:  fiber-side tilted beam -> CE into the -x TE0 (characterize)
Cheap-mode tolerance is 0.5 dB, slack enough to absorb the coarse grid and the
short run; a final design at production resolution should be held to something
much tighter (0.2 dB) — re-tighten per problem when it matters.
"""

from .runner import GateResult

NAME = "reciprocity"
ORDER = 4
REQUIRES = ("gpu",)

TOL_DB = 0.5


def run(cfg, args):
    from invdx.problems import grating_coupler

    pcfg = grating_coupler.GratingCouplerConfig(spacing_um=0.025, sim_time_s=0.8e-12,
                           theta_deg=10.0)
    teeth = grating_coupler.uniform_grating_teeth(pcfg, period=0.575, duty=0.5)
    fwd = grating_coupler.wg_side_characterize(pcfg, teeth)
    rev = grating_coupler.characterize(pcfg, teeth)
    mismatch = abs(fwd["CE_fwd_dB"] - rev["CE_dB"])
    details = {"CE_fwd_dB": fwd["CE_fwd_dB"], "CE_rev_dB": rev["CE_dB"],
               "mismatch_dB": mismatch, "S11_dB": fwd["S11_dB"]}
    if mismatch > TOL_DB:
        return GateResult(NAME, "fail", {
            "reason": f"reciprocity violated: |CE_fwd - CE_rev| = "
                      f"{mismatch:.3f} dB > {TOL_DB} dB — suspect a "
                      f"normalization/convention bug (a missing factor of 2 "
                      f"on one side is exactly what this looks like)",
            **details})
    return GateResult(NAME, "ok", details)
