"""Gate 4 — reciprocity: forward vs reciprocal excitation must give the same
coupling efficiency.

This is the gate that caught pvgc's factor-2 normalization bug (median
mismatch 2.93 dB -> 0.150 dB after the fix) — it is the only single-engine
check that validates *normalization* end-to-end, which is why it outranks
the cross-engine gate for trust in absolute numbers.

Contract (to implement once both excitation directions exist):
    1. forward: problem's nominal source -> CE into the target mode
    2. reciprocal: excite from the target mode -> CE back into the source mode
    3. |CE_fwd - CE_rev| within 0.2 dB at the nominal design (pvgc's strict
       criterion; grayscale high-Q intermediates may need a looser 1.6 dB,
       see pvgc LOG 2026-07-12)

Implementation (PVGC problem, cheap settings — 25nm grid, uniform grating,
theta=10 where CE is strong):
    forward:  wg-side beam -> forward-TE0-normalized CE into the tilted
              upward Gaussian (wg_side_characterize; injection impurity is
              filtered by the forward mode overlap)
    reverse:  fiber-side tilted beam -> CE into the -x TE0 (characterize)
Cheap-mode tolerance 0.5 dB (measured 0.07 dB); pvgc's strict criterion for
final designs at res-80 is 0.2 dB — re-tighten per problem when it matters.
"""

from .runner import GateResult

NAME = "reciprocity"
ORDER = 4
REQUIRES = ("gpu",)

TOL_DB = 0.5


def run(cfg, args):
    from invdx.problems import pvgc

    pcfg = pvgc.PVGCConfig(spacing_um=0.025, sim_time_s=0.8e-12,
                           theta_deg=10.0)
    teeth = pvgc.uniform_grating_teeth(pcfg, period=0.575, duty=0.5)
    fwd = pvgc.wg_side_characterize(pcfg, teeth)
    rev = pvgc.characterize(pcfg, teeth)
    mismatch = abs(fwd["CE_fwd_dB"] - rev["CE_dB"])
    details = {"CE_fwd_dB": fwd["CE_fwd_dB"], "CE_rev_dB": rev["CE_dB"],
               "mismatch_dB": mismatch, "S11_dB": fwd["S11_dB"]}
    if mismatch > TOL_DB:
        return GateResult(NAME, "fail", {
            "reason": f"reciprocity violated: |CE_fwd - CE_rev| = "
                      f"{mismatch:.3f} dB > {TOL_DB} dB — suspect a "
                      f"normalization/convention bug (this gate caught "
                      f"pvgc's 2x factor)", **details})
    return GateResult(NAME, "ok", details)
