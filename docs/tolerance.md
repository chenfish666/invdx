[← back to docs index](README.md)

# Design-for-tolerance: method notes

Why this file: the pvgc target platform (iSiPP50G) is a 193 nm DUV line, and
simulation studies on that node show inverse-designed devices with *hard*
minimum-feature constraints alone can land at Yield_90% = 0% before
lithography-aware correction (OptoSynthesizer, arXiv:2604.15493, Table 1 —
simulation, no fab data). Tolerance is therefore a first-class objective here,
not a post-hoc check. This note maps that literature onto mechanisms this
repo already has, and fixes the reporting conventions.

## What the repo already provides

- `fab/transforms.py` — conic filter (radius = `cfg.min_feature`) + tanh
  projection. The projection threshold `eta` is the erosion/dilation knob:
  `eta_i = 0.5` (nominal), `eta_e` (eroded / over-etch), `eta_d`
  (dilated / under-etch) are already in `config.py`. This is the standard
  three-field robust formulation waiting to be driven.
- Adjoint gradients through the full scene (G2-gated) — which makes
  sensitivity analysis essentially free.

## Planned use (M-numbered)

**M1+ (immediately after the first optimized design exists):**

1. *Sensitivity map* — evaluate `∂CE/∂rho` at the final design and reduce it
   to a per-tooth linewidth sensitivity. Cost: one backward pass. Output: a
   figure + CSV ranking which teeth dominate CE degradation under linewidth
   drift. Rationale: mask-to-wafer error is not uniform; sensitive regions
   dominate performance (arXiv:2604.15493 §3.1.2).
2. *Corner evaluation* — re-rasterize the fixed design at the three
   projection corners (eta_i / eta_e / eta_d) and report CE and 3 dB
   bandwidth per corner. No re-optimization; this quantifies how fragile the
   nominal design is and motivates M2.

**M2 (robust optimization):**

- Move the corners *into* the objective (worst-case or softmin over
  {nominal, eroded, dilated}), i.e. the BOSON⁻¹-style variation-aware loop
  (DATE 2025) rather than heuristic constraints. Report CE and bandwidth as
  mean ± sigma over the corner/variation ensemble plus a Yield-style metric.

## Reporting conventions (fixed now, before results exist)

- Yield metric: `Yield_90% = Pr(CE_corner >= 0.9 * CE_nominal)` over the
  sampled variation set — threshold declared before measurement, per the
  pre-registration discipline in `docs/journal.md`.
- Corner table columns: `corner, CE_dB, bw_3db_nm, ridge_lam_um`.
- All results from this repo are simulation. Write "simulation shows", never
  "experiments show". (Applies equally when citing arXiv:2604.15493 — that
  paper's fab results are digital-twin simulation, not measured wafers.)

## References

- OptoSynthesizer: end-to-end physical design automation for yield-optimized
  inverse-designed EPICs. arXiv:2604.15493 (2026). Motivation numbers and the
  sensitivity-guided correction argument.
- BOSON⁻¹: variation-aware photonic inverse design, DATE 2025. The
  in-the-loop robust formulation M2 follows.
- PRISM: photonics-informed inverse lithography, arXiv:2602.15762. Mask-level
  correction; out of scope here (no mask flow), cited for the non-uniform
  sensitivity argument.
