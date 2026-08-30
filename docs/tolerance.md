> **English** · [繁體中文](tolerance.zh-TW.md)

[← back to docs index](README.md)

# Design-for-tolerance: method notes

Why this file: an inverse-designed device is optimized against one nominal
geometry, while a fab process delivers a distribution of geometries. The
example problem shipped here is a grating coupler and assumes a 193 nm DUV
SOI platform, and published simulation studies of that class of process show
inverse-designed devices carrying *hard* minimum-feature constraints alone
can still land at Yield_90% = 0% before lithography-aware correction
(OptoSynthesizer, arXiv:2604.15493, Table 1 — simulation, no fab data). A minimum-feature rule
is therefore worth reporting against, not worth trusting. This note maps that
literature onto the mechanisms this repo already has, and fixes the reporting
conventions so a tolerance number means the same thing every time it is
quoted.

## How to run it

```bash
make tolerance RUN=runs/<coupler-opt-dir>                 # sensitivity map + corner evaluation
make tolerance RUN=runs/<dir> LAMS=1.27,1.35,9         # plus corner CE spectra
make runs                                              # which run dirs qualify
```

Writes into `<run-dir>/tolerance/`; the source run directory is read-only.
Implemented in `scripts/16_tolerance_report.py`, whose `--help` documents every
flag. An earlier version of this file described the tools below as planned
work while they were already written -- a reader's fair conclusion was that
the feature did not exist yet, which was the opposite of true.

## What the repo already provides

- `fab/transforms.py` — conic filter (radius = `cfg.min_feature`) + tanh
  projection. The projection threshold `eta` is the erosion/dilation knob:
  `eta_i = 0.5` (nominal), `eta_e` (eroded / over-etch), `eta_d`
  (dilated / under-etch) are already in `config.py`. Those three fields are
  the standard erosion/dilation formulation, and are what the corner
  evaluation below re-rasterizes at.
- Adjoint gradients through the full scene (G2-gated) — which makes
  sensitivity analysis essentially free.

## What the report computes

Both steps are implemented in `scripts/16_tolerance_report.py`:

1. *Sensitivity map* — evaluate `∂CE/∂rho` at the final design (CE is the
   coupling efficiency, rho the design density field) and reduce it to a
   per-tooth linewidth sensitivity — a tooth being one periodic line of the
   grating. Cost: one backward pass. Output: a figure + CSV ranking which
   teeth dominate CE degradation under linewidth drift. Rationale:
   mask-to-wafer error is not uniform; sensitive regions dominate performance
   (arXiv:2604.15493 §3.1.2).
2. *Corner evaluation* — re-rasterize the fixed design at the three
   projection corners (eta_i / eta_e / eta_d) and report CE and 3 dB
   bandwidth per corner. No re-optimization; this quantifies how fragile the
   nominal design is.

This is an evaluation of a fixed design, not a robust optimization: moving
the corners *into* the objective (worst-case or softmin over the projection
ensemble) is a different formulation and is not implemented here.

## Reporting conventions

- Yield metric: `Yield_90% = Pr(CE_corner >= 0.9 * CE_nominal)` over the
  sampled variation set — the threshold is fixed here in the method notes so
  it cannot be chosen after seeing a result. The 0.9 factor is on linear CE,
  not on dB: the script converts `CE_dB` back to linear before comparing.
- Corner table columns: `corner, CE_dB, bw_3db_nm, ridge_lam_um`.
- All results from this repo are simulation. Write "simulation shows", never
  "experiments show". (Applies equally when citing arXiv:2604.15493 — that
  paper's fab results are digital-twin simulation, not measured wafers.)

## References

- OptoSynthesizer: end-to-end physical design automation for yield-optimized
  inverse-designed EPICs. arXiv:2604.15493 (2026). Motivation numbers and the
  sensitivity-guided correction argument.
- BOSON⁻¹: variation-aware photonic inverse design, DATE 2025. The reference
  for an in-the-loop robust formulation, which this repo does not implement.
- PRISM: photonics-informed inverse lithography, arXiv:2602.15762. Mask-level
  correction; out of scope here (no mask flow), cited for the non-uniform
  sensitivity argument.
