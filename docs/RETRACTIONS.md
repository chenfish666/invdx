# Retractions (append-only)

Conclusions this project published (in commits, docs, or reports) and later
refuted by evidence. Entries are appended; the original text stays where it
was, corrected in place only with a pointer here.

---

## 2026-08-17 — Gradcheck failure mechanism misattributed to float32 cancellation

**Original claim** (commit `c243ac7`, gradcheck docstrings and report): finite-
difference vs adjoint disagreement on pvgc Device voxels is caused by
"subtracting two nearly-equal float32 numbers" — cancellation noise — and the
signal-floor sampler (only test voxels ≥5% of peak gradient) addresses it.

**What refuted it**: the production-scale (0.8 ps, θ=10) gradcheck failure of
slurm job 153 (voxel 213, rel err 6.29%) was reproduced bit-exactly and probed
with an h-scan. The residual scaled as h³ (×0.126 per halving; cancellation
noise would be flat), sat 200× above the measured float32 noise floor, and
Richardson extrapolation from the same two h values agreed with the adjoint to
0.0106%. Failing voxels were not low-signal (voxel 213 carries 15% of peak,
3× the floor), and θ=10's gradient signal is 63% *stronger* than θ=0's.
Probes: /tmp/probe2_gpu.py, /tmp/probe3_analyze.py (2026-08-17 session).

**Corrected statement**: cancellation noise is real only for voxels *below*
the signal floor (the sampler remains valid for its original purpose). For
eligible voxels at production scale the dominant FD error is *truncation* —
long runs make the FOM more oscillatory in the design parameters, inflating
f‴ and thus the h³ term at the fixed h=0.05. The adjoint was never wrong.

**Consequence**: gradcheck now uses two-h Richardson extrapolation with an FD
self-consistency indicator; tolerances and the signal floor are unchanged.
