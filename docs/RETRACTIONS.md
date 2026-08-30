[← back to docs index](README.md)

# Retractions (append-only)

Conclusions this project published (in code, docs, or reports) and later
refuted by evidence. Entries are appended; the original text stays where it
was, corrected in place only with a pointer here.

---

## 2026-08-17 — Gradcheck failure mechanism misattributed to float32 cancellation

**Original claim** (as the gradcheck docstrings first stated it): finite-
difference vs adjoint disagreement on grating_coupler Device voxels is caused by
"subtracting two nearly-equal float32 numbers" — cancellation noise — and the
signal-floor sampler (only test voxels ≥5% of peak gradient) addresses it.

**What refuted it**: the production-scale (0.8 ps, θ=10) gradcheck failure
that aborted the first θ=10 round (voxel 213, rel err 6.29%) was reproduced
bit-exactly and probed with an h-scan. The residual scaled as h³ (×0.126 per
halving; cancellation noise would be flat), sat 200× above the measured
float32 noise floor, and Richardson extrapolation from the same two h values
agreed with the adjoint to 0.0106%. Failing voxels were not low-signal (voxel
213 carries 15% of peak, 3× the floor), and θ=10's gradient signal is 63%
*stronger* than θ=0's. The h-scan is reproducible from the aborted run's own
`cmd.txt`/`config.json`; the same two-h comparison now ships permanently as
`fd_consistency` in `gradcheck()`.

**Corrected statement**: cancellation noise is real only for voxels *below*
the signal floor (the sampler remains valid for its original purpose). For
eligible voxels at production scale the dominant FD error is *truncation* —
long runs make the FOM more oscillatory in the design parameters, inflating
f‴ and thus the h³ term at the fixed h=0.05. The adjoint was never wrong.

**Consequence**: gradcheck now uses two-h Richardson extrapolation with an FD
self-consistency indicator; tolerances and the signal floor are unchanged.

---

## 2026-08-22 — Checkpoint memory slope `291 B/cell/checkpoint` was a GiB/GB
unit-label bug, not sensor contamination

**Original claim**: the 3D adjoint checkpoint-memory model was
`peak(C) ≈ 370 + 291·C` bytes/cell, fitted from three anchor measurements
(C=10/20/28 → 5.93/11.25/15.42 GB at 1.944M cells) and reported as confirmed
because a second derivation reproduced 370.1/291.3 — from the same constants.

**What refuted it**: a clean re-derivation across independently sourced
anchors. Runtime `peak_bytes_in_use` on a Turing-generation GPU (two
uncontaminated points: C=10 in an isolated process, plus C=20/C=28 each a
genuine new high-water mark when measured), the same measurement on an
Ada-generation GPU, and XLA's compile-time
`compiled(...).memory_analysis()` (no kernel execution at all) all agree on
**268.0329 B/cell/checkpoint** to 1.1e-7 relative. A fourth check — a
leaf-channel byte count of the arrays fdtdx actually checkpoints, 37
channels for the full carry copy plus 30 channels for the second copy the
backward VJP needs (`src/invdx/engines/fdtdx_checkpoint_buffers.py`) —
predicts the same slope to 29 bytes in 521 million (5.6e-8).

The three anchor values were themselves **not** contaminated by the
process-level `peak_bytes` high-water-mark effect this project has hit
before; they were independently reproduced to <1.5% error. The error came
one step later: those points are decimal-GB (1e9-byte) readings, but the
fitted slope/intercept (`0.5274`, `0.670`) were labelled "GiB" and converted
to bytes/cell with ×2^30 instead of ×1e9 — a 7.374% inflation that accounts
for most of the 291-vs-268 gap, with a further ~1.2% from fitting three
two-decimal-rounded points instead of exact byte counts. Converting the same
fit correctly gives ≈271 B/cell/ckpt, already close to the true 268.

**Corrected statement**: the checkpoint-memory slope is **268.0329
B/cell/checkpoint**, invariant across Turing and Ada to 5e-8 relative, and
matching a from-first-principles count of what fdtdx's checkpointed
while-loop actually stores. The C=0 intercept on Turing, from the same clean
two-point data, is **372.18 B/cell** — not 370.1, which the same GiB/GB bug
produced on the same platform and is therefore not evidence of a
cross-architecture difference. Whether Ada's intercept differs by a real
architectural margin is **not established**: no independently-sourced raw
two-point byte data for Ada was available to compute it the same way.

**Consequence**: any extrapolation built on `370 + 291·C` — including the
`reversible ≈ 144·N²·T` vs `checkpointed ≈ 291·N³·C` crossover at
`N* ≈ 519` cells/edge — must be recomputed with 268.

Separately, the framing of "independent paths agreeing to 1.1e-7"
overclaimed. The two runtime paths and the compile-time path all read out
the same underlying XLA `BufferAssignment`: runtime `peak_bytes_in_use` is
the allocator's high-water mark, and the allocator sizes those buffers from
exactly the assignment `memory_analysis()` reports. Their agreement is close
to a tautology for the planned portion — it shows there is no *unplanned*
extra allocation (no donation surprise, no fragmentation beyond the plan, no
host staging), which is a real but narrow result. The cross-architecture
comparison likewise shows the same planner making the same decisions on two
machines, not two independent measurements. The leaf-channel count is the
only path sharing no premises with the XLA planner, and is the actual
independent check.
