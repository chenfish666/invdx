> **English** · [繁體中文](RESULTS.zh-TW.md)

# Lesson 2 reference output

Run: `python scripts/09_toy_adjoint.py` (CPU / float64, ~3 minutes end to end)
Full results: the `runs/` directory the command prints on its very first line.

## Actual output

```
[run] outputs -> runs/20260830-110925-toy-adjoint
[base] intact bend   mean T = 0.972
[base] damaged bend  mean T = 0.613   (defect (1, 0), benchmark Layer-I)
[adjoint] one backward pass = gradients for all 400 design-region parameters  (0.5s, incl. compile)
[gradcheck] pixel (np.int64(17), np.int64(12)): adjoint -3.668794e-07  FD -3.668793e-07  rel err 1.61e-07
[gradcheck] pixel (np.int64(10), np.int64(5)): adjoint +5.088147e-08  FD +5.088152e-08  rel err 9.79e-07
[gradcheck] pixel (np.int64(6), np.int64(0)): adjoint -9.336720e-07  FD -9.336726e-07  rel err 6.67e-07
[gradcheck] PASS (worst 9.79e-07 < 1e-5)
[opt] iter   0  mean T = 0.613
[opt] iter   5  mean T = 0.615
[opt] iter  10  mean T = 0.616
[opt] iter  15  mean T = 0.618
[opt] iter  20  mean T = 0.624
[opt] iter  25  mean T = 0.645
[opt] iter  30  mean T = 0.716
[opt] iter  35  mean T = 0.840
[opt] iter  40  mean T = 0.874
[opt] iter  45  mean T = 0.936
[opt] iter  50  mean T = 0.997
[opt] iter  55  mean T = 0.983
[opt] iter  59  mean T = 0.997

[result] damaged 0.613 -> healed 0.999 (intact 0.972)
[design] healed-region density (2x2 lattice, one entry = one a x a cell):
         0.00 0.03
         0.07 0.15
[done] runs/20260830-110925-toy-adjoint/results.json
```

Two things to know before you diff your own run against this one. The run
directory name (a timestamp) and the `[adjoint]` timing change every time;
everything else above is deterministic on CPU + float64 and reproduced
bit-for-bit across two independent runs. And the `np.int64(...)` wrappers in
the `[gradcheck]` lines are numpy's scalar repr, not part of the pixel index —
those are pixels (17, 12), (10, 5) and (6, 0).

If you run `--gradcheck-only` instead, the output stops after the
`[gradcheck] PASS` line; that path writes `results.json` and returns without
printing a `[done]` line.

## Three things worth remembering

1. **The power of the adjoint method, quantified.** 400 parameters in the
   design region, and one backward pass hands you all 400 gradients in 0.5
   seconds. Getting the same information by finite differences takes 800
   forward simulations — two per parameter, for a central difference. That
   800:1 ratio scales linearly with the number of parameters: a 3D design with
   thousands of parameters buys you a ratio in the thousands. This is your
   first-hand answer to "why does inverse design need the adjoint method?"

2. **The gradient check comes before everything else.** Three random pixels,
   adjoint vs central difference, relative errors in the 1e-7 to 1e-6 range.
   The G2 gate does exactly this to fdtdx — same method, looser tolerance
   (`REL_TOL = 0.05` in
   [src/invdx/gates/g2_gradcheck.py](../../src/invdx/gates/g2_gradcheck.py)),
   because those runs are float32 on GPU while this one is float64 on CPU.
   Your toy engine is now qualified to serve as a third-party witness for a
   gradient.

3. **Healing is not restoring.** The optimum did not grow the defect rod back.
   It *removed* more material near the corner instead — density 0.00–0.15,
   four cells that are nearly empty — and landed at 0.999, **above** the
   0.972 of the intact lattice. "Taking rods out can improve a PhC bend" is a
   classic result in the literature (Mekis et al., Phys. Rev. Lett. 77, 3787,
   1996), rediscovered here by a gradient with no idea what the answer was
   supposed to look like. That freedom from what a structure "ought to" look
   like is exactly what inverse design is worth.

## Honest notes

- This is single-objective optimization on mean T (the average over three
  frequencies), with no fabrication constraints — no minimum-feature rule, no
  binarization. Those belong to the filtering and projection tools in
  [src/invdx/fab/](../../src/invdx/fab/).
- Grey density values between 0 and 1 do not exist in a real process. The
  β-projection (tanh projection) schedule in
  [src/invdx/fab/filters_jax.py](../../src/invdx/fab/filters_jax.py) is what
  drives the grey out. The toy lessons deliberately leave it switched off, so
  that the adjoint method alone stands in the spotlight.
- The toy engine's first-order Mur boundary leaves a residual reflection, so a
  small difference like 0.999 vs 0.972 sits inside the engine's own systematic
  error and is not a number you can trust as physics. The honest reading is
  "the healed bend is at least as good as the intact one", not a precise
  2.7-point improvement.
