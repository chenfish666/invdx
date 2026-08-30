> **English** · [繁體中文](RESULTS.zh-TW.md)

# Lesson 1 Reference Output

The reference answers for blanks A/B/C are in
[src/invdx/toy/fdtd2d_jax.py](../../src/invdx/toy/fdtd2d_jax.py); this file
records what one real checkpoint run printed, so you have something to hold your
own run against.

## Checkpoint output (scripts/08_toy_jax_lesson1.py --gpu)

```
[case] photonic-crystal bulk gap measurement, 110^2 grid, 2000 steps (small, runs in seconds)
[diff] max|dE| = 8.882e-16, max|dH| = 8.882e-16 (field scale 4.140e-01)
[time] numpy 0.38s | jax first run 1.99s (with compile) | jax rerun 0.34s

[PASS] both engines agree to the last bit -- your first JAX FDTD works.
       Next lesson (scripts/09): make eps a parameter, push jax.grad
       through the whole time evolution, then check it against finite differences.
[gpu] jax default device: gpu (<your GPU model>)
[gpu] same code, zero edits, on gpu: first run 0.33s, rerun 0.37s
[gpu] that is the port's second payoff: the numpy version is CPU-only, forever.
```

`<your GPU model>` is a placeholder: jax prints the `device_kind` of whichever
card it found, which is the card's model string. The machine this transcript
came from printed `Quadro RTX 6000`. The word in front of it is
`device.platform`, always lowercase `gpu` — never `GPU`, never `cuda`.

## How to read the numbers

- **8.9e-16**: float64 machine precision (~2.2e-16) times a few steps of
  accumulation. This does not say "the two engines are close", it says "the two
  engines are the same physics" — every floating-point operation is identical,
  only the way it executes differs (interpreted numpy vs compiled XLA).
- **jax first run 1.99s vs rerun 0.34s**: the ~1.6s difference is tracing +
  compilation, and you pay it once. Later calls at the same shapes go straight
  to the compiled artifact.
- **GPU 0.37s is no faster than CPU 0.34s**: this problem is too small (110^2)
  to fill a GPU; moving the data eats the compute advantage. The GPU payoff
  shows up on big grids (a full-size 3D problem runs to tens of millions of
  cells) or at float32 — "do not put a small problem on a GPU" is a lesson in
  itself. Absolute seconds shift from machine to machine; the ratios are what
  is worth reading.
- **The test now guards this**: tests/test_toy_jax.py checks the two engines for
  equivalence on every `make gates` run from here on, so the JAX version can
  never quietly drift away from the numpy one.

## What this lesson sets you up for

That `/ eps[1:-1, 1:-1]` in the E update now lives inside a differentiable XLA
program. Lesson 2 (`tutorials/02-first-adjoint`) goes straight for it:
`jax.grad(transmission)(eps)` gets you "how each cell's material affects the
output" in one shot — the adjoint method, automatically.
