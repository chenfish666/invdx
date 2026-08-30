> **English** · [繁體中文](README.zh-TW.md)

# Lesson 1: Port Your FDTD to JAX

> **How to use this lesson**: the library already ships a written-and-verified
> JAX engine (`src/invdx/toy/fdtd2d_jax.py`, 9e-16 away from the numpy version;
> reference output in [RESULTS.md](RESULTS.md) in this folder). That one is the
> answer key. What you get here is the **teaching version**: the blanked-out
> skeleton [fdtd2d_jax_skeleton.py](fdtd2d_jax_skeleton.py) is yours to fill in
> by hand. Point the checker at it with `--file` and the library copy is never
> touched. If you want the real practice, **do not open** the answer in `src`
> first.

Goal: port `toy/fdtd2d.py` (numpy, ~160 lines; the engine you already used to
reproduce that classic PhC 90-degree bend benchmark) to JAX by hand, and prove
it is **bit-for-bit the same physics**. What comes out of this lesson is not
just a faster engine — it is your ticket to adjoint gradients: once the JAX
version exists, `jax.grad` can reach through the whole time evolution and hand
you "transmission with respect to any design parameter" (lesson 2).

There are only **three blanks** to fill in, all in this folder's
[fdtd2d_jax_skeleton.py](fdtd2d_jax_skeleton.py),
each one mirroring the identically named section of the numpy version. The
scaffolding (the scan loop, the output packing) is already in place.

```bash
cd <invdx repo>
PY=python   # the python from your invdx env
```

---

## Step 0: Concepts (10 minutes, read before you type)

### The one idea JAX and numpy disagree on: arrays are immutable

numpy lets you write in place: `Hx -= ...`, `Ez[1:-1] += ...`.
JAX arrays are **immutable** — every "modification" actually builds a new array:

| numpy (in place) | JAX (functional) |
|---|---|
| `Hx -= a` | `Hx = Hx - a` |
| `Ez[1:-1,1:-1] += a` | `Ez = Ez.at[1:-1,1:-1].add(a)` |
| `Ez[0,:] = b` | `Ez = Ez.at[0,:].set(b)` |

Why does this matter? Because every piece of JAX magic (jit compilation,
autodiff, vmap) rests on functions having no side effects: input in, output
out, nothing quietly changed in between. That is what lets the compiler
reorder, fuse and differentiate your code without fear.
(Do not worry about the cost: after jit, `.at[].add` is optimized back into an
in-place update.)

### lax.scan: a loop that carries state

Run this three-line demo first, and do not move on until it clicks:

```bash
$PY scripts/08_toy_jax_lesson1.py --scan-demo
```

`scan(step, init, xs)`: `step(carry, x) -> (carry, y)` is fed each element of
`xs` in order, the state `carry` is threaded all the way through, and each
step's `y` is stacked into an array for you. For FDTD:

- `carry` = the field state `(Ez, Hx, Hy)`
- `xs` = the source amplitude per step (the whole waveform, precomputed)
- `y` = the probe reading at that step

scan compiles the entire time loop into **one** XLA program — that is the
precondition for `jax.grad` differentiating the whole evolution later, and the
reason it beats a Python for loop.

---

## Step 1: Blank A — the H update (Faraday's law)

Open `fdtd2d_jax_skeleton.py`, find blank A, look at the two "H from curl E"
lines in the numpy version [fdtd2d.py](../../src/invdx/toy/fdtd2d.py), and
rewrite them in immutable style. Delete the `raise NotImplementedError` line.

Ask yourself: does the H update need `.at[]`? Why does E need it?
(Hint: H is recomputed as a whole array, E only touches an interior slice.)

## Step 2: Blank B — the E interior update (Ampere's law)

Mirror the "E interior from curl H" block. Two things not to miss:
1. `Ez_old = Ez` — the snapshot is already in the scaffolding, and it sits
   **before** your update. Think about why the order matters (Mur wants the
   boundary values from the previous step).
2. `/ eps[1:-1, 1:-1]` — the one and only place the material enters. Lesson 2
   takes the gradient with respect to exactly this `eps`, so treat it with
   respect.

## Step 3: Blank C — Mur absorbing boundary (four edges)

Four lines, one pattern, a direct translation of the numpy version:

```
Ez = Ez.at[0, :].set( Ez_old[1, :] + mur * (Ez[1, :] - Ez_old[0, :]) )
```

(That first edge is a gift. The other three are yours: `[-1,:]`, `[:,0]`,
`[:,-1]`.)

## Step 4: Checkpoint

```bash
$PY scripts/08_toy_jax_lesson1.py --file tutorials/01-jax-port/fdtd2d_jax_skeleton.py
```

Passing looks like this (the seconds differ every run; the magnitudes are what
must be right):

```
[case] photonic-crystal bulk gap measurement, 110^2 grid, 2000 steps (small, runs in seconds)
[diff] max|dE| = 8.882e-16, max|dH| = 9.159e-16 (field scale 4.140e-01)
[time] numpy 0.39s | jax first run 1.01s (with compile) | jax rerun 0.32s

[PASS] both engines agree to the last bit -- your first JAX FDTD works.
       Next lesson (scripts/09): make eps a parameter, push jax.grad
       through the whole time evolution, then check it against finite differences.
```

(With `--file` one extra line comes first:
`[mode] checking tutorial skeleton: tutorials/01-jax-port/fdtd2d_jax_skeleton.py`.
While a blank is still empty you get `[todo]` instead, naming the blank you are
stuck on.)

1e-15 is float64 machine precision: your JAX engine and the numpy engine are
**the same physics**, not "close enough". This kind of equivalence proof is how
the whole invdx project earns trust, and you just produced one by hand.

Then run:

```bash
$PY scripts/08_toy_jax_lesson1.py --gpu
```

Same code, zero edits, running on a GPU — the port's second payoff.

---

## Gotchas (look here first when you are stuck)

- **The float32 trap**: JAX defaults to float32, numpy is float64. The checker
  flips `jax_enable_x64` at the very top of its imports; if you write your own
  test script, that line must come before any JAX array is born. Stuck at a
  difference around 1e-7 that will not go lower? Nine times out of ten this is
  it.
- **Python side effects inside `step`** (print, appending to an outer list):
  scan runs your Python exactly once, while tracing; after that it runs the
  compiled artifact — the side effect does not happen per step. Anything you
  want recorded goes out through `y`.
- **The first run is slow**: that is compilation (tracing + XLA optimization);
  the second run is the real speed. The checker prints both times — go look at
  the difference yourself.
- **Shape errors**: the slices in the curl lines must come out exactly
  (nx-2, ny-2). JAX's error message tells you the shapes; line them up against
  the numpy slices one index at a time.

## When you are done

Compare answers: put your fill-ins next to the library version
[src/invdx/toy/fdtd2d_jax.py](../../src/invdx/toy/fdtd2d_jax.py) — the way you
wrote it may differ, but passing at 1e-15 still means it is the same physics.
Then go on to lesson 2 (`tutorials/02-first-adjoint`): **your first adjoint
gradient**.
