> **English** · [繁體中文](README.zh-TW.md)

# Lesson 2: Your First Adjoint Gradient — Inverse Design Heals a Fabrication Defect

> **How this lesson works**: the whole pipeline is already written as a script
> you can run right now (`scripts/09_toy_adjoint.py`; reference output in
> [RESULTS.md](RESULTS.md)). This page explains **what it is doing and why that
> works** — and if you want to work through it by hand, every stage of the
> script takes parameters you can change and rerun.

## The question this lesson answers

At the end of [Lesson 1](../01-jax-port/README.md) there was a `/ eps` sitting
in the E-field update. This lesson asks the next question:
**how much does the material at each grid cell change the output?**

The brute-force answer: nudge one cell, rerun the simulation — N parameters
cost N+1 simulations. A 20×20 = 400-cell design region means 401 runs. Not
viable, and that is exactly why photonic design in the 2000s meant hand-tuning
a handful of parameters (the shift-or-shrink era).

The adjoint answer: **one forward pass plus one backward pass gives you all N
gradients.** This is the mathematical footing inverse design stands on, and it
is what the adjoint machinery in fdtdx and Meep is doing. JAX automates it: the
`lax.scan` from Lesson 1 is a differentiable program, and `jax.grad`
back-propagating through it *is* the adjoint method — no hand-derived adjoint
field equations required.

## How the script works (scripts/09_toy_adjoint.py)

1. **Damage the structure.** Take the 90° bend from `phc_bend` and simulate a
   fabrication defect: knock out the Layer-I horizontal rod, the one the
   point-defect scan singles out as doing the most damage (the scan itself is
   in [docs/phc-bend-walkthrough.md](../../docs/phc-bend-walkthrough.md)). In
   this lesson's own setup, that one missing rod drops mean transmission from
   0.972 to 0.613.
2. **Draw a design region.** A 2a×2a box around the defect (20×20 grid cells),
   with the material inside continuously parameterized:
   `eps = 1 + (eps_rod − 1) · sigmoid(θ)`, starting from the damaged structure
   itself.
3. **Check the gradient (the G2 discipline).** Sample 3 pixels at random, take
   a central finite difference at each, compare them against `jax.grad` one by
   one, and refuse to go further unless the relative error is below 1e-5.
   **An unverified gradient does not ship** — in this repo that rule is
   institutionalized as the G2 gate
   ([src/invdx/gates/g2_gradcheck.py](../../src/invdx/gates/g2_gradcheck.py)),
   and fdtdx gradients have to clear the same finite-difference comparison
   before they count.
4. **Run Adam.** Let the gradient decide where to put material in the design
   region, and watch transmission climb back up from the damaged value.

## Three things to watch while reading the code

- `simulate()` vs `run()`: the differentiable path has to stay inside JAX from
  end to end — the moment it touches numpy, the gradient chain is cut. This is
  a clean case of an engineering choice serving a physics requirement.
- `objective` = the mean of T: swap `jnp.mean` for `jnp.min` (the worst
  frequency) and you have a minimax objective, which is the idea this repo's
  optimizer runs on. Its smooth version is `softmin`
  ([src/invdx/fab/filters_jax.py](../../src/invdx/fab/filters_jax.py)), used to
  aggregate the per-wavelength figures of merit in
  [src/invdx/optimize.py](../../src/invdx/optimize.py).
- Healing the defect does not have to mean growing the rod back. The gradient
  cares about transmission and nothing else, so the solution it finds may look
  nothing like the original lattice. That is the difference in kind between
  inverse design and putting a structure back the way it was.

## Trying it by hand

```bash
PY=python   # the python of your invdx env
$PY scripts/09_toy_adjoint.py --gradcheck-only          # gradient check only
$PY scripts/09_toy_adjoint.py --tag mine                # the full run
$PY scripts/09_toy_adjoint.py --iters 120 --lr 0.1      # tune the optimizer
```

Change `FSTARS` (the target frequencies), `DEFECT` (knock out a different rod),
or the size of the design box, and you get a different "treatment plan" every
time. Every result lands in `runs/`, so you can put them side by side.

## Back to the big picture

By the end of the toy track you own a **complete inverse-design chain you built
by hand, from scratch**: Yee update equations → differentiable simulation → a
verified adjoint gradient → an optimization loop. When fdtdx runs a 3D device
the principle is identical; only the scale differs, by four orders of
magnitude.

Two paths lead further into the repo:
[src/invdx/gates/g2_gradcheck.py](../../src/invdx/gates/g2_gradcheck.py) is the
same gradient check applied to fdtdx, so you can hold your toy result up
against it, and
[docs/phc-bend-walkthrough.md](../../docs/phc-bend-walkthrough.md) shows the
same structure cross-validated on two engines.
One more thing worth carrying with you: the toy uses a first-order Mur
absorbing boundary, and its residual reflection is the coarsest approximation
anywhere in this chain — if you want numbers closer to a mature solver's, the
boundary condition is the first place to look.
