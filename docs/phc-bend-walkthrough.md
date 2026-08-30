> **English** · [繁體中文](phc-bend-walkthrough.zh-TW.md)

[← back to docs index](README.md)

# Reproduce it yourself: the photonic-crystal 90° bend (the classic benchmark from the literature)

This tutorial walks you through reproducing the classic benchmark of the
photonic-crystal waveguide literature **by hand, end to end**: a 2D square
lattice of dielectric rods, a 90° bend waveguide, and what a point defect does
to the transmission. High-transmission bends in structures like this have been
standard teaching material in the PhC-waveguide literature ever since Mekis
et al. (*"High Transmission through Sharp Bends in Photonic Crystal
Waveguides"*, Phys. Rev. Lett. 77, 3787, 1996). You will use both engines that
ship with this project: `toy/` (a ~160-line FDTD you can read line by line) and
Meep (the reference implementation everyone in the field checks against).

**Benchmark parameters** (all of them written into `PhCBendConfig`; no number is
hidden inside a script):

| Parameter | Value | Meaning |
|---|---|---|
| a | 1 µm | lattice constant (every length in the code is in units of a) |
| R | 0.225a | rod radius |
| ε | 10 (n≈3.162) | dielectric constant of the rods |
| gap | f = 0.29–0.41 | normalized frequency f = a/λ (typical literature value, our anchor) |
| polarization | E along the rod axis | ⚠️ see the naming trap below |

> **Naming trap (read this first)**: engineering papers routinely say they launch
> "TE" light, but the gap of a rod lattice belongs to the polarization with
> **E parallel to the rod axis** — which the physics textbook (Joannopoulos,
> Johnson, Winn & Meade, *Photonic Crystals: Molding the Flow of Light*) calls
> **TM**, while a good deal of engineering literature calls it TE. Our toy engine
> evolves (Ez, Hx, Hy), which is exactly E along the rod axis, so the polarization
> is right. Do not memorize the name; remember "E along the rods" and you will
> never get it wrong. This is the most common convention trap when you compare
> results across papers, and it belongs to the same family as the ½-power
> convention recorded in `engines/conventions.py`.

Every step is one command. The defaults are the benchmark size (21×21), so one
step takes roughly 30–60 seconds; if you want to experiment quickly, add
`--set n_side=11 --set toy_steps=3000`, which brings each step down to a few
seconds.

```bash
cd <invdx repo>
PY=python   # the python of your invdx env
```

---

## Step 0: read the engine itself (15 minutes)

Open [src/invdx/toy/fdtd2d.py](../src/invdx/toy/fdtd2d.py). All of
electromagnetism sits in three update lines:

- `Hx -= (dt/dx)*(∂Ez/∂y)`, `Hy += (dt/dx)*(∂Ez/∂x)` — Faraday's law
- `Ez += (dt/dx)/eps * curl(H)` — Ampère's law; the **only place the material
  enters** is that `1/eps`

Checkpoint: why does `eps` appear only as a divisor in the E update, and never
anywhere in the H update? (Hint: non-magnetic material means µ=1; a dielectric
only changes the electric displacement D=εE.)

## Step 1: look at the geometry

```bash
$PY scripts/06_phc_bend.py --stage eps --tag walkthrough
```

This prints an ASCII picture of three layouts (`o`=rod, `.`=empty): `bulk` (the
8-period slab used to measure the gap), `straight` (the straight waveguide = the
normalization reference), and `bend` (the 90° bend).

Checkpoint: the bend takes light in on the left and out at the top; the lattice
carries one ring more than n_side (indices −1 and n_side). That extra ring is
something we learned the hard way: without it, light walks around the crystal
through the vacuum corridor beside it, and the measured in-gap suppression
collapses from −40 dB to −2 dB, drowned in bypass leakage. **The boundary
design of a measurement structure matters as much as the structure itself.**

## Step 2: find the gap (checking against the anchor)

```bash
$PY scripts/06_phc_bend.py --stage gap --tag walkthrough
```

Expect: the stopband (T < −20 dB, floor around −50 dB) lands at
**f ≈ 0.27–0.41**.

Before you hold that up against the literature's 0.29–0.41, there is a physical
detail worth getting straight: what we measure is **transmission at normal
incidence (the Γ-X direction)**, and its stopband is wider than the *complete*
gap — the complete gap is the intersection of the stopbands over all propagation
directions, and its lower edge is usually set by the M direction. So the correct
criterion is that **our stopband must contain the literature's complete gap**
(✓ 0.27–0.41 ⊇ 0.29–0.41), not that the edges agree point by point. The two
independent engines (toy and Meep) put the stopband edges within ~0.01 of each
other, and that carries more evidential weight than matching a single number from
a paper: one more application of **lesson 6** in
[`engines/conventions.py`](../src/invdx/engines/conventions.py) (compare shapes,
not single points).

Hands-on experiments (do at least one):
- `--set eps_rod=8.9` (alumina, the value in the Joannopoulos textbook): the gap
  should narrow and shift up — less dielectric contrast, smaller gap.
- `--set r_rod=0.18`: thinner rods; how do the gap's position and width move?
- `--set res_per_a=30`: finer discretization; the lower edge should sit closer to
  the literature value.

## Step 3: 90° bend transmission (the benchmark's headline result)

```bash
$PY scripts/06_phc_bend.py --stage bend --tag walkthrough
```

Expect: **T ≈ 0.85–1.1 inside the gap** — the gap forbids light from entering the
crystal, so it has no choice but to follow the channel of missing rods around
the 90° corner, almost losslessly. That is the magic of a photonic-crystal
waveguide (the high-transmission bend of Mekis et al. 1996), and the core result
of this benchmark.

Checkpoint: why does the script print T only inside the gap? What are those
|T|>1 and even negative values outside it? (Answer: outside the gap the crystal
forbids nothing, the "waveguide" does not guide at all, and what the output line
measures is the interference of scattered noise — neither numerator nor
denominator means anything physical. Knowing **when a measurement is invalid** is
as important as knowing how to measure.)

## Step 4: Meep cross-check (how trust gets built)

```bash
$PY scripts/06_phc_bend.py --stage meep --tag walkthrough     # ~a few minutes
$PY scripts/06_phc_bend.py --stage compare --tag walkthrough
```

The same rod coordinates (`rod_centers_a`; both engines import the same function)
go into Meep, with PML absorbing boundaries and subpixel smoothing. Expect: the
two engines agree on where the gap is and on the shape of the in-gap transmission
curve; the point-by-point values differ (toy is first-order Mur on a binary grid,
Meep is PML plus smoothing) — and **you can name every source of that
difference**. That is what makes it a cross-check.

## Step 5: point-defect sweep (the literature's conclusion)

```bash
$PY scripts/06_phc_bend.py --stage defect --tag walkthrough   # ~4 minutes
```

Pull out one rod on the outer side of the corner (Layer I = nearest layer,
II = second layer; three orientations: horizontal / vertical / slant) and see how
far the in-gap mean transmission drops (the script prints that drop as the
`delta` column). The common conclusion in the literature: **vertical and
horizontal defects matter far more than slanted ones**. Does the ordering of
your deltas reproduce it?

## Step 6 (looking ahead): from here to inverse design

What the early literature did: pick a few defect positions by hand and simulate
them one at a time (or apply a shift-or-shrink heuristic). What inverse design
does: treat every rod's position and radius as a differentiable parameter, and get
all of the gradients at once with the adjoint method. Same bend, same FDTD, two
decades of method evolution in between — that is the bridge from this classic
benchmark to inverse design. To walk that step yourself, read
[`tutorials/01-jax-port`](../tutorials/01-jax-port/) (porting the toy engine to
JAX) next, and then
[`tutorials/02-first-adjoint`](../tutorials/02-first-adjoint/) (checking your
first adjoint gradient against finite differences).

---

## Appendix: where everything lands after a run

Every command goes through `start_run`, which creates
`runs/<timestamp>-phc-bend-walkthrough/` holding `config.json` (including your
`--set` overrides), `cmdline.txt`, `env.txt`, and each stage's JSON/npy results —
so six months from now you can still rebuild every figure you made today.
