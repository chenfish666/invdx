# fdtdx upstream issue drafts (NOT yet filed)

Two candidate reports against the pinned PyPI release `fdtdx==0.6.2`,
written during invdx's PVGC work. Draft status: review, then decide
whether/where to file (the dev branch has rewritten the source-profile
path, so both should be re-verified against it first — they may already be
fixed there).

---

## Issue 1 (bug): GaussianPlaneSource produces all-NaN fields on strongly rectangular source planes

**Version**: fdtdx 0.6.2 (PyPI), jax 0.4.x, CUDA 12

**Summary**: `GaussianPlaneSource` silently injects NaN into the whole
simulation when its plane is strongly rectangular (e.g. 800 x 4 cells, a
quasi-2D setup with a thin periodic axis). Square planes are unaffected,
which is presumably why tests pass.

**Root cause** (`fdtdx/objects/sources/linear_polarization.py`,
`_gauss_profile`): the coordinate grid is built as

```python
grid = jnp.stack(
    jnp.meshgrid(*map(jnp.arange, (height, width)), indexing="xy"),
    axis=-1) - jnp.asarray(center)
```

With `indexing="xy"` the stacked coordinate pairs are ordered
`(height_coord, width_coord)`, while the `center` argument arrives ordered
`(horizontal, vertical)` from `_get_amplitude_raw`. On a square plane the
swap is invisible; on a rectangular plane every grid point lands outside
the truncation mask (`euc_dist < 1`), so `profile` is all zeros and the
normalization `profile / profile.sum()` evaluates 0/0 = NaN. The NaN
amplitude then propagates through the injected field into the entire
volume within a few steps.

**Minimal reproduction** (quasi-2D cell, 800 x 4 x N grid):

```python
# GaussianPlaneSource on a z-normal plane spanning 800 x 4 cells:
# after ~10 steps every field component is NaN.
# Replacing it with UniformPlaneSource (same placement) runs fine,
# isolating the profile builder as the cause.
```

**Suggested fix**: build the grid in the same order the center is passed,
e.g. `jnp.meshgrid(*map(jnp.arange, (width, height)), indexing="ij")` with
matching stack order — and, independently, guard the normalization
(`profile.sum()` can be zero whenever radius/std clip the whole plane;
an explicit error beats silent NaN).

**Workaround used in invdx**: a subclass overriding only
`_get_amplitude_raw` with a consistently-ordered profile
(`invdx/engines/fdtdx_fixes.py`).

---

## Issue 2 (docs/UX): tilt semantics of azimuth/elevation for plane sources are easy to misread

**Version**: fdtdx 0.6.2 (PyPI)

**Summary**: for a plane source the mapping from `azimuth_angle` /
`elevation_angle` to the physical tilt direction is underdocumented. What
we measured (z-normal plane, horizontal axis = x): `azimuth_angle` tilts
the propagation direction toward the source's *horizontal* transverse
axis, `elevation_angle` toward the *vertical* one. A user thinking in
spherical coordinates about the global z-axis (the usual reading of those
words) will tilt into the wrong plane — in our quasi-2D case, into a
4-cell periodic axis, which fails silently (beam still runs, physics is
wrong).

**Ask**: one docstring paragraph on `LinearlyPolarizedPlaneSource`
defining both angles relative to the plane's horizontal/vertical axes,
plus which sign convention is used. A diagram would be ideal.

**Note**: we deliberately do NOT report a sign bug here. We initially
believed the azimuth sign was inconsistent across grid resolutions, but
that turned out to be an artifact of reading the *sign* of a phasor
phase slope, which is a time-reference convention of the detector, not
physics (the injected time offsets were identical). Only the slope
magnitude is a valid check. Kept here so we don't re-report the red
herring later.
