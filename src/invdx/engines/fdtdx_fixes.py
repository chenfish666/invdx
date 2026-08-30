"""Minimal local fixes for bugs in the pinned fdtdx release.

Subclasses fdtdx (c) Yannik Mahlau et al., MIT License (github.com/ymahlau/fdtdx) (the sanctioned
"vendor one module when blocked" escape hatch — see README Layer A notes).

GaussianBeamSource — replaces fdtdx.GaussianPlaneSource (0.6.2).

Upstream bug: GaussianPlaneSource._gauss_profile builds its coordinate grid
in (vertical, horizontal) order but receives `center` in (horizontal,
vertical) order. For square source planes the swap is invisible, which is why
upstream tests pass; for strongly rectangular planes (the quasi-2D 800x4
cross-section used here) every grid point lands outside the truncation mask and the
profile normalizes to 0/0 = NaN. Verified against fdtdx 0.6.2 source; the
dev branch has rewritten this path entirely, so re-check when re-pinning.

The subclass only overrides the amplitude profile (a 1D Gaussian along the
horizontal transverse axis, matching grating_coupler's line-source convention
exp(-((x-x0)/w0)^2)); everything else — TFSF injection, tilt handling,
energy normalization — is inherited from the upstream class, which the
UniformPlaneSource diagnostic validated end-to-end.
"""

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.sources.linear_polarization import LinearlyPolarizedPlaneSource


@autoinit
class GaussianBeamSource(LinearlyPolarizedPlaneSource):
    """Plane source with Gaussian amplitude exp(-(r/waist_radius)^2).

    profile_axis selects the shape:
      "horizontal" — 1D Gaussian along the horizontal transverse axis,
                     uniform along the vertical one (quasi-2D line beam)
      "vertical"   — 1D Gaussian along the vertical transverse axis
                     (e.g. exciting a slab from an x-plane: profile over z)
      "radial"     — 2D radial Gaussian over both transverse axes (3D beam)
    waist_radius in meters (1/e FIELD radius, matching grating_coupler's exp(-(x/w0)^2)).
    """

    #: 1/e field radius w0 of the beam (meters)
    waist_radius: float = frozen_field(default=4.6e-6)

    #: which transverse axes carry the Gaussian profile
    profile_axis: str = frozen_field(default="horizontal")

    def _get_amplitude_raw(
        self,
        center: jax.Array,
    ) -> jax.Array:
        w0_cells = self.waist_radius / self._config.resolution
        h_axis, v_axis = self.horizontal_axis, self.vertical_axis
        n_h, n_v = self.grid_shape[h_axis], self.grid_shape[v_axis]
        # center comes as (horizontal, vertical) — the order the upstream
        # GaussianPlaneSource gets wrong (see module docstring)
        idx_h = jnp.arange(n_h, dtype=self._config.dtype)
        idx_v = jnp.arange(n_v, dtype=self._config.dtype)
        g_h = jnp.exp(-(((idx_h - center[0]) / w0_cells) ** 2))
        g_v = jnp.exp(-(((idx_v - center[1]) / w0_cells) ** 2))

        sh_h, sh_v = [1, 1, 1], [1, 1, 1]
        sh_h[h_axis], sh_v[v_axis] = n_h, n_v
        if self.profile_axis == "horizontal":
            profile = g_h.reshape(sh_h)
        elif self.profile_axis == "vertical":
            profile = g_v.reshape(sh_v)
        elif self.profile_axis == "radial":
            profile = g_h.reshape(sh_h) * g_v.reshape(sh_v)
        else:
            raise ValueError(f"unknown profile_axis: {self.profile_axis}")
        return jnp.broadcast_to(profile, self.grid_shape).astype(self._config.dtype)
