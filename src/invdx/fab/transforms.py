"""fdtdx ParameterTransformation subclasses wrapping filters_jax.

ConicFilter1D fills the gap in fdtdx (which only mentions the conic filter in
a docstring): a cone-kernel density filter along one voxel axis, pluggable
into a Device pipeline ahead of fdtdx's own projection, e.g.

    fdtdx.Device(
        materials=...,
        param_transforms=[ConicFilter1D(radius_um=0.13), fdtdx.TanhProjection()],
        ...,
    )

Written against the RELEASED fdtdx==0.6.2 transform API
(SameShapeTypeParameterTransform; see fdtdx/objects/device/parameters/transform.py).
"""

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.objects.device.parameters.transform import SameShapeTypeParameterTransform
from fdtdx.typing import ParameterType

from .filters_np import conic_filter_matrix


@autoinit
class ConicFilter1D(SameShapeTypeParameterTransform):
    """Normalized conic (cone-kernel) density filter along one voxel axis.

    The kernel radius would set a GUARANTEED length scale only under the
    three-field robust formulation (optimise the eroded/nominal/dilated
    projections together, eta_e=0.75 giving min feature == radius). This repo
    optimises the nominal field alone, so here the radius is a heuristic and
    nothing about the produced design's minimum feature is guaranteed -- see
    BaseConfig.filter_radius and docs/tolerance.md, and measure the result
    rather than assuming it. The filter matrix is a static trace-time
    constant built from the device's own voxel size.
    """

    #: cone kernel radius in um (typically cfg.filter_radius == cfg.min_feature)
    radius_um: float = frozen_field(default=0.130)

    #: voxel-grid axis along which the 1D design varies
    axis: int = frozen_field(default=0)

    _fixed_input_type: ParameterType = frozen_private_field(
        default=ParameterType.CONTINUOUS)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        result = {}
        for k, v in params.items():
            n = v.shape[self.axis]
            voxel_um = self._single_voxel_size[self.axis] * 1e6
            W = jnp.asarray(conic_filter_matrix(n, self.radius_um, 1.0 / voxel_um),
                            dtype=v.dtype)
            moved = jnp.moveaxis(v, self.axis, 0)
            filtered = jnp.tensordot(W, moved, axes=1)
            result[k] = jnp.moveaxis(filtered, 0, self.axis)
        return result
