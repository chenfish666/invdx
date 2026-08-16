"""invdx — photonic inverse-design toolbox.

Layer A: engines (fdtdx GPU + Meep cross-validation, unmodified deps)
Layer B: methodology (config/cli/runio, fab, modes, gates, export, conventions)
Layer C: toy (self-written minimal FDTD, learning + independent reference)

Import discipline: modules that must run inside the meep conda env
(runio, fab.filters_np, fab.measure, modes, engines.conventions,
engines.meep_worker) may only use numpy/scipy/autograd — never jax/fdtdx.
"""

__version__ = "0.1.0"

from .config import BaseConfig  # noqa: F401
