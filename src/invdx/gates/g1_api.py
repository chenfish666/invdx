"""Gate 1 — engine availability & API surface.

Catches upstream API drift the moment fdtdx is upgraded, before any physics
runs — a name that vanished from the library should fail here in seconds, not
halfway through a GPU job. Also verifies JAX sees a GPU and, when the meep env
exists, that the subprocess bridge answers a ping.
"""

import os

from .runner import GateResult

NAME = "api"
ORDER = 1
REQUIRES = ()
# This gate measures no problem module, so it owes the report no
# `problem` / `problem_module` keys. Declared, not inferred: the runner
# requires the two identity keys from every gate by default, precisely so
# that a gate author who writes nothing gets a loud complaint instead of a
# silent exemption. See `runner._declared_problem`.
MEASURES_PROBLEM = False

# every fdtdx name invdx code touches; extend when the adapter grows.
# targets the RELEASED fdtdx (no UniformGrid — that's post-0.6.2 dev API)
FDTDX_API = (
    "SimulationConfig", "GradientConfig", "Recorder",
    "SimulationVolume", "BoundaryConfig", "boundary_objects_from_config",
    "Material", "UniformMaterialObject", "Device",
    "ParameterTransformation", "TanhProjection", "SubpixelSmoothedProjection",
    "GaussianSmoothing2D", "BrushConstraint2D",
    "UniformPlaneSource", "GaussianPlaneSource", "ModePlaneSource",
    "WaveCharacter", "OnOffSwitch",
    "PoyntingFluxDetector", "EnergyDetector", "ModeOverlapDetector",
    "place_objects", "apply_params", "run_fdtd", "full_backward",
)


def run(cfg, args):
    details = {}

    import fdtdx
    missing = [n for n in FDTDX_API if not hasattr(fdtdx, n)]
    if missing:
        return GateResult(NAME, "fail",
                          {"reason": f"fdtdx API drift, missing: {missing}"})
    details["fdtdx"] = getattr(fdtdx, "__version__", "?")

    import jax
    platforms = [d.platform for d in jax.devices()]
    details["jax_devices"] = str(jax.devices())
    if "gpu" not in platforms:
        return GateResult(NAME, "fail",
                          {"reason": f"no GPU visible to JAX: {platforms}",
                           **details})

    from invdx.engines import meep_bridge
    if os.path.exists(meep_bridge.MEEP_PYTHON):
        pong = meep_bridge.run_job("ping", {})
        details["meep"] = pong.get("meep_version", "?")
    else:
        details["meep"] = "env not present (bridge ping skipped)"

    return GateResult(NAME, "ok", details)
