"""Fabrication-robustness utilities.

filters_np  — conic filter / tanh projection / softmin (numpy+autograd,
              authoritative reference implementation, meep-env safe)
filters_jax — jnp twins of filters_np for the fdtdx differentiable path
              (parity-tested against filters_np)
transforms  — fdtdx ParameterTransformation subclasses wrapping filters_jax
measure     — linewidth / CD-corner measurement (numpy only, never differentiated)
"""
