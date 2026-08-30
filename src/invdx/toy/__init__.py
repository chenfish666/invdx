"""toyfdtd — Layer C: a minimal, self-written FDTD kept deliberately
independent of fdtdx/Meep (no shared interfaces, no shared conventions).

Purpose is double: a learning vehicle for the lowest-level machinery (Yee
staggering, leapfrog, absorbing boundaries, adjoint gradients), and a third
independent implementation for cross-checks. Independence is the whole
point — two implementations that share a convention also share that
convention's bugs, and agreement between them proves nothing.

What is here:
    fdtd2d       numpy 2D TM Yee + first-order Mur boundaries; dielectrics
                 (eps in the E update), carrier-modulated pulses, line
                 probes and spectral flux — enough for real spectroscopy,
                 and what drives problems/phc_bend
    fdtd2d_jax   the same physics in JAX: bit-equivalent to the numpy
                 original (9e-16), same API, runs unmodified on GPU, and
                 differentiable — jax.grad straight through simulate(),
                 finite-difference verified to 1e-6. Walkthroughs in
                 tutorials/01 (the port) and tutorials/02 (the adjoint).

Known limit: the absorbing boundary is first-order Mur, not a PML, so edge
reflections set the noise floor for anything that has to run long.
"""
