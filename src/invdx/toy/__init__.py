"""toyfdtd — Layer C: a minimal, self-written FDTD kept deliberately
independent of fdtdx/Meep (no shared interfaces, no shared conventions).

Purpose is double: a learning vehicle for the lowest-level machinery (Yee
staggering, leapfrog, absorbing boundaries, eventually adjoint), and a third
independent implementation for cross-checks — pvgc showed how much a truly
independent reference catches (the reciprocity gate found a 2x bug).

Growth roadmap:
    M-toy-1  numpy 2D TM Yee + first-order Mur boundaries      (done: fdtd2d)
    M-toy-1b dielectrics (eps in the E update), carrier-modulated pulses,
             line probes + spectral flux — enough for real spectroscopy
             (done: drives problems/phc_bend, the literature benchmark)
    M-toy-2  JAX port — DONE: fdtd2d_jax, bit-equivalent to numpy (9e-16),
             same API, runs unmodified on GPU; tutorial in tutorials/01
    M-toy-3  split-field PML (replace Mur)
    M-toy-4  adjoint gradients — DONE: jax.grad through simulate(), FD-
             verified to 1e-6; defect-healing inverse design on the PhC
             bend (0.613 -> 0.999, beats intact 0.972); tutorials/02
    M-toy-5  registered as a third gradient reference in gates G2/G5
"""
