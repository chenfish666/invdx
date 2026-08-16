"""Engine adapters (Layer A access). The engines themselves are never modified.

fdtdx_engine — scene factories / smoke sims on the fdtdx GPU path (jax)
meep_bridge  — parent side of the cross-env subprocess protocol (invdx env)
meep_worker  — child side, runs inside conda env `meep` (numpy-pure imports only)
conventions  — cross-engine convention contracts (the hard-won-lessons module)
"""
