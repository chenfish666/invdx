"""Concrete design problems. Each module = one problem: a config subclass,
scene builders for the engines, and measurement/FOM definitions with their
convention contracts. Included: `grating_coupler`, a fiber-to-chip grating coupler, and
`phc_bend`, a photonic-crystal waveguide bend benchmark.

There is no registry and no plugin hook here on purpose -- this package is a
namespace, and scripts, gates and tests reach a problem by importing its name.
Adding one is therefore a matter of writing a module that exposes the right
names and then editing the consumers that need it. What those names are, which
module to copy from, and which convention contracts fail silently rather than
raising: docs/new-problem.md.
"""
