"""The probe must say "unknown" where it cannot see, never a plausible number.

Its whole purpose is answering "was this measured on the same hardware", and
a probe that guesses gives a confident wrong answer to exactly that question.
So most of these tests are about the absence of information surviving intact.
"""

import dataclasses

import pytest

from invdx import hardware as hw


def test_tf32_follows_compute_capability():
    """Turing has no TF32, Ada does, and unknown stays unknown.

    This is the field that matters most: JAX's default matmul precision uses
    TF32 wherever it exists, so the same source runs at a different precision
    on different GPU generations, and nothing else in a run record says
    which happened.
    """
    assert hw.DeviceProbe(platform="gpu", compute_capability=(7, 5)).has_tf32 is False
    assert hw.DeviceProbe(platform="gpu", compute_capability=(8, 9)).has_tf32 is True
    assert hw.DeviceProbe(platform="gpu", compute_capability=(8, 0)).has_tf32 is True
    assert hw.DeviceProbe(platform="gpu", compute_capability=None).has_tf32 is None


def test_unknowns_are_none_not_zero():
    """A missing SM count must not read as a machine with no SMs."""
    p = hw.DeviceProbe(platform="cpu")
    for field in ("device_kind", "compute_capability", "core_count",
                  "device_memory_bytes", "bytes_limit"):
        assert getattr(p, field) is None, field


def test_probe_never_raises_and_reports_the_backend():
    """Whatever backend the test host has, probing it must not throw."""
    p = hw.probe()
    assert isinstance(p, hw.DeviceProbe)
    assert isinstance(p.platform, str) and p.platform


def test_describe_prints_unknowns_rather_than_hiding_them():
    """An omitted line reads as "nothing to report"; that is the failure."""
    text = hw.describe(hw.DeviceProbe(platform="cpu", n_devices=1))
    assert "unknown" in text
    for label in ("compute_capability", "core_count", "bytes_limit",
                  "TF32 available"):
        assert label in text, label


def test_as_dict_is_json_shaped_and_keeps_the_derived_field():
    import json

    p = hw.DeviceProbe(platform="gpu", compute_capability=(8, 9),
                       notes=("something odd",))
    d = hw.as_dict(p)
    assert d["compute_capability"] == [8, 9]      # tuple would not survive JSON
    assert d["has_tf32"] is True                  # derived, must be recorded
    assert d["notes"] == ["something odd"]
    json.dumps(d)                                 # must not raise


def test_a_frozen_probe_cannot_be_edited_after_the_fact():
    """A run record that can be adjusted later is not a record."""
    p = hw.probe()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.device_kind = "something else"


def test_pin_matmul_precision_reports_what_it_set():
    jax = pytest.importorskip("jax")
    # This jax reads flags as attributes; config.read() is gone.
    before = jax.config.jax_default_matmul_precision
    try:
        assert hw.pin_matmul_precision("highest") == "highest"
        assert jax.config.jax_default_matmul_precision == "highest"
        # And the point of pinning: it must not be whatever the card implies.
        assert hw.pin_matmul_precision("default") == "default"
        assert jax.config.jax_default_matmul_precision == "default"
    finally:
        jax.config.update("jax_default_matmul_precision", before)
