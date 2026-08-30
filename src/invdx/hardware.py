"""What device this is, and what that implies -- reported, never applied.

Two runs of the same command on two machines should produce the same numbers
or say plainly why they cannot. That is the whole design constraint here, and
it is why this module probes and advises but changes nothing on its own.

The two things it exists to fix:

**Run records could not tell the machines apart.** `env.txt` wrote
`jax.devices: [CudaDevice(id=0)]`, which is byte-identical on a Turing card
and an Ada one. Any question of the form "was this measured on the same
hardware as that?" was unanswerable from the stored run record.

**JAX picks a different float precision per architecture, silently.**
`Precision.DEFAULT` on GPU means "use TF32 where available" -- so the same
source runs in float32 on compute capability 7.5 and in TF32 on 8.9, with a
relative error two orders of magnitude larger, and nothing in the run record
says which happened. PyTorch enabled the same behaviour by default in 1.7,
took two years of user reports, and turned it back off in 1.12 for the stated
reason of giving users clarity about the math mode they are in. JAX still
defaults it on. `pin_matmul_precision()` below makes the choice explicit and
recordable.

What is deliberately NOT here: anything that decides. `recommend()` returns
values with reasons attached and the caller does what it likes. Nothing sets
an environment variable, mutates a config, or picks a grid.
"""

from __future__ import annotations

import dataclasses
import os


@dataclasses.dataclass(frozen=True)
class DeviceProbe:
    """Read-only facts about the accelerator, or None where unknowable.

    Every field is optional on purpose. `compute_capability` reaches JAX
    through `__getattr__` rather than being a declared attribute and does not
    exist on the CPU backend; `memory_stats()` returns None there entirely;
    `core_count` and `device_memory_bytes` are not documented anywhere and
    could vanish in a jaxlib release. A probe that guessed when it could not
    see would be worse than one that says None.
    """

    platform: str
    device_kind: str | None = None
    compute_capability: tuple[int, int] | None = None
    core_count: int | None = None
    device_memory_bytes: int | None = None
    bytes_limit: int | None = None
    n_devices: int = 0
    platform_version: str | None = None
    jax_version: str | None = None
    prealloc: str | None = None
    mem_fraction: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def has_tf32(self) -> bool | None:
        """Ada and later do TF32; Turing does not. None when unknown.

        JAX has no capability query for this -- only a way to declare what you
        want, checked at compile time. Compute capability is the only handle.
        """
        if self.compute_capability is None:
            return None
        return self.compute_capability >= (8, 0)


def probe() -> DeviceProbe:
    """Ask JAX what it is running on. Never raises; unknowns come back None."""
    notes: list[str] = []
    env = {
        "prealloc": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    }
    try:
        import jax
    except Exception as e:
        return DeviceProbe(platform="<no jax>", notes=(f"import jax failed: {e}",),
                           **env)

    devs = jax.devices()
    if not devs:
        return DeviceProbe(platform="<no devices>", jax_version=jax.__version__,
                           notes=("jax.devices() is empty",), **env)
    d = devs[0]

    cc = getattr(d, "compute_capability", None)
    if isinstance(cc, str):
        try:
            major, _, minor = cc.partition(".")
            cc = (int(major), int(minor or 0))
        except ValueError:
            notes.append(f"compute_capability {cc!r} did not parse")
            cc = None
    elif cc is not None and not isinstance(cc, tuple):
        notes.append(f"compute_capability had type {type(cc).__name__}")
        cc = None

    stats = None
    try:
        stats = d.memory_stats()
    except Exception as e:
        notes.append(f"memory_stats() raised: {e}")
    if stats is None and d.platform == "gpu":
        notes.append("memory_stats() returned None on a GPU backend")

    limit = stats.get("bytes_limit") if stats else None
    if stats is not None and limit is None:
        # bytes_limit is one of the conditional keys; its absence is normal on
        # some platforms and must not be read as "no limit".
        notes.append("memory_stats() has no bytes_limit key on this platform")

    if len({getattr(x, "device_kind", None) for x in devs}) > 1:
        notes.append("visible devices are not all the same kind; probe "
                     "describes device 0 only")

    return DeviceProbe(
        platform=d.platform,
        device_kind=getattr(d, "device_kind", None),
        compute_capability=cc,
        core_count=getattr(d, "core_count", None),
        device_memory_bytes=getattr(d, "device_memory_bytes_limit", None),
        bytes_limit=limit,
        n_devices=len(devs),
        platform_version=getattr(getattr(d, "client", None),
                                 "platform_version", None),
        jax_version=jax.__version__,
        notes=tuple(notes),
        **env,
    )


def pin_matmul_precision(precision: str = "highest") -> str:
    """Fix the float precision so it does not depend on which card ran it.

    Left alone, JAX uses TF32 on Ada and float32 on Turing for the same code.
    "highest" means float32 everywhere; pass "default" to accept the
    per-architecture behaviour deliberately rather than by omission. Returns
    what was set, for the run record.
    """
    import jax

    jax.config.update("jax_default_matmul_precision", precision)
    return precision


def as_dict(p: DeviceProbe) -> dict:
    d = dataclasses.asdict(p)
    d["compute_capability"] = list(p.compute_capability) if p.compute_capability else None
    d["notes"] = list(p.notes)
    d["has_tf32"] = p.has_tf32
    return d


def describe(p: DeviceProbe) -> str:
    """One human-readable block. Unknowns are printed as unknown, not omitted."""
    def g(x, unit=""):
        return "unknown" if x is None else f"{x}{unit}"

    lines = [
        f"platform            {p.platform}  ({g(p.n_devices)} device(s))",
        f"device_kind         {g(p.device_kind)}",
        f"compute_capability  {'.'.join(map(str, p.compute_capability))
                               if p.compute_capability else 'unknown'}",
        f"core_count (SMs)    {g(p.core_count)}",
        f"device memory       {'unknown' if p.device_memory_bytes is None
                               else f'{p.device_memory_bytes / 2**20:.0f} MiB'}",
        f"jax bytes_limit     {'unknown' if p.bytes_limit is None
                               else f'{p.bytes_limit / 2**20:.0f} MiB'}"
        + "   <- what JAX will actually let you allocate",
        f"TF32 available      {'unknown' if p.has_tf32 is None
                               else ('yes' if p.has_tf32 else 'no')}",
        f"jax                 {g(p.jax_version)}   {g(p.platform_version)}",
        f"PREALLOCATE         {p.prealloc if p.prealloc is not None else '<unset>'}",
        f"MEM_FRACTION        {p.mem_fraction if p.mem_fraction is not None else '<unset>'}",
    ]
    for n in p.notes:
        lines.append(f"note                {n}")
    return "\n".join(lines)


def main():
    p = probe()
    print(describe(p))
    if p.bytes_limit and p.device_memory_bytes:
        frac = p.bytes_limit / p.device_memory_bytes
        print(f"\nbytes_limit is {100 * frac:.1f}% of device memory. That "
              f"fraction applies to memory FREE at\ninitialisation, not to the "
              f"nameplate size -- another process holding a few hundred\nMiB "
              f"moves it, so it has to be read at runtime rather than derived "
              f"from the model.")


if __name__ == "__main__":
    main()
