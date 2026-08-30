"""Minimal 2D TM-polarization FDTD (Ez, Hx, Hy) on a uniform Yee grid.

Natural units: c = 1, eps0 = mu0 = 1. Grid spacing dx, time step
dt = courant * dx (2D stability bound is courant < 1/sqrt(2)).

Yee staggering (2D TM):
    Ez[i, j]  at (i,      j     )
    Hx[i, j]  at (i,      j+1/2 )   shape (nx, ny-1)
    Hy[i, j]  at (i+1/2,  j     )   shape (nx-1, ny)

Leapfrog: update H from curl E, then E from curl H, then inject the soft
source, then record probes. Boundaries: first-order Mur absorbing on all
four edges — no PML, so edge reflections set the noise floor (see
toy/__init__.py).
"""

import numpy as np


def gaussian_pulse(t, t0, spread, fcen=None):
    """Gaussian envelope, optionally carrier-modulated.

    Without fcen the pulse is BASEBAND: its spectrum peaks at f = 0 and dies
    like exp(-(pi*f*spread)^2) — fine for travel-time tests, useless for
    spectroscopy away from DC (measured: 8 orders of magnitude of power
    collapse across f = 0.2..0.5). With fcen the envelope multiplies a
    sin carrier and the spectrum is centered on fcen (Meep's GaussianSource
    convention).
    """
    env = np.exp(-(((t - t0) / spread) ** 2))
    if fcen is None:
        return env
    return np.sin(2 * np.pi * fcen * (t - t0)) * env


def run(nx, ny, dx, steps, source, probes=(), courant=0.5, eps=None,
        line_probes=None, field_dft_freqs=None):
    """Run the simulation.

    source — dict(i=..., j=..., t0=..., spread=...): soft Gaussian-pulse
             point source added to Ez at (i, j)
    probes — sequence of (i, j) grid points; Ez is recorded there every step
    eps    — optional (nx, ny) relative permittivity. Enters the Yee update
             exactly once: the E step becomes dEz/dt = (curl H)/eps — this is
             THE line where a material changes electromagnetics. Must be 1.0
             on the outermost cells (the Mur boundary assumes vacuum speed).
    field_dft_freqs — optional list of frequencies: accumulate the full-domain
             complex Ez(f) phasor (running DFT) over the whole run. Re(Ez(f))
             is the steady-state field pattern at that frequency — the
             "light flowing through the structure" figure — extracted from
             the same pulsed run that measures spectra.
    line_probes — {name: (axis, k, lo, hi)}: record complex-flux ingredients
             along a grid line every step.
               axis "x": vertical line at i=k, span Ez[k, lo:hi] and
                         Hy[k, lo:hi] (power flow along +x: Sx = -Ez*Hy)
               axis "y": horizontal line at j=k, span Ez[lo:hi, k] and
                         Hx[lo:hi, k] (power flow along +y: Sy = +Ez*Hx)
             The recorded H sits half a cell off the Ez line (Yee staggering);
             transmission RATIOS between runs on the same grid cancel that
             offset, which is the only way this toy uses line probes.

    Returns dict with:
        Ez            final Ez field (nx, ny)
        t             time axis, shape (steps,)
        probes        {(i, j): Ez trace (steps,)}
        lines         {name: {"E": (steps, n), "H": (steps, n)}}
        energy        0.5*sum(eps*Ez^2 + Hx^2 + Hy^2)*dx^2 per step
    """
    dt = courant * dx
    Ez = np.zeros((nx, ny))
    Hx = np.zeros((nx, ny - 1))
    Hy = np.zeros((nx - 1, ny))
    if eps is None:
        eps = np.ones((nx, ny))
    else:
        eps = np.asarray(eps, dtype=float)
        edge = np.concatenate([eps[0], eps[-1], eps[:, 0], eps[:, -1]])
        if not np.allclose(edge, 1.0):
            raise ValueError("eps must be 1.0 on the outermost cells "
                             "(Mur boundary assumes vacuum)")
    Ez_old = Ez.copy()                       # previous step, for Mur
    mur = (dt - dx) / (dt + dx)              # first-order Mur coefficient, c=1

    t_axis = np.arange(steps) * dt
    traces = {tuple(p): np.zeros(steps) for p in probes}
    field_dft_freqs = tuple(field_dft_freqs or ())
    field_dft = {f: np.zeros((nx, ny), dtype=complex)
                 for f in field_dft_freqs}
    line_probes = line_probes or {}
    lines = {}
    for name, (axis, k, lo, hi) in line_probes.items():
        lines[name] = {"E": np.zeros((steps, hi - lo)),
                       "H": np.zeros((steps, hi - lo))}
    energy = np.zeros(steps)

    for n in range(steps):
        # H from curl E (half-step)
        Hx -= (dt / dx) * (Ez[:, 1:] - Ez[:, :-1])
        Hy += (dt / dx) * (Ez[1:, :] - Ez[:-1, :])

        # E interior from curl H, slowed by the local permittivity
        Ez_old[:] = Ez
        Ez[1:-1, 1:-1] += (dt / dx) / eps[1:-1, 1:-1] * (
            (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) - (Hx[1:-1, 1:] - Hx[1:-1, :-1]))

        # soft source: point at (i, j), or a vertical line at column i
        # spanning j0:j1 when those keys are present (quasi plane wave)
        amp = gaussian_pulse(n * dt, source["t0"], source["spread"],
                             source.get("fcen"))
        if "j0" in source:
            Ez[source["i"], source["j0"]:source["j1"]] += amp
        else:
            Ez[source["i"], source["j"]] += amp

        # first-order Mur on the four edges
        Ez[0, :] = Ez_old[1, :] + mur * (Ez[1, :] - Ez_old[0, :])
        Ez[-1, :] = Ez_old[-2, :] + mur * (Ez[-2, :] - Ez_old[-1, :])
        Ez[:, 0] = Ez_old[:, 1] + mur * (Ez[:, 1] - Ez_old[:, 0])
        Ez[:, -1] = Ez_old[:, -2] + mur * (Ez[:, -2] - Ez_old[:, -1])

        for p, trace in traces.items():
            trace[n] = Ez[p]
        for name, (axis, k, lo, hi) in line_probes.items():
            if axis == "x":
                lines[name]["E"][n] = Ez[k, lo:hi]
                lines[name]["H"][n] = Hy[min(k, nx - 2), lo:hi]
            else:
                lines[name]["E"][n] = Ez[lo:hi, k]
                lines[name]["H"][n] = Hx[lo:hi, min(k, ny - 2)]
        for f in field_dft_freqs:
            field_dft[f] += Ez * (np.exp(-2j * np.pi * f * n * dt) * dt)
        energy[n] = 0.5 * (np.sum(eps * Ez ** 2) + np.sum(Hx ** 2)
                           + np.sum(Hy ** 2)) * dx * dx

    return {"Ez": Ez, "t": t_axis, "probes": traces, "lines": lines,
            "energy": energy, "field_dft": field_dft}


def line_flux_spectrum(line, freqs, dt, dx, sign=1.0):
    """Spectral power through a recorded line probe.

    DFT the E and H traces at the requested frequencies (cycles per unit
    time, c=1 units) and form P(f) = sign * 0.5 * Re sum(E(f) * conj(H(f)))
    * dx. For an "x" line, power flowing toward +x is sign=-1 (Sx = -Ez*Hy);
    for a "y" line, +y flow is sign=+1 (Sy = +Ez*Hx). Absolute values carry
    the arbitrary source spectrum — only RATIOS between runs with the same
    source and grid are physical. This is the toy engine's instance of a rule
    that holds for every engine here: a run and the normalization run it is
    divided by must share every numerical convention, or the ratio silently
    measures the difference between the conventions instead of the physics.
    """
    E, H = line["E"], line["H"]
    steps = E.shape[0]
    t = np.arange(steps) * dt
    ker = np.exp(-2j * np.pi * np.asarray(freqs)[:, None] * t[None, :]) * dt
    Ef = ker @ E                        # (nf, nline)
    Hf = ker @ H
    return sign * 0.5 * np.real(np.sum(Ef * np.conj(Hf), axis=1)) * dx
