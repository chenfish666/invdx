"""Figure rendering from run-dir snapshots (matplotlib, headless Agg)."""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Font stack with a CJK fallback. Built-in labels are English, but titles can
# come from run snapshots or user-supplied strings that carry non-Latin text,
# and DejaVu alone renders those as tofu boxes. addfont() registers the system
# Noto CJK explicitly — rcParams alone is not enough when matplotlib's font
# cache predates the font install.
from matplotlib import font_manager

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if os.path.exists(_f):
        try:
            font_manager.fontManager.addfont(_f)
        except Exception:
            pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK JP",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Validated CVD-safe categorical palette, FIXED order (dataviz reference
# palette, light mode). Never cycle: series 4+ means the chart is overloaded.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#1a1a19", "#5f5e56", "#e6e6e3"


def _ax(title, xlabel, ylabel, figsize=(7.2, 4.4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    return fig, ax


def _save(fig, path, pdf=False):
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    if pdf:
        fig.savefig(path[:-4] + ".pdf", facecolor="white")
    plt.close(fig)
    return path


def _peak_label(ax, lam, db):
    ax.annotate(f"peak {db:.2f} dB @ {lam:.3f} µm",
                xy=(lam, db), xytext=(8, 10), textcoords="offset points",
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))


def plot_ce_spectrum(res, out_png, title="CE spectrum", pdf=False):
    """CE(λ) line(s): nominal + optional CD corners (07 verify results)."""
    fig, ax = _ax(title, "wavelength (µm)", "CE (dB)")
    lam = [r["lam_um"] for r in res["spectrum"]]
    ce = [r["CE_dB"] for r in res["spectrum"]]
    ax.plot(lam, ce, color=C1, linewidth=2, marker="o", markersize=4,
            label="nominal")
    pk = max(res["spectrum"], key=lambda r: r["CE_dB"])
    _peak_label(ax, pk["lam_um"], pk["CE_dB"])

    corners = res.get("corners") or {}
    styles = {"erode_10nm": (C2, "erode -10 nm"),
              "dilate_10nm": (C3, "dilate +10 nm")}
    for name, c in corners.items():
        col, lab = styles.get(name, (INK2, name))
        ax.plot([r["lam_um"] for r in c["spectrum"]],
                [r["CE_dB"] for r in c["spectrum"]],
                color=col, linewidth=2, marker="o", markersize=4, label=lab)
    if corners:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, out_png, pdf)


def plot_transmission(curves, out_png, title, ylabel="T (dB)", gap_ref=None,
                      pdf=False):
    """T(f) curves for phc_bend results. curves = [(freqs, T_dB, label)];
    gap_ref = (f_lo, f_hi) shades the reference band-gap window."""
    fig, ax = _ax(title, "normalized frequency f = a/λ", ylabel)
    if gap_ref:
        ax.axvspan(*gap_ref, color=GRID, alpha=0.6, lw=0,
                   label=f"reference gap {gap_ref[0]}–{gap_ref[1]}")
    for (f, T, label), col in zip(curves, (C1, C2, C3)):
        ax.plot(f, T, color=col, linewidth=2, label=label)
    if len(curves) > 1 or gap_ref:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    return _save(fig, out_png, pdf)


def plot_history(res, out_png, title="optimization trace", pdf=False):
    """Adjoint-optimization trace with damaged/intact reference levels."""
    fig, ax = _ax(title, "iteration", "mean T")
    h = res["history"]
    ax.plot(range(len(h)), h, color=C1, linewidth=2, label="_")
    for key, lab in (("T_intact", "intact"), ("T_damaged", "damaged")):
        if key in res:
            ax.axhline(res[key], color=INK2, linewidth=1, linestyle="--")
            ax.annotate(f"{lab} {res[key]:.3f}", xy=(0, res[key]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=INK2)
    if "T_healed" in res:
        ax.annotate(f"healed {res['T_healed']:.3f}",
                    xy=(len(h) - 1, h[-1]), xytext=(-8, 10),
                    textcoords="offset points", ha="right",
                    fontsize=9, color=INK)
    return _save(fig, out_png, pdf)


def plot_eps(eps, out_png, title="permittivity", extent=None, pdf=False):
    """Permittivity map — sequential single-hue ramp (magnitude data)."""
    fig, ax = _ax(title, "x", "y", figsize=(6.4, 5.6))
    ax.grid(False)
    im = ax.imshow(np.asarray(eps).T, origin="lower", cmap="Blues",
                   extent=extent, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="ε", shrink=0.85)
    return _save(fig, out_png, pdf)


def plot_field(field, eps, out_png, title="Re Ez", extent=None, pdf=False,
               xlabel="x (a)", ylabel="y (a)"):
    """Steady-state field map: Re(Ez) on a diverging red/blue ramp (field
    polarity is genuinely diverging data — zero is physical), structure
    overlaid as a translucent gray mask (the Meep-tutorial look)."""
    re = np.real(np.asarray(field))
    vmax = float(np.percentile(np.abs(re), 99.5))
    fig, ax = _ax(title, xlabel, ylabel, figsize=(6.8, 6.0))
    ax.grid(False)
    ax.imshow(re.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=extent, interpolation="bilinear")
    struct = np.ma.masked_where(np.asarray(eps).T <= 1.5,
                                np.ones_like(np.asarray(eps).T))
    ax.imshow(struct, origin="lower", cmap="gray", vmin=0, vmax=2,
              alpha=0.35, extent=extent, interpolation="nearest")
    ax.set_aspect("equal")
    return _save(fig, out_png, pdf)


def _load(path):
    with open(path) as f:
        return json.load(f)


def render_run(run_dir, out_dir=None, pdf=False):
    """Auto-detect what a run dir contains and render every known figure."""
    out_dir = out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    made = []

    rp = os.path.join(run_dir, "results.json")
    if os.path.exists(rp):
        res = _load(rp)
        if "history" in res:
            made.append(plot_history(
                res, os.path.join(out_dir, "history.png"),
                title="adjoint-gradient optimization trace", pdf=pdf))
        if "spectrum" in res:
            made.append(plot_ce_spectrum(
                res, os.path.join(out_dir, "spectrum.png"),
                title="CE spectrum (independent fdtdx measurement)"
                if "linewidth" in res else "CE spectrum", pdf=pdf))

    gp = os.path.join(run_dir, "gap.json")
    if os.path.exists(gp):
        g = _load(gp)
        f = np.array(g["freqs"])
        Tdb = 10 * np.log10(np.abs(g["T"]) + 1e-12)
        made.append(plot_transmission(
            [(f, Tdb, "toy normal-incidence transmission")],
            os.path.join(out_dir, "bulk_gap.png"),
            "photonic-crystal band gap (bulk transmission)",
            gap_ref=(0.29, 0.41), pdf=pdf))

    bp = os.path.join(run_dir, "bend.json")
    if os.path.exists(bp):
        b = _load(bp)
        f = np.array(b["freqs"])
        curves = [(f, np.array(b["T"]), "toy")]
        mp = os.path.join(run_dir, "meep.json")
        if os.path.exists(mp):
            m = _load(mp)
            curves.append((np.array(m["freqs"]), np.array(m["T_bend"]),
                           "Meep"))
        # only the in-gap portion is physical (outside it the "waveguide"
        # does not guide and the ratio is noise — see the walkthrough)
        curves = [(fq[(fq >= 0.29) & (fq <= 0.41)],
                   np.asarray(T)[(fq >= 0.29) & (fq <= 0.41)], lab)
                  for fq, T, lab in curves]
        made.append(plot_transmission(
            curves, os.path.join(out_dir, "bend_T.png"),
            "90° bend transmission (in-gap)", ylabel="T", pdf=pdf))

    for eps_path in sorted(glob.glob(os.path.join(run_dir, "eps_*.npy"))):
        name = os.path.basename(eps_path)[:-4]
        made.append(plot_eps(np.load(eps_path),
                             os.path.join(out_dir, f"{name}.png"),
                             title=name, pdf=pdf))
    for fz in sorted(glob.glob(os.path.join(run_dir, "field_*.npz"))):
        z = np.load(fz)
        name = os.path.basename(fz)[:-4]
        if "extent" in z:
            ext = tuple(np.asarray(z["extent"], dtype=float))
        elif "extent_a" in z:
            L = float(z["extent_a"])
            ext = (0, L, 0, L)
        else:
            ext = None
        title = str(z["title"]) if "title" in z else "steady-state field Re Ez"
        unit = "µm" if "extent" in z else "a"
        made.append(plot_field(z["field"], z["eps"],
                               os.path.join(out_dir, f"{name}.png"),
                               title=title, extent=ext, pdf=pdf,
                               xlabel=f"x ({unit})", ylabel=f"y ({unit})"))

    dz = os.path.join(run_dir, "design.npz")
    if os.path.exists(dz):
        made.append(plot_eps(np.load(dz)["eps"],
                             os.path.join(out_dir, "design_eps.png"),
                             title="healed ε (design region optimized)",
                             pdf=pdf))
    return made
