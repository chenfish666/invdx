"""Paper-numbers report: format a verification run's results.json into a
Markdown table (the numbers section of the paper, machine-generated so the
manuscript can never drift from the measured data).

    python -m invdx.report runs/<verify-run-dir> [...]

With several run dirs, emits one combined comparison table — exactly the
efficiency/linewidth/robustness tradeoff table the paper needs.
"""

import argparse
import json
import os


def _row(res):
    lw = res.get("linewidth", {})
    pk = res.get("peak", {})
    bw = res.get("bandwidth_3db", {})
    band = [r["CE_dB"] for r in res.get("spectrum", [])]
    r = {
        "design": os.path.basename(res.get("source_run", "?"))
                  + ("/" + res["rho_name"] if res.get("rho_name",
                     "design_rho.npy") != "design_rho.npy" else ""),
        "peak CE (dB)": f"{pk.get('CE_dB', float('nan')):.2f}",
        "λ_peak (µm)": f"{pk.get('lam_um', float('nan')):.3f}",
        "3dB BW (nm)": (f"{bw['bw_nm']:.0f}" + ("*" if bw.get("note") else "")
                        ) if bw else "—",
        "min solid/void (nm)": (f"{lw['min_solid_um']*1e3:.0f}/"
                                f"{lw['min_void_um']*1e3:.0f}"
                                ) if lw else "—",
        "band mean (dB)": f"{sum(band)/len(band):.2f}" if band else "—",
    }
    if "s11" in res:
        r["S11 (dB)"] = f"{res['s11']['S11_dB']:.2f}"
        r["recip. Δ (dB)"] = f"{res['s11']['reciprocity_mismatch_dB']:.2f}"
    corners = res.get("corners") or {}
    for name, c in corners.items():
        r[f"{name} peak (dB)"] = f"{c['peak']['CE_dB']:.2f}"
    return r


def markdown_table(rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    lines = ["| " + " | ".join(keys) + " |",
             "|" + "|".join("---" for _ in keys) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r.get(k, "—") for k in keys) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("-o", "--out", default=None,
                   help="write the table to this file as well")
    args = p.parse_args()
    rows = []
    for d in args.run_dirs:
        with open(os.path.join(d, "results.json")) as f:
            rows.append(_row(json.load(f)))
    table = markdown_table(rows)
    footer = ("\n\n\\* bandwidth clipped by the sampled wavelength range — "
              "value is a lower bound")
    if any("*" in r.get("3dB BW (nm)", "") for r in rows):
        table += footer
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\n[report] written to {args.out}")


if __name__ == "__main__":
    main()
