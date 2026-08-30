"""viz — render figures from run-directory snapshots.

Design rule: plots are DERIVED artifacts. Every figure is rendered from the
JSON/npy results a run already wrote (never from live simulation state), so
any figure can be re-rendered months later from the run dir alone — the same
reproducibility contract as everything else in runs/.

Usage:
    python -m invdx.viz runs/<run-dir>          # auto-detect & render all
    python -m invdx.viz runs/<run-dir> -o out/  # render elsewhere

Chart conventions (from the dataviz method): one axis per chart, categorical
colors in fixed order (blue/orange/aqua — a validated CVD-safe palette),
sequential data on a single-hue ramp, thin marks, recessive grid, text in
ink colors (never series colors).
"""

from .plots import render_run  # noqa: F401
