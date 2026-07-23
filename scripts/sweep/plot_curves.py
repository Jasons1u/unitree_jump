#!/usr/bin/env python
"""Overlay seed-swept training curves, one line per motion.

Reads the per-run CSVs produced by pull_curves.py, groups them by motion
(via run-name prefix), and plots the mean curve per motion with a translucent
+/- 1 std band across that motion's seeds.

The number of seeds per motion is NOT hardcoded -- every CSV found for a group
is used. Within a group, seeds are truncated to the shortest run's step range
before computing mean/std (so all seeds contribute over the shared range).

Usage:
    python scripts/sweep/plot_curves.py
    python scripts/sweep/plot_curves.py --metric Metrics/motion/error_body_pos
    python scripts/sweep/plot_curves.py --indir logs/sweep/curves --out logs/sweep/curves/plot.png
    python scripts/sweep/plot_curves.py --smooth 20
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import seaborn as sns

# Motion groups: (prefix, human-readable label). Mirrors seed_sweep.sh.
GROUPS = [
    ("traj_opt_kino_ablation_", "SRB-KD-TO"),
    ("kino_backflip_ablation_", "SRB-KD"),
    ("srb_ik_backflip_ablation_", "SRB"),
    ("srb_traj_backflip_ablation_", "SRB-TO"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="logs/sweep/curves", help="Directory of per-run CSVs (default: logs/sweep/curves).")
    p.add_argument("--metric", default="Train/mean_reward", help="Top-subplot metric (default: Train/mean_reward).")
    p.add_argument("--metric-std", default="Policy/mean_std", help="Bottom-subplot metric (default: Policy/mean_std).")
    p.add_argument("--out", default=None, help="Output image path (default: <indir>/<metric>.svg).")
    p.add_argument("--no-show", action="store_true", help="Don't open an interactive window; just save the file.")
    p.add_argument("--smooth", type=int, default=0, help="Moving-average window (in samples) for smoothing. 0 = off.")
    p.add_argument(
        "--xaxis",
        choices=["wall", "step"],
        default="wall",
        help="X-axis: wall=relative wall-clock time in minutes (default), step=iteration.",
    )
    p.add_argument(
        "--max-steps",
        type=float,
        default=10000,
        help="Only plot iterations <= this value (default: 10000). Use a large value or 0 to disable. "
        "Applies to the step count regardless of --xaxis.",
    )
    p.add_argument("--title", default=None, help="Plot title (default: the metric name).")
    p.add_argument(
        "--figsize",
        default="10,6",
        help="Figure size in inches as 'W,H' (default: 10,6 — two stacked subplots).",
    )
    p.add_argument("--no-latex", action="store_true", help="Disable LaTeX text rendering (use if no LaTeX installed).")
    p.add_argument(
        "--agg",
        choices=["median", "mean"],
        default="median",
        help="Center line across seeds (default: median, robust to a lagging/outlier seed).",
    )
    p.add_argument(
        "--band",
        choices=["iqr", "std", "sem", "minmax", "none"],
        default="iqr",
        help="Shaded band across seeds: iqr=25-75pct (default), std=+/-1 std, "
        "sem=+/-std/sqrt(n), minmax=full range, none=no band.",
    )
    return p.parse_args()


def aggregate(stacked, agg, band):
    """Return (center, low, high) across seeds (axis 0) for the chosen statistics."""
    if agg == "median":
        center = np.median(stacked, axis=0)
    else:
        center = stacked.mean(axis=0)

    n = stacked.shape[0]
    if band == "none" or n < 2:
        return center, center, center
    if band == "iqr":
        low = np.percentile(stacked, 25, axis=0)
        high = np.percentile(stacked, 75, axis=0)
    elif band == "std":
        s = stacked.std(axis=0)
        low, high = center - s, center + s
    elif band == "sem":
        s = stacked.std(axis=0) / np.sqrt(n)
        low, high = center - s, center + s
    elif band == "minmax":
        low = stacked.min(axis=0)
        high = stacked.max(axis=0)
    return center, low, high


def read_csv(path, metric, xaxis):
    """Return (x, steps, values) as float arrays for `metric`, dropping blank rows.

    `x` is the plotting axis (wall-clock minutes if xaxis="wall", else iteration).
    `steps` is always the iteration count, used for --max-steps filtering.
    """
    xs, steps, vals = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            return None  # metric absent in this run
        if xaxis == "wall" and "wall_time" not in reader.fieldnames:
            return None  # wall time not pulled for this run
        for row in reader:
            s, v = row.get("step", ""), row.get(metric, "")
            w = row.get("wall_time", "")
            if s == "" or v == "":
                continue
            if xaxis == "wall" and w == "":
                continue
            try:
                step = float(s)
                val = float(v)
                x = float(w) / 60.0 if xaxis == "wall" else step
            except ValueError:
                continue
            xs.append(x)
            steps.append(step)
            vals.append(val)
    if not steps:
        return None
    return np.asarray(xs), np.asarray(steps), np.asarray(vals)


def moving_average(y, window):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def tex_escape(s):
    """Escape characters that are special in LaTeX so arbitrary labels render."""
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def setup_style(use_latex):
    """Professional look: clean white background, serif LaTeX text.

    Returns whether LaTeX is actually active (falls back to mathtext if the
    system has no LaTeX installation).
    """
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.4)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "0.3",
        "axes.linewidth": 1.0,
    })

    def use_cm_fonts():
        """Computer Modern look via matplotlib's bundled cmr10 (no LaTeX toolchain)."""
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,  # cmr10 lacks the unicode minus glyph
        })

    if not use_latex:
        use_cm_fonts()
        return False
    # Probe for a working LaTeX toolchain before committing to it. matplotlib's
    # usetex preamble needs the latex/dvipng binaries plus the type1cm/type1ec
    # fonts (Ubuntu: the `cm-super` and `dvipng` packages).
    missing = [b for b in ("latex", "dvipng") if shutil.which(b) is None]
    if not missing and shutil.which("kpsewhich"):
        for sty in ("type1ec.sty", "type1cm.sty"):
            if subprocess.run(["kpsewhich", sty], capture_output=True).returncode != 0:
                missing.append(sty)
    if missing:
        print(
            "[plot] LaTeX toolchain incomplete (missing: "
            + ", ".join(missing)
            + "); using matplotlib's bundled Computer Modern font instead.\n"
            "       For true usetex rendering: sudo apt-get install cm-super dvipng",
            file=sys.stderr,
        )
        use_cm_fonts()
        return False
    plt.rcParams.update({
        "text.usetex": True,
        "font.serif": ["Computer Modern Roman"],
    })
    return True


def load_group(indir, prefix, metric, xaxis, max_steps=0):
    """Load all seeds for a group. Returns (x, stacked_values[n_seeds, n_steps], names)."""
    paths = sorted(glob.glob(os.path.join(indir, f"{prefix}*.csv")))
    seeds, names = [], []
    for path in paths:
        got = read_csv(path, metric, xaxis)
        if got is None:
            print(f"  [warn] {os.path.basename(path)}: no usable '{metric}' data, skipping", file=sys.stderr)
            continue
        xs, steps, vals = got
        if max_steps and max_steps > 0:
            mask = steps <= max_steps
            xs, vals = xs[mask], vals[mask]
            if len(xs) == 0:
                continue
        seeds.append((xs, vals))
        names.append(os.path.splitext(os.path.basename(path))[0])
    if not seeds:
        return None

    # Truncate to the shortest run's length so all seeds share an x grid.
    min_len = min(len(s[0]) for s in seeds)
    # Average the x-axis across seeds (wall time varies per seed; iteration is identical).
    x = np.mean(np.vstack([xs[:min_len] for (xs, _) in seeds]), axis=0)
    stacked = np.vstack([v[:min_len] for (_, v) in seeds])
    return x, stacked, names


def plot_metric(ax, args, metric, use_latex, esc, palette):
    """Draw all motion groups' curves for `metric` onto `ax`. Returns count plotted."""
    plotted = 0
    for (prefix, label), color in zip(GROUPS, palette):
        loaded = load_group(args.indir, prefix, metric, args.xaxis, args.max_steps)
        if loaded is None:
            print(f"[plot] {label}: no CSVs with '{metric}', skipping")
            continue
        x, stacked, names = loaded
        n = stacked.shape[0]

        center, low, high = aggregate(stacked, args.agg, args.band)
        if args.smooth > 1:
            center = moving_average(center, args.smooth)
            low = moving_average(low, args.smooth)
            high = moving_average(high, args.smooth)

        disp = label.replace("_", " ")  # underscores render as accents in cmr10
        ax.plot(x, center, color=color, label=disp, linewidth=2)
        if args.band != "none" and n >= 2:
            ax.fill_between(x, low, high, color=color, alpha=0.2, linewidth=0)
        print(f"[plot] {metric} / {label}: {n} seeds, {len(x)} samples -> {', '.join(names)}")
        plotted += 1
    return plotted


def main():
    args = parse_args()
    if not os.path.isdir(args.indir):
        print(f"[plot] input dir not found: {args.indir} (run pull_curves.py first)", file=sys.stderr)
        sys.exit(1)

    use_latex = setup_style(not args.no_latex)
    esc = tex_escape if use_latex else (lambda s: s)
    palette = ["#D81B60", "#1E88E5", "#FFC107", "#004D40"]

    figsize = tuple(float(v) for v in args.figsize.split(","))
    # Two stacked subplots sharing the x-axis: reward on top, policy std below.
    fig, (ax_rew, ax_std) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    plotted = plot_metric(ax_rew, args, args.metric, use_latex, esc, palette)
    plotted += plot_metric(ax_std, args, args.metric_std, use_latex, esc, palette)

    if plotted == 0:
        print("[plot] nothing to plot.", file=sys.stderr)
        sys.exit(1)

    xlabel = "wall-clock time (min)" if args.xaxis == "wall" else "Training Iterations"
    ax_rew.set_ylabel("Avg. Reward, $r$")
    ax_std.set_ylabel(r"Avg. Std, $\sigma$")
    ax_std.set_xlabel(esc(xlabel))
    # sharex=True links both axes, so one set_xlim covers both.
    ax_std.set_xlim(0, 7e3)

    # Slightly denser y axis: a few more labeled major ticks than the default.
    for ax in (ax_rew, ax_std):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    legend = ax_std.legend(
        loc="best", frameon=True, fontsize=20, handlelength=1.5,
        ncol=len(GROUPS), columnspacing=1.2, handletextpad=0.5,
        fancybox=False, shadow=False, edgecolor="0.3", framealpha=1.0,
    )
    legend.get_frame().set_linewidth(0.8)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.align_ylabels([ax_rew, ax_std])
    fig.suptitle(esc(args.title or "Ablation 1: Reduced Order Model TO"), fontsize="medium", y=0.98)
    fig.subplots_adjust(top=0.93)

    out = args.out or os.path.join(args.indir, args.metric.replace("/", "_") + ".svg")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] saved -> {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
