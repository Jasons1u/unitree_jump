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
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Motion groups: (prefix, human-readable label). Mirrors seed_sweep.sh.
GROUPS = [
    ("traj_opt_kino_ablation_", "traj_opt_kino"),
    ("kino_backflip_ablation_", "kino_backflip"),
    ("srb_ik_backflip_ablation_", "srb_ik_backflip"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", default="logs/sweep/curves", help="Directory of per-run CSVs (default: logs/sweep/curves).")
    p.add_argument("--metric", default="Train/mean_reward", help="Metric column to plot (default: Train/mean_reward).")
    p.add_argument("--out", default=None, help="Output image path (default: <indir>/<metric>.png).")
    p.add_argument("--smooth", type=int, default=0, help="Moving-average window (in steps) for smoothing. 0 = off.")
    p.add_argument(
        "--max-steps",
        type=float,
        default=10000,
        help="Only plot steps <= this value (default: 10000). Use a large value or 0 to disable.",
    )
    p.add_argument("--title", default=None, help="Plot title (default: the metric name).")
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


def read_csv(path, metric):
    """Return (steps, values) as float arrays for `metric`, dropping blank rows."""
    steps, vals = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            return None  # metric absent in this run
        for row in reader:
            s, v = row.get("step", ""), row.get(metric, "")
            if s == "" or v == "":
                continue
            try:
                steps.append(float(s))
                vals.append(float(v))
            except ValueError:
                continue
    if not steps:
        return None
    return np.asarray(steps), np.asarray(vals)


def moving_average(y, window):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def load_group(indir, prefix, metric, max_steps=0):
    """Load all seeds for a group. Returns (steps, stacked_values[n_seeds, n_steps], names)."""
    paths = sorted(glob.glob(os.path.join(indir, f"{prefix}*.csv")))
    seeds, names = [], []
    for path in paths:
        got = read_csv(path, metric)
        if got is None:
            print(f"  [warn] {os.path.basename(path)}: no usable '{metric}' data, skipping", file=sys.stderr)
            continue
        steps, vals = got
        if max_steps and max_steps > 0:
            mask = steps <= max_steps
            steps, vals = steps[mask], vals[mask]
            if len(steps) == 0:
                continue
        seeds.append((steps, vals))
        names.append(os.path.splitext(os.path.basename(path))[0])
    if not seeds:
        return None

    # Truncate to the shortest run's length so all seeds share a step grid.
    min_len = min(len(s[0]) for s in seeds)
    steps = seeds[0][0][:min_len]
    stacked = np.vstack([v[:min_len] for (_, v) in seeds])
    return steps, stacked, names


def main():
    args = parse_args()
    if not os.path.isdir(args.indir):
        print(f"[plot] input dir not found: {args.indir} (run pull_curves.py first)", file=sys.stderr)
        sys.exit(1)

    sns.set_theme(style="darkgrid", context="talk")
    palette = sns.color_palette("colorblind", n_colors=len(GROUPS))

    fig, ax = plt.subplots(figsize=(11, 7))
    plotted = 0

    for (prefix, label), color in zip(GROUPS, palette):
        loaded = load_group(args.indir, prefix, args.metric, args.max_steps)
        if loaded is None:
            print(f"[plot] {label}: no CSVs matching '{prefix}*', skipping")
            continue
        steps, stacked, names = loaded
        n = stacked.shape[0]

        center, low, high = aggregate(stacked, args.agg, args.band)
        if args.smooth > 1:
            center = moving_average(center, args.smooth)
            low = moving_average(low, args.smooth)
            high = moving_average(high, args.smooth)

        ax.plot(steps, center, color=color, label=f"{label} (n={n})", linewidth=2)
        if args.band != "none" and n >= 2:
            ax.fill_between(steps, low, high, color=color, alpha=0.2, linewidth=0)
        print(f"[plot] {label}: {n} seeds, {len(steps)} steps -> {', '.join(names)}")
        plotted += 1

    if plotted == 0:
        print(f"[plot] nothing to plot for metric '{args.metric}'.", file=sys.stderr)
        sys.exit(1)

    ax.set_xlabel("iteration")
    ax.set_ylabel(args.metric)
    ax.set_title(args.title or args.metric)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    out = args.out or os.path.join(args.indir, args.metric.replace("/", "_") + ".png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
