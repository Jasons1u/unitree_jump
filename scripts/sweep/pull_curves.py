#!/usr/bin/env python
"""Pull training curves for the seed-sweep runs from W&B to local CSVs.

For each matching run, writes one wide-format CSV (one row per logged step,
one column per metric) to the output directory. These are ready to load with
pandas/numpy/matplotlib for plotting.

Runs are matched by W&B group. By default it grabs every run in the
`ablation` group of the `mjlab` project.

Usage:
    python scripts/sweep/pull_curves.py                      # all ablation-group runs -> logs/sweep/curves/
    python scripts/sweep/pull_curves.py --outdir /tmp/curves
    python scripts/sweep/pull_curves.py --group ablation
    python scripts/sweep/pull_curves.py --keys Train/mean_reward,Train/mean_episode_length
    python scripts/sweep/pull_curves.py --list             # just list matching runs, don't download
"""

import argparse
import csv
import os
import re
import statistics
import sys

import wandb

DEFAULT_GROUP = "ablation"

# Metrics summarized at the target iteration, in report order.
STAT_KEYS = ("Train/mean_reward", "Policy/mean_std", "Train/mean_episode_length")
STAT_LABELS = ("mean_reward", "policy_std", "episode_length")

# Step axis logged by rsl_rl. `_step` is always present as a fallback.
# `_runtime` is relative wall-clock seconds since the run started (the
# actual history key behind W&B's "_relative_time(wall)" UI label).
STEP_KEYS = ("_step", "_runtime")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=None, help="W&B entity (default: your default entity).")
    p.add_argument("--project", default="mjlab", help="W&B project (default: mjlab).")
    p.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=f"W&B group to match (default: {DEFAULT_GROUP}).",
    )
    p.add_argument(
        "--outdir",
        default="logs/sweep/curves",
        help="Directory to write CSVs into (default: logs/sweep/curves).",
    )
    p.add_argument(
        "--keys",
        default="all",
        help="Comma-separated metric keys to pull, or 'all' (default) for every logged metric.",
    )
    p.add_argument("--list", action="store_true", help="List matching runs and exit (no download).")
    p.add_argument("--overwrite", action="store_true", help="Re-download runs whose CSV already exists.")
    p.add_argument(
        "--stat-iter",
        type=float,
        default=7000,
        help="Iteration at which to summarize per-motion stats (default: 7000).",
    )
    p.add_argument("--no-stats", action="store_true", help="Skip the per-motion summary table.")
    return p.parse_args()


def motion_of(run_name):
    """Motion name = run name with the trailing seed suffix (`_<digits>`) removed."""
    return re.sub(r"_\d+$", "", run_name)


def values_at_iter(csv_path, keys, target_iter):
    """For each key, return the value from the row whose `step` is nearest target_iter.

    Each metric is matched independently (nearest non-blank sample), so a metric
    logged on a different cadence still resolves to its closest value.
    Returns (values{key->float|None}, steps{key->float|None}).
    """
    best = {k: None for k in keys}  # key -> (abs_diff, value, step)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        present = [k for k in keys if k in reader.fieldnames]
        for row in reader:
            try:
                step = float(row.get("step", ""))
            except ValueError:
                continue
            diff = abs(step - target_iter)
            for k in present:
                v = row.get(k, "")
                if v == "":
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if best[k] is None or diff < best[k][0]:
                    best[k] = (diff, fv, step)
    values = {k: (best[k][1] if best[k] else None) for k in keys}
    steps = {k: (best[k][2] if best[k] else None) for k in keys}
    return values, steps


def summarize(run_csvs, target_iter, out_path):
    """Aggregate per-motion mean/std across seeds at target_iter; print + write CSV."""
    # motion -> {key: [per-seed values]}
    motions = {}
    for name, csv_path in run_csvs:
        if not os.path.exists(csv_path):
            continue
        vals, _ = values_at_iter(csv_path, STAT_KEYS, target_iter)
        m = motions.setdefault(motion_of(name), {k: [] for k in STAT_KEYS})
        for k in STAT_KEYS:
            if vals[k] is not None:
                m[k].append(vals[k])

    if not motions:
        print("[pull-curves] no data to summarize.", file=sys.stderr)
        return

    print(f"\n[pull-curves] per-motion summary at ~{int(target_iter)} iters (mean +/- std across seeds):")
    header = f"    {'motion':<28} {'n':>3}"
    for lab in STAT_LABELS:
        header += f"  {lab:>22}"
    print(header)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["motion", "n_seeds"]
        for lab in STAT_LABELS:
            cols += [f"{lab}_mean", f"{lab}_std"]
        w.writerow(cols)
        for motion in sorted(motions):
            per = motions[motion]
            n = max(len(per[k]) for k in STAT_KEYS)
            line = f"    {motion:<28} {n:>3}"
            row = [motion, n]
            for k in STAT_KEYS:
                xs = per[k]
                if xs:
                    mean = statistics.fmean(xs)
                    std = statistics.stdev(xs) if len(xs) > 1 else 0.0
                    line += f"  {mean:>10.3f} +/- {std:<7.3f}"
                    row += [f"{mean:.6f}", f"{std:.6f}"]
                else:
                    line += f"  {'n/a':>22}"
                    row += ["", ""]
            print(line)
            w.writerow(row)
    print(f"[pull-curves] summary written -> {out_path}")


def matching_runs(api, path, group):
    runs = api.runs(path, filters={"group": group})
    out = list(runs)
    # Stable, human-friendly ordering.
    out.sort(key=lambda r: r.name)
    return out


def metric_keys(run, requested):
    if requested != "all":
        return [k.strip() for k in requested.split(",") if k.strip()]
    return sorted(k for k in run.summary.keys() if not k.startswith("_"))


def pull_run(run, keys, out_path):
    """Stream full history for `keys` and write a wide CSV. Returns row count."""
    fields = list(STEP_KEYS) + keys
    rows = list(run.scan_history(keys=fields, page_size=10000))
    header = ["step", "wall_time"] + keys
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            step = row.get("_step", "")
            wall = row.get("_runtime", "")
            w.writerow([step, wall] + [row.get(k, "") for k in keys])
    return len(rows)


def main():
    args = parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = f"{entity}/{args.project}"
    print(f"[pull-curves] project: {path}")
    print(f"[pull-curves] group: {args.group}")

    runs = matching_runs(api, path, args.group)
    if not runs:
        print("[pull-curves] no matching runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"[pull-curves] {len(runs)} matching runs:")
    for r in runs:
        print(f"    {r.name:<32} {r.state:<10} id={r.id}")

    if args.list:
        return

    os.makedirs(args.outdir, exist_ok=True)
    run_csvs = []
    for r in runs:
        out_path = os.path.join(args.outdir, f"{r.name}.csv")
        run_csvs.append((r.name, out_path))
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[pull-curves] skip (exists): {out_path}  (use --overwrite to refresh)")
            continue
        keys = metric_keys(r, args.keys)
        print(f"[pull-curves] downloading {r.name} ({len(keys)} metrics) ...", flush=True)
        n = pull_run(r, keys, out_path)
        print(f"[pull-curves]   -> {out_path}  ({n} steps)")

    print(f"[pull-curves] done. CSVs in: {args.outdir}")

    if not args.no_stats:
        summary_path = os.path.join(args.outdir, f"summary_{int(args.stat_iter)}.csv")
        summarize(run_csvs, args.stat_iter, summary_path)


if __name__ == "__main__":
    main()
