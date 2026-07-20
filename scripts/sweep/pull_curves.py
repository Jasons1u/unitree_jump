#!/usr/bin/env python
"""Pull training curves for the seed-sweep runs from W&B to local CSVs.

For each matching run, writes one wide-format CSV (one row per logged step,
one column per metric) to the output directory. These are ready to load with
pandas/numpy/matplotlib for plotting.

Runs are matched by W&B display-name prefix. By default it grabs the three
ablation groups produced by seed_sweep.sh:

    traj_opt_kino_ablation_*
    kino_backflip_ablation_*
    srb_ik_backflip_ablation_*

Usage:
    python scripts/sweep/pull_curves.py                      # all sweep runs -> logs/sweep/curves/
    python scripts/sweep/pull_curves.py --outdir /tmp/curves
    python scripts/sweep/pull_curves.py --prefix kino_backflip_ablation_
    python scripts/sweep/pull_curves.py --keys Train/mean_reward,Train/mean_episode_length
    python scripts/sweep/pull_curves.py --list             # just list matching runs, don't download
"""

import argparse
import csv
import os
import sys

import wandb

DEFAULT_PREFIXES = [
    "traj_opt_kino_ablation_",
    "kino_backflip_ablation_",
    "srb_ik_backflip_ablation_",
]

# Step axis logged by rsl_rl. `_step` is always present as a fallback.
STEP_KEYS = ("_step",)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default=None, help="W&B entity (default: your default entity).")
    p.add_argument("--project", default="mjlab_ablation", help="W&B project (default: mjlab_ablation).")
    p.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Run-name prefix to match. Repeatable. Defaults to the three ablation groups.",
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
    return p.parse_args()


def matching_runs(api, path, prefixes):
    runs = api.runs(path)
    out = []
    for r in runs:
        if any(r.name.startswith(pre) for pre in prefixes):
            out.append(r)
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
    header = ["step"] + keys
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            step = row.get("_step", "")
            w.writerow([step] + [row.get(k, "") for k in keys])
    return len(rows)


def main():
    args = parse_args()
    prefixes = args.prefix or DEFAULT_PREFIXES

    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = f"{entity}/{args.project}"
    print(f"[pull-curves] project: {path}")
    print(f"[pull-curves] prefixes: {', '.join(prefixes)}")

    runs = matching_runs(api, path, prefixes)
    if not runs:
        print("[pull-curves] no matching runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"[pull-curves] {len(runs)} matching runs:")
    for r in runs:
        print(f"    {r.name:<32} {r.state:<10} id={r.id}")

    if args.list:
        return

    os.makedirs(args.outdir, exist_ok=True)
    for r in runs:
        out_path = os.path.join(args.outdir, f"{r.name}.csv")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[pull-curves] skip (exists): {out_path}  (use --overwrite to refresh)")
            continue
        keys = metric_keys(r, args.keys)
        print(f"[pull-curves] downloading {r.name} ({len(keys)} metrics) ...", flush=True)
        n = pull_run(r, keys, out_path)
        print(f"[pull-curves]   -> {out_path}  ({n} steps)")

    print(f"[pull-curves] done. CSVs in: {args.outdir}")


if __name__ == "__main__":
    main()
