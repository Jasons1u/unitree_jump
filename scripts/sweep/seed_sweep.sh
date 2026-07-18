#!/usr/bin/env bash
#
# Seed sweep for the G1 tracking ablation.
#
# Runs train.py once per (motion, seed) with --randomize-seed, targeting 5 total
# runs per motion. Runs that already exist on wandb are skipped (see table below),
# so this script only launches the remaining ones.
#
#   traj_opt_kino_backflip.npz -> traj_opt_kino_ablation_{2,3,4}   (0,1 already done)
#   kino_backflip.npz          -> kino_backflip_ablation_{1,2,3,4} (0 already done)
#   srb_backflip.npz           -> srb_ik_backflip_ablation_{1,2,3,4} (0 already done)
#
# Usage:
#   scripts/seed_sweep.sh                 # run the full remaining sweep sequentially (GPU 0)
#   scripts/seed_sweep.sh --gpu 1         # run on GPU 1 (passes --gpu-ids '[1]')
#   scripts/seed_sweep.sh --dry-run       # print the commands without running them
#   scripts/seed_sweep.sh --gpu 1 --dry-run
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK="Unitree-G1-Tracking-Ablation"
NUM_ENVS=4096
MAX_ITER=10000
MOTION_DIR="src/assets/motions/g1"

REPO_DIR="~/humanoid_ws/unitree_jump"
CONDA_ENV="unitree_rl_mjlab"

DRY_RUN=0
GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --gpu) GPU="$2"; shift 2 ;;
    --gpu=*) GPU="${1#*=}"; shift ;;
    *) echo "[seed-sweep] unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Job list: "<motion_file> <run_name>"
# ---------------------------------------------------------------------------
JOBS=(
  "traj_opt_kino_backflip.npz traj_opt_kino_ablation_2"
  "traj_opt_kino_backflip.npz traj_opt_kino_ablation_3"
  "traj_opt_kino_backflip.npz traj_opt_kino_ablation_4"

  "kino_backflip.npz kino_backflip_ablation_1"
  "kino_backflip.npz kino_backflip_ablation_2"
  "kino_backflip.npz kino_backflip_ablation_3"
  "kino_backflip.npz kino_backflip_ablation_4"

  "srb_backflip.npz srb_ik_backflip_ablation_1"
  "srb_backflip.npz srb_ik_backflip_ablation_2"
  "srb_backflip.npz srb_ik_backflip_ablation_3"
  "srb_backflip.npz srb_ik_backflip_ablation_4"
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cd "$REPO_DIR"

if [[ $DRY_RUN -eq 0 ]]; then
  # shellcheck disable=SC1091
  source /home/jason/anaconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

TOTAL=${#JOBS[@]}
echo "[seed-sweep] $TOTAL runs to launch (gpu=$GPU, dry_run=$DRY_RUN)"
echo

i=0
for job in "${JOBS[@]}"; do
  i=$((i + 1))
  read -r motion run_name <<< "$job"
  motion_path="${MOTION_DIR}/${motion}"

  if [[ ! -f "$motion_path" ]]; then
    echo "[seed-sweep] ($i/$TOTAL) SKIP: motion file not found: $motion_path" >&2
    continue
  fi

  echo "==============================================================="
  echo "[seed-sweep] ($i/$TOTAL) run_name=$run_name  motion=$motion"
  echo "==============================================================="

  cmd=(python scripts/train.py "$TASK"
    --motion_file="$motion_path"
    --env.scene.num-envs="$NUM_ENVS"
    --randomize-seed True
    --agent.max-iterations "$MAX_ITER"
    --agent.run-name "$run_name"
    --gpu-ids "[$GPU]")

  if [[ $DRY_RUN -eq 1 ]]; then
    printf '  %q' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
    echo "[seed-sweep] ($i/$TOTAL) done: $run_name"
  fi
  echo
done

echo "[seed-sweep] all $TOTAL runs complete."
