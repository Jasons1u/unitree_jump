"""Extract a raw trajectory-opt NPZ (state/input/reference format) into the CSV
format consumed by csv_to_npz.py.

The source `state` rows are MuJoCo [pos(3), quat wxyz(4), dof(N), qvel(...)].
csv_to_npz.py expects CSV columns [pos(3), quat xyzw(4), dof(N)], so we slice
qpos and reorder the quaternion wxyz -> xyzw.

Usage:
    python scripts/npz_to_csv.py src/assets/motions/g1/sbo_jump_up.npz \
        src/assets/motions/g1/sbo_jump_up.csv --num-dof 29
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", type=str, help="Source .npz with a 'state' array")
    parser.add_argument("output_csv", type=str, help="Destination .csv path")
    parser.add_argument("--num-dof", type=int, default=29, help="Number of actuated joints")
    parser.add_argument(
        "--key", type=str, default="state", help="NPZ key to read (state or reference)"
    )
    args = parser.parse_args()

    data = np.load(args.input_npz)
    state = np.asarray(data[args.key], dtype=np.float64)

    pos = state[:, 0:3]
    quat_wxyz = state[:, 3:7]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof = state[:, 7 : 7 + args.num_dof]

    qpos = np.concatenate([pos, quat_xyzw, dof], axis=1)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, qpos, delimiter=",")

    print(f"Wrote {qpos.shape[0]} frames x {qpos.shape[1]} cols -> {out}")
    print(f"  cols = pos(3) + quat_xyzw(4) + dof({args.num_dof})")
    print(f"  first quat xyzw: {np.round(quat_xyzw[0], 4)}")


if __name__ == "__main__":
    main()
