"""Inspect which part of a trajectory a given adaptive-sampling bin covers.

Prints the frame range, time range, and root-z heights, then opens the MuJoCo
viewer frozen at the bin's start frame. Use arrow keys to step through frames
within the bin.

Usage:
    python scripts/inspect_bin.py src/assets/motions/g1/srb_backflip.npz 2
    python scripts/inspect_bin.py path/to/motion.npz 1 --robot g1 --step-dt 0.02
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE_XMLS: dict[str, Path] = {
    "g1": Path(__file__).parent / "../src/assets/robots/unitree_g1/xmls/scene_g1.xml",
    "g1_23dof": Path(__file__).parent / "../src/assets/robots/unitree_g1/xmls/scene_g1_23dof.xml",
    "a2": Path(__file__).parent / "../src/assets/robots/unitree_a2/xmls/scene_a2.xml",
    "go2": Path(__file__).parent / "../src/assets/robots/unitree_go2/xmls/scene_go2.xml",
    "h1_2": Path(__file__).parent / "../src/assets/robots/unitree_h1_2/xmls/scene_h1_2.xml",
}

BAR_WIDTH = 60


def ascii_bar(value: float, min_val: float, max_val: float) -> str:
    span = max_val - min_val or 1.0
    filled = int((value - min_val) / span * BAR_WIDTH)
    return "[" + "#" * filled + " " * (BAR_WIDTH - filled) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an adaptive-sampling bin in a trajectory.")
    parser.add_argument("motion_file", type=str, help="Path to .npz trajectory file")
    parser.add_argument("bin", type=str, help="Bin index (int) or normalized sampling_top1_bin metric (float 0-1)")
    parser.add_argument("--robot", type=str, default="g1", choices=list(SCENE_XMLS.keys()))
    parser.add_argument(
        "--step-dt",
        type=float,
        default=0.02,
        help="Env step_dt = decimation * sim_dt (default: 0.02 = 4 * 0.005)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Only print info, skip MuJoCo viewer",
    )
    args = parser.parse_args()

    data = np.load(args.motion_file)
    fps: float = float(data["fps"][0])
    joint_pos: np.ndarray = data["joint_pos"]
    body_pos_w: np.ndarray = data["body_pos_w"]
    body_quat_w: np.ndarray = data["body_quat_w"]

    total_frames = joint_pos.shape[0]
    num_joints = joint_pos.shape[1]

    # Replicate the exact bin_count formula from MotionCommand.__init__
    bin_count = int(total_frames // (1.0 / args.step_dt)) + 1

    raw = args.bin
    if "." in raw:
        bin_idx = round(float(raw) * bin_count)
        print(f"[bin] {raw} * {bin_count} bins = bin {bin_idx}")
    else:
        bin_idx = int(raw)
    args.bin = bin_idx

    if args.bin < 0 or args.bin >= bin_count:
        raise ValueError(f"Bin {args.bin} out of range — motion has {bin_count} bins (0..{bin_count-1})")

    # Frame range for this bin (matches the sampling formula in _adaptive_sampling)
    frame_start = int(args.bin * (total_frames - 1) / bin_count)
    frame_end = int((args.bin + 1) * (total_frames - 1) / bin_count)
    frame_end = min(frame_end, total_frames - 1)

    time_start = frame_start / fps
    time_end = frame_end / fps

    root_z = body_pos_w[frame_start : frame_end + 1, 0, 2]
    z_min, z_max = float(root_z.min()), float(root_z.max())
    global_z_min = float(body_pos_w[:, 0, 2].min())
    global_z_max = float(body_pos_w[:, 0, 2].max())

    print(f"\nMotion : {Path(args.motion_file).name}")
    print(f"Frames : {total_frames} @ {fps:.0f} FPS = {total_frames/fps:.2f}s total")
    print(f"Bins   : {bin_count}  (step_dt={args.step_dt}s)")
    print(f"\n--- Bin {args.bin} ---")
    print(f"Frames : {frame_start} – {frame_end}  ({frame_end - frame_start + 1} frames)")
    print(f"Time   : {time_start:.3f}s – {time_end:.3f}s")
    print(f"Root Z : min={z_min:.3f}m  max={z_max:.3f}m")
    print(f"Flight : {'YES — robot leaves ground' if z_max > global_z_min + 0.1 else 'no'}\n")

    print("Root Z over bin frames (global scale):")
    for i, (frame, z) in enumerate(zip(range(frame_start, frame_end + 1), root_z)):
        marker = " <-- bin start" if i == 0 else (" <-- bin end" if i == len(root_z) - 1 else "")
        print(f"  f{frame:03d} t={frame/fps:.2f}s z={z:.3f} {ascii_bar(z, global_z_min, global_z_max)}{marker}")

    if args.no_viewer:
        return

    xml_path = SCENE_XMLS[args.robot].resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(model)

    mid_frame = (frame_start + frame_end) // 2
    current_frame = [mid_frame]

    def set_frame(f: int) -> None:
        qpos = np.zeros(model.nq)
        qpos[0:3] = body_pos_w[f, 0]
        qpos[3:7] = body_quat_w[f, 0]
        qpos[7 : 7 + num_joints] = joint_pos[f]
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(model, mj_data)

    def key_callback(key: int) -> None:
        # Right arrow = 262, Left arrow = 263
        if key == 262:
            current_frame[0] = min(current_frame[0] + 1, frame_end)
        elif key == 263:
            current_frame[0] = max(current_frame[0] - 1, frame_start)
        f = current_frame[0]
        print(f"  frame {f}  t={f/fps:.3f}s  z={body_pos_w[f, 0, 2]:.3f}m")

    set_frame(mid_frame)

    print(f"\n[viewer] Showing bin {args.bin} (frames {frame_start}–{frame_end}), starting at midpoint f{mid_frame}")
    print("[viewer] Left/Right arrows to step through bin frames | Ctrl+C or close to exit")

    with mujoco.viewer.launch_passive(model, mj_data, key_callback=key_callback) as viewer:
        viewer.cam.azimuth = -140.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance = 3.5
        viewer.cam.lookat[:] = body_pos_w[mid_frame, 0]

        frame_dt = 1.0 / fps
        while viewer.is_running():
            set_frame(current_frame[0])
            viewer.sync()
            time.sleep(frame_dt)


if __name__ == "__main__":
    main()
