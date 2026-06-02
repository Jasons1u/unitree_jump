"""Visualize a reference trajectory NPZ file in MuJoCo before training.

Usage:
    python scripts/visualize_traj.py src/assets/motions/g1/srb_squat.npz
    python scripts/visualize_traj.py path/to/motion.npz --robot g1 --speed 0.5 --alpha 0.6
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a trajectory NPZ file in MuJoCo.")
    parser.add_argument("motion_file", type=str, help="Path to .npz trajectory file")
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        choices=list(SCENE_XMLS.keys()),
        help="Robot model to use (default: g1)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Robot transparency: 0=invisible, 1=opaque (default: 0.7)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Stop after one playthrough instead of looping",
    )
    args = parser.parse_args()

    motion_path = Path(args.motion_file)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    data_np = np.load(motion_path)
    fps: float = float(data_np["fps"][0])
    joint_pos: np.ndarray = data_np["joint_pos"]      # (N, num_joints)
    body_pos_w: np.ndarray = data_np["body_pos_w"]    # (N, num_bodies, 3)
    body_quat_w: np.ndarray = data_np["body_quat_w"]  # (N, num_bodies, 4) wxyz

    num_frames = joint_pos.shape[0]
    num_joints = joint_pos.shape[1]
    duration = num_frames / fps
    print(f"[traj] {motion_path.name}: {num_frames} frames @ {fps:.0f} FPS = {duration:.2f}s, {num_joints} joints")

    xml_path = SCENE_XMLS[args.robot].resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"Scene XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mj_data = mujoco.MjData(model)

    # Apply transparency to all geoms.
    if args.alpha < 1.0:
        model.geom_rgba[:, 3] = args.alpha

    # Verify qpos has room for free joint (7) + actuated joints.
    expected_nq = 7 + num_joints
    if model.nq != expected_nq:
        print(
            f"[warn] model.nq={model.nq} but trajectory has {num_joints} joints "
            f"(expected nq={expected_nq}). Joint mapping may be wrong."
        )

    def set_frame(frame: int) -> None:
        qpos = np.zeros(model.nq)
        qpos[0:3] = body_pos_w[frame, 0]            # root xyz
        qpos[3:7] = body_quat_w[frame, 0]           # root quat (wxyz)
        qpos[7 : 7 + num_joints] = joint_pos[frame] # actuated joints
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(model, mj_data)

    set_frame(0)

    paused = False

    def key_callback(key: int) -> None:
        nonlocal paused
        # Space bar = 32
        if key == 32:
            paused = not paused
            print("[traj] " + ("paused" if paused else "resumed"))

    print("[traj] Controls: Space = pause/resume | Right-click drag = camera | Scroll = zoom")
    print("[traj] Press Ctrl+C or close window to exit")

    with mujoco.viewer.launch_passive(model, mj_data, key_callback=key_callback) as viewer:
        viewer.cam.azimuth = -140.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance = 3.5
        viewer.cam.lookat[:] = body_pos_w[0, 0]  # look at root

        sim_time = 0.0
        last_wall = time.time()
        frame_dt = 1.0 / fps

        while viewer.is_running():
            step_start = time.time()

            if not paused:
                now = time.time()
                sim_time += (now - last_wall) * args.speed
                last_wall = now

                frame_idx = int(sim_time * fps)

                if frame_idx >= num_frames:
                    if args.no_loop:
                        break
                    sim_time = 0.0
                    frame_idx = 0
                    print("[traj] looping")

                set_frame(frame_idx)
            else:
                last_wall = time.time()

            viewer.sync()

            # Sleep to match real-time fps.
            elapsed = time.time() - step_start
            remaining = frame_dt / args.speed - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
