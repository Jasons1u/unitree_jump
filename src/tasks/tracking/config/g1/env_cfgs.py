"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import BoxFlatTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg

from src.tasks.tracking.terrains import BoxTiltedPlaneTerrainCfg

import src.tasks.tracking.mdp as local_mdp
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["contact_material"].params["asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["base_mass"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


##################################################################
# Agility — soft mat terrain + relaxed terminations
##################################################################

def unitree_g1_agility_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Tracking config for dynamic aerial motions (e.g. backflip) on a soft mat.

  Key differences from the flat config:
  - Terrain: mix of flat (30%) and slightly tilted (70%, ≤5°) patches to
    simulate heel/toe sinking on a foam mat.
  - Terminations: ``anchor_ori`` and ``ee_body_pos`` are removed; the root
    height threshold is relaxed to 0.4 m to survive the aerial/inverted phase.
  - Push: disabled so random velocity impulses don't destroy backflip attempts.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # Terrain generator re-enable after confirming DR events are stable.
  # --- Terminations ---
  # Drop orientation and end-effector checks — both fire during a backflip.
  # cfg.terminations.pop("anchor_ori", None)
  # cfg.terminations.pop("ee_body_pos", None)
  # Loosen root height threshold: 25 cm is too tight for the aerial phase.
  cfg.terminations["anchor_pos"].params["threshold"] = 0.2

  # --- Events ---
  # Disable push entirely for now — isolating Warp 710 error.
  cfg.events.pop("push_robot", None)

  return cfg


##################################################################
# CUSTOM W/ PELVIS & NO STATE ESTIMATION
##################################################################

def unitree_g1_pelvis_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:

  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False,
    play=play,
  )

  # Use pelvis as the anchor instead of torso.
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "pelvis"

  return cfg
