"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import BoxFlatTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
  """G1 agility config for dynamic motions (jumps, backflips) on soft mat."""
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # Terrain: mix of flat and ≤5° tilted patches simulating heel/toe sinking on a soft mat.
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    max_init_terrain_level=2,
    terrain_generator=TerrainGeneratorCfg(
      size=(3.0, 3.0),
      num_rows=10,
      num_cols=8,
      curriculum=True,
      difficulty_range=(0, 1.0),
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0),
        "tilted": BoxTiltedPlaneTerrainCfg(proportion=0.0, max_tilt_deg=5.0),
      },
    ),
  )

  # Terminations: drop ori + ee checks (fire during flight); relax height.
  cfg.terminations.pop("anchor_ori", None)
  cfg.terminations.pop("ee_body_pos", None)
  cfg.terminations["anchor_pos"].params["threshold"] = 0.35

  # Curriculum: log mean terrain level, progress based on episode survival.
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(func=local_mdp.terrain_levels_tracking),
  }

  # Height-gated push: skip robots that are airborne.
  # if "push_robot" in cfg.events:
  #   cfg.events["push_robot"].func = local_mdp.push_by_setting_velocity_grounded
  #   cfg.events["push_robot"].params["height_threshold"] = 0.7
  cfg.events.pop("push_robot", None)

  # Penalize asymmetric hip joints:
  #   pitch (Y-axis): symmetric  → sign= +1  (same range both sides)
  #   roll  (X-axis): mirrored ranges (-0.52,2.97) vs (-2.97,0.52) → sign= -1
  #   yaw   (Z-axis): anti-symmetric → sign= -1  (toe-out convention)
  cfg.rewards["hip_symmetry"] = RewardTermCfg(
    func=local_mdp.joint_pair_symmetry_l2,
    weight=-2.0,
    params={
      "left_cfg": SceneEntityCfg("robot", joint_names=(
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
      )),
      "right_cfg": SceneEntityCfg("robot", joint_names=(
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
      )),
      "signs": (1.0, -1.0, -1.0),  # pitch symmetric, roll/yaw anti-symmetric
    },
  )

  # Penalize waist roll — prevents robot from leaning to one side to hop on one leg.
  cfg.rewards["waist_roll"] = RewardTermCfg(
    func=local_mdp.joint_pos_l2,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_roll_joint",)),
    },
  )

  # Small constant reward for staying alive — encourages longer episodes, critical
  # when the adaptive sampler is hammering hard bins with very short episodes.
  cfg.rewards["alive"] = RewardTermCfg(func=local_mdp.is_alive, weight=0.5)

  # Penalize ground contact during reference flight frames.
  cfg.rewards["flight_contact"] = RewardTermCfg(
    func=local_mdp.flight_contact_penalty,
    weight=-2.0,
    params={
      "command_name": "motion",
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      ),
    },
  )

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
