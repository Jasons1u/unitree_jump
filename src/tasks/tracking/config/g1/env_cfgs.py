"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from src.assets.robots import (
  G1_AGILITY_ACTION_SCALE,
  get_g1_agility_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import BoxFlatTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.tracking.terrains import BoxTiltedPlaneTerrainCfg

import src.tasks.tracking.mdp as local_mdp
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from src.tasks.tracking.tracking_env_cfg_baseline import make_baseline_tracking_env_cfg


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
  # Finer reset-curriculum bins so adaptive sampling can target short, hard
  # moments (e.g. takeoff/flip) instead of pinning to coarse 1s bins.
  motion_cmd.adaptive_bin_seconds = 0.25

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

  # Use the Agility robot variant with custom (softer) waist gains.
  cfg.scene.entities = {"robot": get_g1_agility_robot_cfg()}
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_AGILITY_ACTION_SCALE

  # Standalone actuator command delay (training only): swap the position action
  # for a delayed variant that lags the position target 0-0.02s (0 to one control
  # step) per env, modeling policy-to-motor latency. Implemented purely on the
  # public action API (no fork actuator engine), so it works against stock mjlab.
  # Play/eval keeps the stock action, mirroring the drop of perturbation-style DR.
  if not play:
    cfg.actions["joint_pos"] = local_mdp.DelayedJointPositionActionCfg.from_position_cfg(
      joint_pos_action, min_delay_sec=0.0, max_delay_sec=0.02
    )

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
        "flat": BoxFlatTerrainCfg(proportion=0.3),
        "tilted": BoxTiltedPlaneTerrainCfg(proportion=0.7, max_tilt_deg=5.0),
      },
    ),
  )

  # Terminations: drop ori + ee checks (fire during flight); relax height.
  # cfg.terminations.pop("anchor_ori", None)
  # cfg.terminations.pop("ee_body_pos", None)
  cfg.terminations["anchor_pos"].params["threshold"] = 0.35

  # Terminate if the anchor orientation deviates more than this many degrees
  # from the reference (geodesic angle across all three axes combined). Relavent for fast flips and twists
  cfg.terminations["anchor_ori_angle"] = TerminationTermCfg(
    func=local_mdp.bad_anchor_ori_angle,
    params={"command_name": "motion", "threshold_deg": 45.0},
  )

  # Curriculum: log mean terrain level, progress based on episode survival.
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(func=local_mdp.terrain_levels_tracking),
  }

  # Height-gated push: skip robots that are airborne.
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].func = local_mdp.push_by_setting_velocity_grounded
    cfg.events["push_robot"].params["height_threshold"] = 0.7
  # cfg.events.pop("push_robot", None)

  # Penalize asymmetric hip joints:
  #   pitch (Y-axis): symmetric  → sign= +1  (same range both sides)
  #   roll  (X-axis): mirrored ranges (-0.52,2.97) vs (-2.97,0.52) → sign= -1
  #   yaw   (Z-axis): anti-symmetric → sign= -1  (toe-out convention)

  
  # cfg.rewards["hip_symmetry"] = RewardTermCfg(
  #   func=local_mdp.joint_pair_symmetry_l2,
  #   weight=-2.0,
  #   params={
  #     "left_cfg": SceneEntityCfg("robot", joint_names=(
  #       "left_hip_pitch_joint",
  #       "left_hip_roll_joint",
  #       "left_hip_yaw_joint",
  #     )),
  #     "right_cfg": SceneEntityCfg("robot", joint_names=(
  #       "right_hip_pitch_joint",
  #       "right_hip_roll_joint",
  #       "right_hip_yaw_joint",
  #     )),
  #     "signs": (1.0, -1.0, -1.0),  # pitch symmetric, roll/yaw anti-symmetric
  #   },
  # )

  # Penalize waist roll — prevents robot from leaning to one side to hop on one leg.
  # cfg.rewards["waist_roll"] = RewardTermCfg(
  #   func=local_mdp.joint_pos_l2,
  #   weight=-1.0,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_roll_joint",)),
  #   },
  # )

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
# BASELINE FLAT ENV (ablation control)
#
# G1 flat builder that inherits from `make_baseline_tracking_env_cfg` (the
# isolated original-mimic base) instead of the custom `make_tracking_env_cfg`.
# It reproduces the *upstream* g1 flat builder and only ever touches the 4
# original DR events, so the custom DR stack can never leak into the ablation.
##################################################################

def unitree_g1_baseline_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 flat tracking config on the isolated baseline base."""
  cfg = make_baseline_tracking_env_cfg()

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
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

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
# Ablation — original-mimic baseline env, minus the pitch (anchor_ori)
# termination, plus the 0.25s adaptive sampling bin size.
##################################################################

def unitree_g1_ablation_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation control: the isolated original-mimic baseline env, differing from
  it by exactly two deltas.

  Inherits from `unitree_g1_baseline_flat_tracking_env_cfg` (original upstream
  DR / rewards) rather than the custom flat builder, so none of the custom DR
  stack leaks in. The only differences vs the baseline are the two deltas below.
  """
  cfg = unitree_g1_baseline_flat_tracking_env_cfg(
    has_state_estimation=False, play=play
  )

  # Ablation delta 1: remove the pitch/orientation termination.
  cfg.terminations.pop("anchor_ori", None)

  # Ablation delta 2: finer adaptive reset bins (default 1.0s) so adaptive
  # sampling can target short, hard moments instead of coarse 1s bins.
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.adaptive_bin_seconds = 0.25

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
