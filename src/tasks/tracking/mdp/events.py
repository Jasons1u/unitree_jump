from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def push_by_setting_velocity_grounded(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  height_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply a random velocity push only to robots currently near the ground.

  Identical to ``push_by_setting_velocity`` but gates on root height: envs
  where ``root_z > height_threshold`` are skipped so aerial robots (e.g.
  mid-backflip) are never disturbed.

  Args:
    env: The RL environment.
    env_ids: Candidate environment indices selected by the event manager.
    velocity_range: Per-axis velocity perturbation ranges, same format as
      ``push_by_setting_velocity``.
    height_threshold: Maximum root link height (m) for a push to be applied.
      Robots above this are considered airborne and are skipped.
    asset_cfg: Which scene asset to push.
  """
  asset: Entity = env.scene[asset_cfg.name]

  root_z = asset.data.root_link_pos_w[env_ids, 2]
  grounded_mask = root_z <= height_threshold
  grounded_ids = env_ids[grounded_mask]

  if grounded_ids.numel() == 0:
    return

  vel_w = asset.data.root_link_vel_w[grounded_ids]
  range_list = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  vel_w = vel_w + sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=grounded_ids)
