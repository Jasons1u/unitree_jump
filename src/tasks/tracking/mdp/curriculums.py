from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_tracking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> torch.Tensor:
  """Curriculum based on episode survival for tracking tasks.

  Move up if the robot survived most of the episode; move down if it
  terminated early (fell / lost tracking).
  """
  terrain = env.scene.terrain
  assert terrain is not None

  # Fraction of max episode length the robot survived before reset.
  survival = env.episode_length_buf[env_ids].float() / env.max_episode_length

  move_up = survival > 0.8
  move_down = (survival < 0.2) & ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())
