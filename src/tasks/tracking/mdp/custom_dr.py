"""Custom domain-randomization terms vendored into this repo.

These live here (not patched into the installed ``mjlab`` package) so they
survive ``pip install`` / fresh-env recreation and travel with the repo.

They are thin wrappers around mjlab's stock ``_randomize_model_field`` engine
(in ``mjlab.envs.mdp.dr._core``), which ships with mjlab 1.2.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _randomize_model_field,
)
from mjlab.envs.mdp.dr._types import Distribution, Operation
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["geom_solref"]


@requires_model_fields("geom_solref")
def geom_solref(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom solver reference parameters (timeconst and dampratio).

  ``geom_solref`` has two axes:
  - axis 0: timeconst — spring time constant (smaller = stiffer surface).
  - axis 1: dampratio — damping ratio (smaller = more elastic / higher effective COR).

  Default randomizes axis 1 (dampratio) only, which is the closest MuJoCo analogue to
  coefficient of restitution. Pass ``axes=[0, 1]`` to also vary surface stiffness.
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_solref",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[1],
    valid_axes=[0, 1],
  )
