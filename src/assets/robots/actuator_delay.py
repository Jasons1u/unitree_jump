"""Actuator command-delay helper, vendored into this repo.

Lives here (not patched into the installed ``mjlab`` package) so it survives
``pip install`` / fresh-env recreation and travels with the repo. It depends
only on stock mjlab 1.2.0 primitives: the ``delay_*`` fields on
``BuiltinPositionActuatorCfg`` (mjlab actuator PRs #857 / #1001) and the
built-in delay engine in ``mjlab.actuator.builtin_group``.

Actuator delay is a built-in actuator property, not an event term: each control
step a per-env lag is sampled uniformly in ``[min, max]`` physics steps and
applied to the command targets, modeling policy-to-motor latency.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

__all__ = ["add_actuator_delay"]


def add_actuator_delay(
  cfg: ManagerBasedRlEnvCfg,
  min_delay_sec: float = 0.0,
  max_delay_sec: float = 0.02,
) -> None:
  """Add per-env command delay to the robot's built-in position actuators in place.

  Delays are given in seconds and converted to physics steps via the sim
  timestep; the lag is re-sampled once per control step (``cfg.decimation``).

  Rebuilds the robot entity functionally (via ``dataclasses.replace``) so the
  shared actuator constants in the robot's ``*_constants`` module are left
  untouched. Only ``BuiltinPositionActuatorCfg`` actuators are modified; any
  other actuator type is passed through unchanged.

  Args:
    cfg: An already-built env config whose ``robot`` entity uses the built-in
      position actuators.
    min_delay_sec: Minimum command delay in seconds.
    max_delay_sec: Maximum command delay in seconds. ``0`` disables delay.
  """
  robot_cfg = cfg.scene.entities["robot"]
  assert isinstance(robot_cfg, EntityCfg)
  assert robot_cfg.articulation is not None

  timestep = cfg.sim.mujoco.timestep
  min_lag = round(min_delay_sec / timestep)
  max_lag = round(max_delay_sec / timestep)

  actuators = tuple(
    replace(
      act,
      delay_min_lag=min_lag,
      delay_max_lag=max_lag,
      delay_update_period=cfg.decimation,
      delay_per_env_phase=True,
    )
    if isinstance(act, BuiltinPositionActuatorCfg)
    else act
    for act in robot_cfg.articulation.actuators
  )
  cfg.scene.entities["robot"] = replace(
    robot_cfg,
    articulation=replace(robot_cfg.articulation, actuators=actuators),
  )
