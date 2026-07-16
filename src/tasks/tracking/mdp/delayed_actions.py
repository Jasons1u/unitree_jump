"""Standalone actuator command delay, implemented at the action-term layer.

Vendored into this repo so it works against **stock** ``mjlab`` (e.g. the PyPI
``mjlab==1.2.0`` release), which has no built-in actuator-delay engine. Instead
of patching the actuator, we subclass the public ``JointPositionAction`` and lag
the *position target* — for a position policy the action->target map is a
time-invariant affine transform, so delaying the target is equivalent to
delaying the command reaching the motor.

Cadence (mjlab ``ManagerBasedRlEnv.step``):
  * ``process_actions`` runs once per **control** step (before the decimation
    loop) — we (re)sample a per-env lag there.
  * ``apply_actions`` runs once per **physics** substep (inside the loop) — we
    push the current target into a ring buffer and serve the target from
    ``lag`` physics steps ago.

This reproduces the fork's actuator DelayBuffer behaviour: per-env integer lag
in ``[min, max]`` physics steps, re-sampled each control step and held across
the substeps, at physics-step resolution.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["DelayedJointPositionActionCfg", "DelayedJointPositionAction"]


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """``JointPositionActionCfg`` that lags the position target by a random delay.

  Delays are given in seconds and converted to physics steps via the sim
  timestep. ``max_delay_sec = 0`` makes it behave exactly like the stock action.
  """

  min_delay_sec: float = 0.0
  max_delay_sec: float = 0.02

  def build(self, env: ManagerBasedRlEnv) -> DelayedJointPositionAction:
    return DelayedJointPositionAction(self, env)

  @classmethod
  def from_position_cfg(
    cls,
    base: JointPositionActionCfg,
    *,
    min_delay_sec: float,
    max_delay_sec: float,
  ) -> DelayedJointPositionActionCfg:
    """Build a delayed cfg copying every field of an existing position cfg."""
    return cls(
      **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
      min_delay_sec=min_delay_sec,
      max_delay_sec=max_delay_sec,
    )


class DelayedJointPositionAction(JointPositionAction):
  """Position action that serves a per-env, physics-step-lagged target."""

  cfg: DelayedJointPositionActionCfg

  def __init__(
    self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg=cfg, env=env)

    timestep = env.cfg.sim.mujoco.timestep
    self._min_lag = round(cfg.min_delay_sec / timestep)
    self._max_lag = round(cfg.max_delay_sec / timestep)
    assert 0 <= self._min_lag <= self._max_lag, (
      f"invalid delay lags: min={self._min_lag}, max={self._max_lag}"
    )
    self._enabled = self._max_lag > 0
    if not self._enabled:
      return

    # Ring buffer of position targets: (num_envs, buf_len, action_dim).
    self._buf_len = self._max_lag + 1
    resting = self._resting_target()
    self._buf = resting.unsqueeze(1).repeat(1, self._buf_len, 1).contiguous()
    self._ptr = 0  # next write slot (shared: all envs step in lockstep)
    self._lags = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._arange = torch.arange(self.num_envs, device=self.device)

  def _resting_target(self) -> torch.Tensor:
    """Default joint pose for the controlled joints — a safe fill value so a
    just-reset env never reads stale/zero targets out of the buffer."""
    return self._entity.data.default_joint_pos[:, self._target_ids]

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)  # sets self._processed_actions
    if self._enabled:
      # Re-sample the per-env lag once per control step; held across substeps.
      self._lags = torch.randint(
        self._min_lag,
        self._max_lag + 1,
        (self.num_envs,),
        device=self.device,
      )

  def apply_actions(self) -> None:
    if not self._enabled:
      super().apply_actions()
      return
    # Write current target, then read the one `lag` physics steps back.
    self._buf[:, self._ptr] = self._processed_actions
    read_idx = (self._ptr - self._lags) % self._buf_len
    delayed = self._buf[self._arange, read_idx]
    self._ptr = (self._ptr + 1) % self._buf_len

    # Mirror JointPositionAction.apply_actions with the delayed target.
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      delayed - encoder_bias, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if not self._enabled:
      return
    resting = self._resting_target()
    if env_ids is None:
      self._buf[:] = resting.unsqueeze(1)
      self._lags[:] = 0
    else:
      self._buf[env_ids] = resting[env_ids].unsqueeze(1)
      self._lags[env_ids] = 0
