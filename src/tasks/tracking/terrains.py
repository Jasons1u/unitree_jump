from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class BoxTiltedPlaneTerrainCfg(SubTerrainCfg):
  """Flat ground plane with a small random tilt in roll and pitch.

  Simulates soft surface deformation (e.g. a foam mat compressing unevenly
  under foot contact), producing a heel-to-toe or side-to-side slope.
  Tilt is sampled independently per patch at terrain-generation time.
  """

  max_tilt_deg: float = 5.0
  """Maximum tilt magnitude in degrees. Each axis is sampled from
  [-max_tilt_deg * difficulty, +max_tilt_deg * difficulty]."""
  plane_thickness: float = 1.0
  """Thickness of the ground box, in meters."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])

    max_tilt = np.deg2rad(self.max_tilt_deg * difficulty)
    roll = rng.uniform(-max_tilt, max_tilt)
    pitch = rng.uniform(-max_tilt, max_tilt)

    # Compose Rx(roll) * Ry(pitch) into a single unit quaternion (w, x, y, z).
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    quat = np.array([cr * cp, sr * cp, sp * cr, -sr * sp])

    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, self.plane_thickness / 2),
      pos=(self.size[0] / 2, self.size[1] / 2, -self.plane_thickness / 2),
      quat=quat,
    )
    return TerrainOutput(
      origin=origin,
      geometries=[TerrainGeometry(geom=geom, color=(0.55, 0.50, 0.45, 1.0))],
    )
