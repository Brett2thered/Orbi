#!/usr/bin/env python3
"""Write the Orbi MJCF model (SI units, +Z up) from the page's own geometry.

USAGE: python build_model.py [--terrain course|flat|step] [--step-h 0.122] [--out orbi/model.xml]

Geometry is taken from index.html (1 page unit = 1 ft = 0.3048 m, page Y-up -> here Z-up via
(x, y, z)_page -> (x, -z, y)). Mass, actuator force and pad speed are NOT on the page: they are
the ASSUMED block below, and every measured number depends on them.
"""
import argparse
from pathlib import Path

import numpy as np

FT = 0.3048
# ── from the page ──
R = 1.0 * FT                    # sphere radius (ø 610 mm)
PAD_ANG = np.radians(21.0)      # half-angle of each panel cap
STROKE = 1.0 * R                # EXT_MAX = 1.0 x radius
# ── ASSUMED (not on the page) ──
MASS_TOTAL = 12.0               # kg
PAD_MASS = 0.20                 # kg each
ACT_FORCE = 300.0               # N, per zipper-chain actuator
ACT_KP = 8000.0                 # N/m
ACT_KV = 120.0                  # N s/m
FRICTION = 1.0                  # elastomer skin on floor

A_PAD = R * np.sin(PAD_ANG)      # panel rim radius
C_PAD = A_PAD ** 2 / R           # ellipsoid depth matching the sphere's curvature at the apex


def contact_ext(down, h=R):
    """Extension at which a panel (the model's ellipsoid) first touches flat ground.

    down = minus the world-Z component of the panel axis; h = centre height over the ground.
    A tilted panel touches with its rim, well before its apex would. inf when it cannot reach."""
    down = np.asarray(down, float)
    rim = np.sqrt(A_PAD ** 2 * np.clip(1 - down ** 2, 0, 1) + C_PAD ** 2 * down ** 2)
    ext = (h - rim) / np.maximum(down, 1e-6) - R + C_PAD
    return np.where(down > 0.05, np.maximum(ext, 0.0), np.inf)


AXES = [np.array(v, float) for s in (1, -1) for v in ((s, 0, 0), (0, s, 0), (0, 0, s))]
AXES += [np.array((x, y, z), float) / np.sqrt(3) for x in (1, -1) for y in (1, -1) for z in (1, -1)]

# page terrain: boxes (x0, x1, z0, z1, h) and pipes (ax, az, bx, bz, radius), page units
BOXES = [(4, 5.6, -2.5, 2.5, .4), (5.6, 7.2, -2.5, 2.5, .8), (7.2, 8.8, -2.5, 2.5, 1.2), (8.8, 14.4, -2.5, 2.5, 1.6),
         (10, 14.4, 2.5, 4.1, 1.2), (10, 14.4, 4.1, 5.7, .8), (10, 14.4, 5.7, 7.3, .4),
         (-4.8, -3.2, 2.2, 3.8, .5), (-7, -5, -2.6, -1.4, .35), (-2.6, -1.4, -5.6, -4.4, .7),
         (.5, 3.5, 6, 6.6, .3), (.75, 1.25, -10, -6.4, .3),
         (-11, -10, -6.5, -5.5, 3), (6.2, 7.2, -7.4, -6.4, 2.2)]
_hs = [.15, .35, .2, .45, .3, .1, .4, .25, .2, .45, .15, .3, .4, .2, .35, .1]
BOXES += [(-9.5 + i, -8.5 + i, 5 + j, 6 + j, _hs[i * 4 + j]) for i in range(4) for j in range(4)]
PIPES = [(3, -6, 3, -2.6, .3), (-5, 9.6, -1.4, 9.6, .45)]


def quat_z_to(a):
    """Quaternion (w x y z) rotating +Z onto unit vector a."""
    z = np.array([0, 0, 1.0])
    c = float(z @ a)
    if c > 1 - 1e-9:
        return np.array([1, 0, 0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0, 1, 0, 0.0])
    v = np.cross(z, a)
    q = np.array([1 + c, *v])
    return q / np.linalg.norm(q)


def f(*xs):
    return " ".join(f"{x:.6g}" for x in xs)


def terrain_xml(kind, step_h):
    out = []
    if kind == "course":
        for i, (x0, x1, z0, z1, h) in enumerate(BOXES):
            sx, sy, sz = (x1 - x0) / 2 * FT, (z1 - z0) / 2 * FT, h / 2 * FT
            out.append(f'<geom name="box{i}" class="terrain" type="box" size="{f(sx, sy, sz)}" '
                       f'pos="{f((x0 + x1) / 2 * FT, -(z0 + z1) / 2 * FT, sz)}"/>')
        for i, (ax, az, bx, bz, c) in enumerate(PIPES):
            c *= FT
            out.append(f'<geom name="pipe{i}" class="terrain" type="capsule" size="{f(c)}" '
                       f'fromto="{f(ax * FT, -az * FT, c, bx * FT, -bz * FT, c)}"/>')
    elif kind == "step":
        out.append(f'<geom name="step" class="terrain" type="box" size="{f(2.0, 1.5, step_h / 2)}" '
                   f'pos="{f(1.5 + 2.0, 0, step_h / 2)}"/>')
    return "\n    ".join(out)


def build(kind="course", step_h=0.122):
    a_pad, c_pad = A_PAD, C_PAD
    shell_mass = MASS_TOTAL - 14 * PAD_MASS
    I_shell = 0.5 * shell_mass * R ** 2  # between a solid ball (0.4) and a thin shell (0.667): core + frame + skin
    I_ax, I_tr = 0.5 * PAD_MASS * a_pad ** 2, 0.25 * PAD_MASS * a_pad ** 2
    pads, acts, sens = [], [], []
    for i, a in enumerate(AXES):
        n = f"{i + 1:02d}"
        pads.append(f'''
      <body name="pad{n}" quat="{f(*quat_z_to(a))}">
        <joint name="ext{n}" type="slide" axis="0 0 1" range="0 {f(STROKE)}" armature="0.05" damping="2"/>
        <inertial pos="{f(0, 0, R - c_pad)}" mass="{f(PAD_MASS)}" diaginertia="{f(I_tr, I_tr, I_ax)}"/>
        <geom name="pad{n}_col" class="pad" pos="{f(0, 0, R - c_pad)}" size="{f(a_pad, a_pad, c_pad)}"/>
        <site name="pad{n}_touch" type="ellipsoid" pos="{f(0, 0, R - c_pad)}" size="{f(a_pad * 1.05, a_pad * 1.05, c_pad * 1.3)}" group="4"/>
      </body>''')
        acts.append(f'<position name="act{n}" joint="ext{n}" kp="{f(ACT_KP)}" kv="{f(ACT_KV)}" '
                    f'ctrlrange="0 {f(STROKE)}" forcerange="{f(-ACT_FORCE, ACT_FORCE)}"/>')
        sens.append(f'<touch name="touch{n}" site="pad{n}_touch"/>')
    nl = "\n    "
    return f'''<mujoco model="orbi">
  <!-- Generated by sim/build_model.py; edit that file, not this one. SI units, +Z up. terrain: {kind} -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast">
    <flag energy="enable"/>
  </option>
  <default>
    <default class="terrain"><geom contype="2" conaffinity="1" friction="{f(FRICTION)} 0.005 0.0001" rgba="0.8 0.8 0.8 1"/></default>
    <default class="pad"><geom type="ellipsoid" contype="1" conaffinity="2" friction="{f(FRICTION)} 0.005 0.0001" rgba="0.95 0.95 0.95 1"/></default>
  </default>
  <worldbody>
    <light pos="0 -3 5" dir="0 0.5 -1"/>
    <geom name="floor" class="terrain" type="plane" size="12 12 0.05"/>
    {terrain_xml(kind, step_h)}
    <body name="shell" pos="0 0 {f(R)}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{f(shell_mass)}" diaginertia="{f(I_shell, I_shell, I_shell)}"/>
      <geom name="shell_col" type="sphere" size="{f(R)}" contype="1" conaffinity="2" condim="6"
            friction="{f(FRICTION)} 0.005 0.002" rgba="0.95 0.95 0.95 1"/>
      <site name="center"/>{"".join(pads)}
    </body>
  </worldbody>
  <actuator>
    {nl.join(acts)}
  </actuator>
  <sensor>
    <framepos name="pos" objtype="site" objname="center"/>
    <framelinvel name="vel" objtype="site" objname="center"/>
    <frameangvel name="angvel" objtype="site" objname="center"/>
    {nl.join(sens)}
  </sensor>
  <keyframe>
    <key name="start" qpos="0 0 {f(R)} 0.9689 0.2 0.14 0.05 {f(*([0] * 14))}"/>
  </keyframe>
</mujoco>
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="course", choices=["course", "flat", "step"])
    ap.add_argument("--step-h", type=float, default=0.122)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "orbi" / "model.xml")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(build(a.terrain, a.step_h))
    print(f"wrote {a.out} (terrain={a.terrain})")
