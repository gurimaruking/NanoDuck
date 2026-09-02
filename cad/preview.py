#!/usr/bin/env python3
"""Assemble the printed parts in the home pose and render them.

    python preview.py

The MJCF in sim/ is boxes -- it is a kinematic and mass model, not a shape.
This puts the actual STLs where they go, so what you see is what comes off the
printer.  It also doubles as a coarse assembly check: parts that cannot be
placed at the home-pose transforms will show it here.

Scope: STRUCTURE ONLY, plus the soles.  The head and body shells are MicroDuck
exports whose origins and axes are whatever Onshape gave them, and locating
them properly is a real alignment job, not a guess.  A first version of this
script placed them by centring and a 90 degree rotation; the result looked
plausible at a glance and was wrong, which is worse than leaving them out.
They print and fit as separate parts -- they are just not posed here.

Output: print/assembly.stl (one merged mesh) and print/assembly.png.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh

import build_nanoduck as B
import parts as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "print")
MM = 0.001


def rot_y(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0])


def rot_x(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0])


def place(mesh, xyz, rot=None):
    m = mesh.copy()
    if rot is not None:
        m.apply_transform(rot)
    m.apply_translation(xyz)
    return m


def assemble():
    links = P.build_links()
    cos = P.cosmetics()
    out = []

    # Home pose, in degrees, from build_nanoduck.
    hp, kn, an = (np.degrees(B.HIP_PITCH_0), np.degrees(B.KNEE_0), np.degrees(B.ANKLE_0))

    for side, sgn in (("left", +1), ("right", -1)):
        # Chain of transforms down one leg, mirroring the MJCF exactly.
        base = np.array([0.0, sgn * B.HIP_Y, -B.TRUNK[2] / 2])
        T = np.eye(4)
        T[:3, 3] = base
        chain = [
            (links["hip_yoke"], 0.0, B.HIP_ROLL_TO_PITCH),
            (links["thigh"], hp, B.THIGH),
            (links["shin"], kn, B.SHIN),
            (links["foot_mount"], an, B.ANKLE_TO_SOLE),
        ]
        ang = 0.0
        pos = base.copy()
        for mesh, dtheta, length in chain:
            ang += dtheta
            R = rot_y(ang)
            m = mesh.copy()
            if sgn < 0:
                m = P.mirror(m)
            m.apply_transform(R)
            m.apply_translation(pos)
            out.append(m)
            # Step down the link along its own -Z, rotated by the accumulated angle.
            d = R[:3, :3] @ np.array([0.0, 0.0, -length])
            pos = pos + d
        # Cosmetic sole at the ankle.
        sole = cos.get("cosmetic_sole_%s" % ("left" if sgn > 0 else "right"))
        if sole is not None:
            s = sole.copy()
            s.apply_translation(-s.bounds.mean(axis=0))
            s.apply_transform(rot_y(ang))
            s.apply_translation(pos + np.array([B.FOOT_X, 0, -2.0]))
            out.append(s)

    # Neck link only; the head shells are not posed (see the module docstring).
    neck_base = np.array([B.NECK_X, 0.0, B.TRUNK[2] / 2])
    n = links["neck_link"].copy()
    n.apply_transform(rot_y(180 - np.degrees(B.NECK_PITCH_0)))
    n.apply_translation(neck_base)
    out.append(n)

    merged = trimesh.util.concatenate(out)
    return merged


def main():
    os.makedirs(OUT, exist_ok=True)
    a = assemble()
    # Sit it on the floor.
    a.apply_translation([0, 0, -a.bounds[0][2]])
    path = os.path.join(OUT, "assembly.stl")
    a.export(path)
    print("wrote %s" % path)
    print("assembled envelope: %.0f x %.0f x %.0f mm" % tuple(a.extents))

    scene_xml = """<mujoco model="nanoduck_print">
  <compiler angle="radian" meshdir="."/>
  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="140" elevation="-15" offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.42 0.52" rgb2="0.05 0.08 0.12"
             width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.24 0.28 0.32"
             rgb2="0.18 0.22 0.26" markrgb="0.7 0.7 0.7" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="6 6" reflectance="0.15"/>
    <mesh name="duck" file="assembly.stl" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 0.8" dir="-0.3 0.3 -0.8" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="grid"/>
    <geom name="duck" type="mesh" mesh="duck" pos="0 0 0" rgba="0.93 0.91 0.85 1"/>
  </worldbody>
</mujoco>
"""
    xml_path = os.path.join(OUT, "_preview.xml")
    with open(xml_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(scene_xml)

    import mujoco
    from PIL import Image
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=760, width=620)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 0.085]
    cam.distance = 0.40
    cam.elevation = -10
    imgs = []
    for az in (140, 90, 180):
        cam.azimuth = az
        r.update_scene(d, cam)
        imgs.append(Image.fromarray(r.render()))
    sheet = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height))
    x = 0
    for i in imgs:
        sheet.paste(i, (x, 0))
        x += i.width
    png = os.path.join(OUT, "assembly.png")
    sheet.save(png)
    print("wrote %s" % png)


if __name__ == "__main__":
    main()
