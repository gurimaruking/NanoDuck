"""Place MicroDuck's shells on NanoDuck, using MicroDuck's own transforms.

The shells are Onshape exports whose origins and axes are whatever the CAD gave
them.  Guessing at them does not work -- a first attempt centred each mesh and
rotated it 90 degrees, which looked plausible in a render and was wrong.

But the transforms are not actually unknown.  `robot_walk.xml` positions every
one of these meshes inside a body, with an exact pos and quat.  Reusing those
is the difference between alignment and decoration:

    left_shell / right_shell   body trunk_base   quat 0.707107 0 0 0.707107
    top/bottom_head_shell      body jaw_soft     quat 0.5 0.5 0.5 0.5
    face_part, jaw, noenoeil   body jaw_soft     quat 0.5 0.5 0.5 0.5
    sole_left                  body ankle_left   quat 0.5 -0.5 -0.5 -0.5

Within a group the relative positions are already right, so a group only needs
one offset: where the cluster sits on NanoDuck's own body.  That is computed,
not chosen -- each cluster is centred on the primitive geom already carrying its
mass in the MJCF, so the visual shell and the inertia box describe the same
object.
"""

from __future__ import annotations

import os
import re
import struct

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "microduck_src")
MJCF = os.path.join(SRC, "robot_walk.xml")

# group -> (scale, meshes, extra XYZ euler in degrees)
#
# Scale differs per group because the body has to swallow a battery and a PCB
# that do not scale, so it is chubbier than MicroDuck's.
#
# The extra rotation is needed because MicroDuck's body frames are not
# NanoDuck's.  Reusing the upstream pos/quat puts each cluster together
# correctly, but the cluster as a whole still arrives in MicroDuck's axes: the
# head lands with its long (beak) axis along Z instead of X, and a sole with
# its thin axis along Y instead of Z.  One rotation per group fixes that, and
# `python cosmetics.py` prints the resulting extents so it can be checked
# rather than believed.
GROUPS = {
    # head: 90 deg about Y puts the beak axis along +X. Checked by extent, not
    # by eye -- the cluster measures 76.1 x 56.9 x 36.8 mm, and 76.1 is the
    # 123 mm MicroDuck head at 0.62, so it is exactly fore-aft. The nose-up look
    # in a render is MicroDuck's own wedge-shaped skull, not a misalignment;
    # trying 55 and 135 deg by eye made it worse in both directions.
    "head": (0.62, ["bottom_head_shell", "top_head_shell", "face_part", "jaw", "noenoeil"],
             (180.0, 90.0, 0.0)),
    "trunk": (0.85, ["left_shell", "right_shell"], (0.0, 0.0, 0.0)),
    "sole_left": (0.75, ["sole_left"], (90.0, 0.0, 0.0)),
    "sole_right": (0.75, ["sole_right"], (90.0, 0.0, 0.0)),
}


def extra_matrix(group):
    rx, ry, rz = GROUPS[group][2]
    return trimesh.transformations.euler_matrix(
        np.radians(rx), np.radians(ry), np.radians(rz), "sxyz")


def extra_quat(group):
    q = trimesh.transformations.quaternion_from_matrix(extra_matrix(group))
    return np.asarray(q)


def compose(group, quat):
    """Upstream mesh quat, then the group's extra rotation."""
    q = trimesh.transformations.quaternion_multiply(extra_quat(group), quat)
    return q / np.linalg.norm(q)


def upstream_transforms():
    """mesh name -> (pos [m], quat wxyz) as MicroDuck places it."""
    out = {}
    for line in open(MJCF, encoding="utf-8").read().splitlines():
        g = re.search(r'<geom type="mesh"[^>]*mesh="([^"]+)"', line)
        if not g:
            continue
        p = re.search(r'pos="([^"]*)"', line)
        q = re.search(r'quat="([^"]*)"', line)
        if not p or not q or g.group(1) in out:
            continue
        out[g.group(1)] = (np.array([float(v) for v in p.group(1).split()]),
                           np.array([float(v) for v in q.group(1).split()]))
    return out


def load_stl_mm(name):
    with open(os.path.join(SRC, name + ".stl"), "rb") as f:
        d = f.read()
    n = struct.unpack("<I", d[80:84])[0]
    a = np.frombuffer(d[84:84 + 50 * n], dtype=np.uint8).reshape(n, 50)
    v = a[:, 12:48].copy().view(np.float32).reshape(n, 3, 3) * 1000.0
    return trimesh.Trimesh(vertices=v.reshape(-1, 3),
                           faces=np.arange(n * 3).reshape(n, 3), process=True)


def quat_matrix(q):
    w, x, y, z = q
    return trimesh.transformations.quaternion_matrix([w, x, y, z])


def cluster_bounds(group):
    """Bounds [mm] of a posed, scaled cluster, in NanoDuck's body axes."""
    scale, names, _ = GROUPS[group]
    tf = upstream_transforms()
    E = extra_matrix(group)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name in names:
        if name not in tf:
            continue
        m = load_stl_mm(name)
        m.apply_scale(scale)
        pos, quat = tf[name]
        m.apply_transform(quat_matrix(quat))
        m.apply_translation(pos * 1000.0 * scale)
        m.apply_transform(E)
        lo = np.minimum(lo, m.bounds[0])
        hi = np.maximum(hi, m.bounds[1])
    return lo, hi


def cluster_geoms(group, target_centre_mm):
    """MJCF <geom> lines placing the cluster so its centre lands on `target`."""
    scale, names, _ = GROUPS[group]
    tf = upstream_transforms()
    E = extra_matrix(group)
    lo, hi = cluster_bounds(group)
    shift = np.asarray(target_centre_mm, dtype=float) - (lo + hi) / 2.0
    out = []
    for name in names:
        if name not in tf:
            continue
        pos, quat = tf[name]
        p = E[:3, :3] @ (pos * 1000.0 * scale) + shift
        quat = compose(group, quat)
        out.append('        <geom name="skin_%s" type="mesh" mesh="%s" class="skin" '
                   'pos="%s" quat="%s"/>\n'
                   % (name, name,
                      " ".join("%.5f" % (v * 0.001) for v in p),
                      " ".join("%.6f" % v for v in quat)))
    return "".join(out)


def mesh_assets():
    """<mesh> asset lines for every cosmetic mesh, with its group's scale."""
    out = []
    for group, (scale, names, _) in GROUPS.items():
        for name in names:
            if any(('name="%s"' % name) in o for o in out):
                continue
            out.append('    <mesh name="%s" file="%s.stl" scale="%.4f %.4f %.4f"/>\n'
                       % (name, name, scale, scale, scale))
    return "".join(out)


if __name__ == "__main__":
    for g in GROUPS:
        lo, hi = cluster_bounds(g)
        print("%-12s bbox %6.1f x %6.1f x %6.1f mm   centre %s"
              % (g, *(hi - lo), np.round((lo + hi) / 2, 1)))


# --- the printed structure, as visual meshes ---------------------------------
# MicroDuck has no leg shells: on that robot the servos and their brackets ARE
# the leg. The same is true here, so the visual for a leg link is simply the
# part that gets printed.  parts.py builds those in the LINK frame -- origin on
# the driven joint axis, link along -Z -- which is exactly the MJCF body frame,
# so they drop in with no transform at all.
PRINTED = {
    "left_hip": "hip_yoke_L", "right_hip": "hip_yoke_R",
    "left_thigh": "thigh_L", "right_thigh": "thigh_R",
    "left_shin": "shin_L", "right_shin": "shin_R",
    "left_foot": "foot_mount_L", "right_foot": "foot_mount_R",
    "neck": "neck_link",
}


def printed_assets():
    seen, out = set(), []
    for part in PRINTED.values():
        if part in seen:
            continue
        seen.add(part)
        # Units differ by a thousand between the two mesh sources, and MuJoCo
        # will not tell you: the upstream MicroDuck STLs are in METRES (Onshape
        # export), while parts.py writes MILLIMETRES. Forgetting this scale put
        # a 35-metre thigh in the scene.
        out.append('    <mesh name="%s" file="../print/%s.stl" '
                   'scale="0.001 0.001 0.001"/>\n' % (part, part))
    return "".join(out)


def printed_geom(body):
    part = PRINTED.get(body)
    if part is None:
        return ""
    return ('        <geom name="skin_%s" type="mesh" mesh="%s" class="skin" '
            'rgba="0.22 0.22 0.24 1"/>\n' % (part, part))
