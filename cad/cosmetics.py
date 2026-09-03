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
    # head: solved, not chosen. See check_orientation() -- this is the only
    # axis-aligned rotation that puts the top shell above the bottom one, the
    # bill forward, and the long axis fore-aft.
    #
    # It was (180, 90, 0) for a while, which passed a test on the cluster's
    # fore-aft EXTENT and shipped the duck upside down. Extent along X says the
    # long axis is X; it says nothing whatever about a 180 degree flip around
    # it. Up/down and front/back are separate facts and need separate tests.
    # head at the SAME scale as the body, which is the whole point: MicroDuck's
    # head-to-body ratio is 1.52, and any two different scales destroy it. At
    # 0.62 the head measured 1.11 -- 27% too small, which is exactly why the
    # first renders read as "a duck, but wrong". It costs 33 g.
    "head": (0.85, ["bottom_head_shell", "top_head_shell", "face_part", "jaw", "noenoeil"],
             (0.0, 270.0, 0.0)),
    "trunk": (0.85, ["left_shell", "right_shell"], (0.0, 0.0, 0.0)),
    "sole_left": (0.75, ["sole_left"], (90.0, 0.0, 0.0)),
    "sole_right": (0.75, ["sole_right"], (90.0, 0.0, 0.0)),
}


# Per-mesh colour. The shells arrive as separate parts, so this costs nothing
# and it is most of what was left of "it does not look like a MicroDuck": the
# real robot is off-white with an ORANGE bill and brow band and a black eye, and
# a render in one flat cream reads as a generic white robot instead.
COLOUR = {
    "_default":          "0.93 0.92 0.89 1",   # shells
    "jaw":               "0.95 0.45 0.05 1",   # the bill
    "face_part":         "0.95 0.45 0.05 1",   # the band around the brow
    "noenoeil":          "0.09 0.09 0.11 1",   # eye
    "sole_left":         "0.95 0.55 0.10 1",   # webbed feet
    "sole_right":        "0.95 0.55 0.10 1",
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
                   'rgba="%s" pos="%s" quat="%s"/>\n'
                   % (name, name, COLOUR.get(name, COLOUR["_default"]),
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


def posed_mesh(group, name):
    """One mesh of a group, in NanoDuck's body axes."""
    scale, _, _ = GROUPS[group]
    pos, quat = upstream_transforms()[name]
    m = load_stl_mm(name)
    m.apply_scale(scale)
    m.apply_transform(quat_matrix(quat))
    m.apply_translation(pos * 1000.0 * scale)
    m.apply_transform(extra_matrix(group))
    return m


def check_orientation(verbose=True):
    """Is the duck the right way up and facing forward?

    A bounding-box extent cannot answer this.  "76.1 mm fore-aft" proves the
    long axis is X and is perfectly happy with the head rotated 180 degrees
    about that axis -- which is exactly how a flipped head shipped once.

    So test the facts separately, against parts whose relative position is not
    in doubt: the top shell is above the bottom one, the bill is ahead of the
    skull, the eyes are in the upper half, and the left panel is to the left.
    Each is a signed distance between two centroids; each must be positive.
    """
    checks = []

    def cen(group, name):
        return posed_mesh(group, name).centroid

    h = lambda n: cen("head", n)
    checks += [
        ("head is not upside down", h("top_head_shell")[2] - h("bottom_head_shell")[2]),
        ("bill points forward", h("jaw")[0] - h("bottom_head_shell")[0]),
        ("eyes are in the upper half", h("noenoeil")[2] - h("bottom_head_shell")[2]),
        ("left body panel is on the left",
         cen("trunk", "left_shell")[1] - cen("trunk", "right_shell")[1]),
    ]
    lo, hi = cluster_bounds("head")
    checks.append(("head is longest fore-aft",
                   (hi - lo)[0] - max((hi - lo)[1], (hi - lo)[2])))

    ok = True
    for label, value in checks:
        good = value > 0.0
        ok &= good
        if verbose:
            print("   %-32s %+7.1f mm   %s" % (label, value, "ok" if good else "WRONG"))
    return ok


if __name__ == "__main__":
    for g in GROUPS:
        lo, hi = cluster_bounds(g)
        print("%-12s bbox %6.1f x %6.1f x %6.1f mm   centre %s"
              % (g, *(hi - lo), np.round((lo + hi) / 2, 1)))
    print()
    print("orientation:")
    raise SystemExit(0 if check_orientation() else 1)


# --- the printed structure, as visual meshes ---------------------------------
# MicroDuck has no leg shells: on that robot the servos and their brackets ARE
# the leg. The same is true here, so the visual for a leg link is simply the
# part that gets printed.  parts.py builds those in the LINK frame -- origin on
# the driven joint axis, link along -Z -- which is exactly the MJCF body frame,
# so they drop in with no transform at all.
# body -> (parts, quat) . The hip and neck turn a corner, so they are two
# bolted parts each (cad/parts.corner_link). The hip assembly is also rotated
# when it is placed: the part is built with its driven axis on the part frame's
# Y, and hip_roll is X, so it goes in turned -90 deg about Z.
PRINTED = {
    "left_hip": (["hip_yoke_a_L", "hip_yoke_b_L"], (0.707107, 0, 0, -0.707107)),
    "right_hip": (["hip_yoke_a_R", "hip_yoke_b_R"], (0.707107, 0, 0, -0.707107)),
    "left_thigh": (["thigh_L"], None), "right_thigh": (["thigh_R"], None),
    "left_shin": (["shin_L"], None), "right_shin": (["shin_R"], None),
    "left_foot": (["foot_mount_L"], None), "right_foot": (["foot_mount_R"], None),
    # The neck is the one link that runs UP. parts.link() builds every part
    # with its driving joint at -Z, which is right for all six leg parts and
    # upside down here: the head sits at +20 mm and the part reached from -44
    # to +5, i.e. down into the trunk. Flipped 180 deg about X, which also
    # leaves neck_pitch on Y and head_yaw on Z, just with reversed sense.
    "neck": (["neck_link_a", "neck_link_b"], (0.0, 1.0, 0.0, 0.0)),
}


def printed_assets():
    seen, out = set(), []
    for parts, _ in PRINTED.values():
      for part in parts:
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
    """Visual geoms for the printed parts on one body.

    A body can carry more than one part: the hip and the neck each turn a 90
    degree corner and are split across a bolted plate, because a carrier
    twisted inside a single part sweeps into its own yoke arms
    (cad/parts.corner_link).
    """
    entry = PRINTED.get(body)
    if entry is None:
        return ""
    parts, quat = entry
    q = "" if quat is None else ' quat="%s"' % " ".join("%.6f" % v for v in quat)
    return "".join(
        '        <geom name="skin_%s" type="mesh" mesh="%s" class="skin"%s '
        'rgba="0.22 0.22 0.24 1"/>\n' % (part, part, q) for part in parts)
