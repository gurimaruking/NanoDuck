"""The two halves of a NanoDuck joint, and the servo negative they are cut from.

Everything is in the LINK frame, not the servo's, because that is the frame the
kinematics live in and mixing the two is how the first draft ended up with the
parent's joint axis 90 degrees off the child's:

    +Y   the joint axis
    +Z   along the link, toward the joint this link is driven by
    +X   fore-aft; where the servo body sticks out
    origin on the joint axis, in the plane of the servo's top (horn) face

Where the yoke actually straddles
---------------------------------
An MG90S's output shaft leaves the TOP face -- the 22.8 x 12.2 one -- so the
shaft axis runs along the case HEIGHT.  A yoke that supports the joint on both
sides must span that height (22.7 mm plus the boss), not the 12.2 mm width.
Getting this backwards gives a part that renders beautifully and cannot be
assembled, which is exactly what the first version of this file did.

It does not weaken the transverse-mounting argument in check_servo_fit.py.
What the LINK direction consumes is still the 12.2 mm thin axis:

    along the link (Z):   12.2 case + walls        ~ 20 mm of chain
    across the axis (Y):  22.7 + boss + 2 arms     ~ 34 mm  <- the joint is wide
    fore-aft (X):         32.2 case and ears                <- the drumstick

The Y figure is why the hips sit 40 mm apart and the feet end up slightly
outboard of the body.  On a duck that is correct anatomy.

Assembly of one joint, from the top down:
    child yoke, +Y arm   screwed to the servo horn (3x M2 self-tapper)
    parent carrier       servo dropped in from +Y, ears screwed down
    child yoke, -Y arm   MR63ZZ pressed in, running on a 3x8 mm pin through
                         the carrier floor
"""

from __future__ import annotations

import numpy as np
import trimesh

from servo_spec import (BEARING_OD, BEARING_T, CLEAR, FLOOR, M2_CLEAR, M2_TAP,
                        PIVOT_D, WALL, ServoSpec)

RUN = 0.5          # running clearance between a rotating face and a fixed one


# --- primitives ---------------------------------------------------------------

def box(size, center=(0, 0, 0)):
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    return m


def cyl(d, h, center=(0, 0, 0), axis="z", sections=64):
    m = trimesh.creation.cylinder(radius=d / 2.0, height=h, sections=sections)
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(center)
    return m


def union(*parts):
    parts = [p for p in parts if p is not None]
    return trimesh.boolean.union(parts, engine="manifold") if len(parts) > 1 else parts[0]


def diff(a, *cuts):
    cuts = [c for c in cuts if c is not None]
    return trimesh.boolean.difference([a] + cuts, engine="manifold") if cuts else a


# --- key planes of a joint, so both halves agree on them ----------------------

def planes(s: ServoSpec):
    """Y coordinates that both halves must be cut to."""
    arm_t = WALL + 0.6
    floor_y = -(s.case_h + CLEAR + FLOOR)        # underside of the carrier floor
    return dict(
        arm_t=arm_t,
        floor_y=floor_y,
        horn_y=s.boss_h + RUN + arm_t / 2,       # centre of the +Y (horn) arm
        pin_y=floor_y - RUN - arm_t / 2,         # centre of the -Y (bearing) arm
        ear_y=-(s.case_h - s.ear_z),             # underside of the mounting ears
        side_z=s.case_w / 2 + CLEAR + WALL,      # outer face of the carrier walls
    )


def joint_width(s: ServoSpec):
    p = planes(s)
    return (p["horn_y"] + p["arm_t"] / 2) - (p["pin_y"] - p["arm_t"] / 2)


# --- the servo as a negative volume -------------------------------------------

def servo_negative(s: ServoSpec, clear=CLEAR):
    xl, xh = -s.shaft_from_end, s.shaft_to_far_end
    p = planes(s)
    case = box((xh - xl + 2 * clear, s.case_h + 2 * clear, s.case_w + 2 * clear),
               ((xl + xh) / 2, -s.case_h / 2, 0))
    ears = box((s.ear_span + 2 * clear, s.ear_t + 2 * clear, s.case_w + 2 * clear),
               ((xl + xh) / 2, p["ear_y"] + s.ear_t / 2, 0))
    boss = cyl(s.boss_d + 2 * clear, s.boss_h + 4.0, (0, s.boss_h / 2 + 1.0, 0), "y")
    cable = box((8.0, 7.0, s.cable_w), (xh - 1.0, -s.case_h / 2, 0))
    return union(case, ears, boss, cable)


# --- child half: the yoke ------------------------------------------------------

def yoke(s: ServoSpec, drop: float, arm_r: float = 6.0):
    """Straddles the parent's servo. Horn on +Y, MR63ZZ on -Y. NO WEB.

    The obvious way to close a U is a plate across the open end.  Here that
    plate is exactly wrong: it spans the full joint width in Y, so it lies in
    the same Y band as the NEXT link's yoke arms, and the two foul each other
    a few degrees off the home pose.  Two attempts at this file died that way.

    So the arms are left open and are tied to their own carrier by short tabs
    instead (see `tabs()`).  A tab only spans Y from the carrier face out to
    the arm's inner face -- it never enters the band the child's arms sweep in.

    `arm_r` is kept just big enough for the horn recess.  It is also what
    decides how far the disc pokes up past its own joint axis, and therefore
    how close the parent's carrier can sit.
    """
    p = planes(s)
    at = p["arm_t"]

    def arm(y):
        return union(cyl(2 * arm_r, at, (0, y, 0), "y"),
                     box((2 * arm_r, at, drop), (0, y, -drop / 2)))

    body = union(arm(p["horn_y"]), arm(p["pin_y"]))

    horn_recess = cyl(s.boss_d + 3.4, 1.8, (0, p["horn_y"] - at / 2 + 0.9, 0), "y")
    horn_screws = union(*[cyl(M2_TAP, 10.0, (dx, p["horn_y"], 0), "y")
                          for dx in (-4.2, 0.0, 4.2)])
    seat = cyl(BEARING_OD + 0.05, BEARING_T,
               (0, p["pin_y"] + at / 2 - BEARING_T / 2, 0), "y")
    thru = cyl(PIVOT_D + 0.4, 16.0, (0, p["pin_y"], 0), "y")
    return diff(body, horn_recess, horn_screws, seat, thru)


def arm_r(s: ServoSpec) -> float:
    """Yoke disc radius: just big enough for the horn recess, and no bigger.

    It has to stay under carrier_reach(s), because the slice between the two is
    the only place a collar can tie the arms to the carrier without fouling the
    child (see parts.link). A fixed 6.0 mm broke the micro servo, whose carrier
    only reaches 6.05.
    """
    return max(s.boss_d / 2.0 + 2.2, 4.6)


# --- parent half: the carrier --------------------------------------------------

def carrier(s: ServoSpec):
    """Servo pocket, open at +Y so the servo drops in and prints without support.

    The ears land on ledges and take M2 self-tappers straight down into the
    plastic. The 3 mm pin the child's bearing runs on passes through the floor
    on the joint axis.
    """
    p = planes(s)
    length = s.ear_span + 2 * WALL
    xc = (s.case_l / 2) - s.shaft_from_end
    height = -p["floor_y"]                       # floor underside up to y = 0

    shell = box((length, height, 2 * p["side_z"]), (xc, -height / 2, 0))
    # Local thickening around the pin hole: the floor alone is 1.2 mm and this
    # is the only bearing surface carrying the leg.
    pin_pad = cyl(PIVOT_D + 2 * WALL + 2.0, FLOOR + 2.0,
                  (0, p["floor_y"] + (FLOOR + 2.0) / 2, 0), "y")
    body = union(shell, pin_pad)

    cuts = [servo_negative(s)]
    # Ear screws: down through the ear, into ~4 mm of plastic below it.
    xh = (s.case_l / 2) - s.shaft_from_end
    for dx in (-s.hole_pitch / 2, s.hole_pitch / 2):
        cuts.append(cyl(M2_TAP, 9.0, (xh + dx, p["ear_y"] - 3.0, 0), "y"))
    # Slide the servo in from +Y: open the whole top face above the ears.
    cuts.append(box((length + 2, 8.0, 2 * p["side_z"] + 2), (xc, 4.0, 0)))
    # Pin hole on the joint axis.
    cuts.append(cyl(PIVOT_D + 0.15, 30.0, (0, p["floor_y"], 0), "y"))
    # Lighten the side walls, keeping material around the ear screws.
    cuts.append(box((s.ear_span - 16.0, s.case_h - 8.0, 2 * p["side_z"] + 4),
                    (xc + 4.0, -s.case_h / 2 - 1.0, 0)))
    return diff(body, *cuts)


def carrier_reach(s: ServoSpec):
    """How far the carrier extends either side of the joint axis, along the link."""
    return planes(s)["side_z"]
