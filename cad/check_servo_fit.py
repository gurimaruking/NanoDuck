#!/usr/bin/env python3
"""Can MG90S servos be packed into a half-size MicroDuck?

The torque analysis (analysis/servo_sizing.py) says "make it smaller".  Geometry
says the opposite, and the two have to be reconciled before any part is bought.

The trap: an MG90S is NOT a small servo.  It is a *thin* one.

    XL330-M288 (from xl330.stl)   34.0 x 29.0 x 20.0 mm, 18.0 g
    MG90S (incl. mounting ears)   32.2 x 28.5 x 12.2 mm, 13.4 g
    per-axis ratio, sorted:        0.947   0.983   0.610

Same length, same height, only thinner.  "45% of the volume" is true and
completely misleading when the binding constraint is length along the chain.

Why the obvious test is the wrong test
--------------------------------------
"Is each link longer than a servo?" fails here, because MicroDuck's multi-DOF
clusters already interleave: the hip packs three XL330s into a 71 mm chain, i.e.
23.7 mm of chain per 34 mm servo.  The servos overlap by 43% and the design
works anyway.  So the meaningful number is the PACKING RATIO

    packing ratio = servo length / (chain length per servo)

with 1.0 meaning servos sit end to end and >1 meaning they interleave.  What
matters is not NanoDuck's absolute ratio but how it compares to the ratio
MicroDuck already demonstrates: that is the density the mechanical design would
have to beat.
"""

from __future__ import annotations

import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(HERE, "microduck_src", "robot_walk.xml")
HEIGHT_MM = 250.0

XL330_LEN_MM = 34.0
MG90S_LEN_MM = 32.2       # across the mounting ears; 22.8 mm for the bare case

# Kinematic clusters: the child bodies whose offsets make up each chain, and how
# many servos live in that chain.
CLUSTERS = {
    "hip (yaw+roll+pitch)":  (["yaw2roll", "hip_l", "upper_leg_left"], 3),
    "thigh -> knee":         (["leg"], 1),
    "shin -> ankle":         (["ankle_left"], 1),
    "neck (pitch)":          (["neck"], 1),
    "head (pitch+yaw+roll)": (["neck_pitch", "yaw_roll_motion"], 3),
}

TORQUE_CEILING = 0.56     # MG90S @6.0V, from analysis/servo_sizing.py


def link_lengths() -> dict[str, float]:
    x = open(MJCF, encoding="utf-8").read()
    return {m.group(1): 1000.0 * float(np.linalg.norm(
        np.array([float(v) for v in m.group(2).split()])))
        for m in re.finditer(r'<body name="([^"]+)" pos="([^"]+)"', x)}


def main() -> None:
    L = link_lengths()
    print("XL330  %.1f mm long   MG90S  %.1f mm long   (%.0f%% as long)\n"
          % (XL330_LEN_MM, MG90S_LEN_MM, 100 * MG90S_LEN_MM / XL330_LEN_MM))
    print("%-24s %8s %3s %10s %10s %10s"
          % ("cluster", "chain", "n", "mm/servo", "MicroDuck", "NanoDuck"))
    print("%-24s %8s %3s %10s %10s %10s"
          % ("", "[mm]", "", "@ s=1", "ratio", "ratio @ s=%.2f" % TORQUE_CEILING))
    print("-" * 74)
    worst = 0.0
    for label, (bodies, n) in CLUSTERS.items():
        chain = sum(L[b] for b in bodies if b in L)
        if chain <= 0:
            continue
        per = chain / n
        micro = XL330_LEN_MM / per
        nano = MG90S_LEN_MM / (per * TORQUE_CEILING)
        worst = max(worst, nano / micro)
        print("%-24s %8.1f %3d %10.1f %10.2f %10.2f" % (label, chain, n, per, micro, nano))
    print("-" * 74)
    print("\nNanoDuck at s=%.2f would need up to %.1fx the servo packing density that"
          % (TORQUE_CEILING, worst))
    print("MicroDuck already achieves.  That is the blocker, and it is geometric,")
    print("not a torque problem -- no amount of retraining moves it.\n")

    # What scale would keep the density MicroDuck already proves is buildable?
    s_geom = MG90S_LEN_MM / XL330_LEN_MM
    print("Keeping MicroDuck's demonstrated density exactly, MG90S allows only")
    print("   s = %.2f  (%.0f mm robot) -- i.e. almost no shrink at all," % (s_geom, s_geom * HEIGHT_MM))
    print("   because the servo barely got shorter.")
    print("Torque allows only s <= %.2f (%.0f mm).  The two do not overlap.\n"
          % (TORQUE_CEILING, TORQUE_CEILING * HEIGHT_MM))

    print("Ways out, cheapest first:")
    print("  1. FEWER DOF.  The 23.7 mm head cluster (pitch+yaw+roll) and the hip yaw")
    print("     exist for expressiveness, not for walking.  14 -> 10 or 11 DOF frees")
    print("     exactly the tightest chains.  Retraining is already mandatory, so the")
    print("     14-action contract is not a reason to keep them.")
    print("  2. MIXED SERVO SIZES.  Head joints need <=0.10 Nm where the knee needs")
    print("     0.49 Nm (see analysis output).  Put 3.7 g micro servos in the head and")
    print("     neck and keep MG90S for the legs.")
    print("  3. REMOTE ACTUATION.  Move knee/ankle servos into the trunk and drive")
    print("     through linkages, as most sub-200 mm bipeds do.  Frees the leg volume")
    print("     entirely, at the cost of added backlash -- which the sim already models.")


if __name__ == "__main__":
    main()
