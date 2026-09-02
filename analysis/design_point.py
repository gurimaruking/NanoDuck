#!/usr/bin/env python3
"""Clean-sheet NanoDuck sizing: mass and leg length as independent variables.

`servo_sizing.py` asked "how far can MicroDuck be shrunk", which forces one
scale factor onto everything at once.  That was the right question while the
design was a copy of MicroDuck.  It is the wrong question now: a NanoDuck built
around the MG90S need not be geometrically similar to anything, and the single-s
framing hides the fact that the two things the joints actually care about --
how heavy the robot is and how long its legs are -- trade against each other.

Design variables:

    M        total robot mass [kg]     -- set by the BOM
    L_leg    thigh + shin [m]          -- set by the mechanical layout
    servo    assigned per joint        -- they need not all be the same part

Demand transfers off MicroDuck's measured per-joint gait as

    tau  =  tau_micro * (M / M_micro) * (L_leg / L_micro)      (tau ~ M g L)
    w    =  w_micro   * sqrt(L_micro / L_leg)                  (pendulum period)
    P    ~  tau * w   ~  M * sqrt(L_leg)

**Mass is a linear lever; leg length is only a square-root lever.**  Halving the
legs buys 29% of the power budget; halving the mass buys 50%.  That is why the
design below spends its effort on grams rather than millimetres.

Two honest caveats:

* The demand comes from MicroDuck's crouched, athletic gait.  A policy trained
  against the MG90S envelope will find a straighter-legged, shorter-stepped
  walk that needs materially less knee torque, because knee torque in stance is
  M*g*(horizontal offset from the knee axis) and a straighter leg shrinks that
  offset.  So these numbers are an upper bound on a *like-for-like* gait, not a
  verdict.  The verdict comes from retraining with sim/mg90s.py in the loop.
* NanoDuck drops hip_yaw and two head DOF, and demand measured on a 14-DOF robot
  does not transfer exactly to a 10-DOF one.  Close enough to choose parts.
"""

from __future__ import annotations

import numpy as np

Z = np.load("microduck_demand.npz", allow_pickle=True)
NAMES = [str(x) for x in Z["names"]]
TAU = np.abs(Z["tau"])
W = Z["w"]
SGN = np.sign(Z["tau"])
M_MICRO = float(Z["mass_kg"])
L_MICRO = 0.0916                 # thigh 42.2 + shin 49.4 mm, from the MJCF
QUANTILE = 99.0

# --- Servo envelopes: tau_max(w) = min(cap, tau_stall * (1 - w/w_noload)) -----
# MG90S/MG92B figures are catalogue values for Tower Pro parts.  MG92B in
# particular is quoted inconsistently across vendors; the conservative end is
# used here and it MUST be bench-measured before the design leans on it.
SERVOS = {
    "MG90S@6V": dict(tau=0.216, w=13.09, m=0.0134, jpy=350),
    "MG90S@4V8": dict(tau=0.177, w=10.47, m=0.0134, jpy=350),
    "MG92B@6V": dict(tau=0.310, w=13.09, m=0.0138, jpy=900),   # PROVISIONAL
    "micro3.7g": dict(tau=0.060, w=10.47, m=0.0037, jpy=600),
    "XL330@7V4": dict(tau=0.963, w=20.22, m=0.018, cap=0.641, jpy=4500),
}


def envelope(sv, w):
    t = sv["tau"] * np.clip(1.0 - w / sv["w"], 0.0, None)
    return np.minimum(t, sv["cap"]) if "cap" in sv else t


def demand(joint, mass, leg):
    """Paired (tau, signed w) samples transferred to this mass and leg length."""
    i = NAMES.index(joint)
    tau = TAU[:, i] * (mass / M_MICRO) * (leg / L_MICRO)
    w = W[:, i] * np.sqrt(L_MICRO / leg)
    aligned = np.abs(w) * np.where(np.sign(w) * SGN[:, i] >= 0, 1.0, -1.0)
    return tau, aligned


def utilisation(joint, mass, leg, sv):
    tau, w = demand(joint, mass, leg)
    return float(np.percentile(tau / np.maximum(envelope(sv, w), 1e-9), QUANTILE))


def worst(assign, mass, leg):
    return max((utilisation(j, mass, leg, SERVOS[s]), j) for j, s in assign.items())


def max_mass(assign, leg, target=1.0):
    grid = np.linspace(0.05, 1.3, 251)
    ok = [m for m in grid if worst(assign, m, leg)[0] <= target]
    return max(ok) if ok else None


# --- NanoDuck v2 --------------------------------------------------------------
# 10 DOF.  Per leg: hip_roll, hip_pitch, knee, ankle.  Plus neck_pitch and
# head_yaw on 3.7 g micro servos.
#
# hip_yaw is gone: a 4-DOF leg is the classic minimum for walking and RL can
# turn by asymmetric stepping.  head_pitch and head_roll are gone: they sat in
# the tightest cluster on the robot (24.6 mm of chain per servo) and contribute
# nothing to walking.  The head keeps two DOF because a duck that cannot move
# its head is furniture.
#
# The knees get MG92B rather than MG90S.  Same 22.8 x 12.2 footprint, +0.4 g,
# ~45% more stall torque -- and the knee is the only joint that needs it.
# Spending 550 JPY x2 there is far cheaper than shaving 30 g off the airframe.
ASSIGN = {
    "left_hip_roll": "MG90S@6V", "right_hip_roll": "MG90S@6V",
    "left_hip_pitch": "MG90S@6V", "right_hip_pitch": "MG90S@6V",
    "left_knee": "MG92B@6V", "right_knee": "MG92B@6V",
    "left_ankle": "MG90S@6V", "right_ankle": "MG90S@6V",
    "neck_pitch": "micro3.7g", "head_yaw": "micro3.7g",
}
ALL_MG90S = {j: ("micro3.7g" if s == "micro3.7g" else "MG90S@6V") for j, s in ASSIGN.items()}

LEG_MM = 50.0     # thigh 25 + shin 25; see cad/build_nanoduck.py
# Printed masses are now MEASURED, not estimated: cad/parts.py generates the
# real parts and reports their volume, and these are those volumes at PETG
# density and 35% infill (cosmetic shells count at 90%, being thin-wall CAD).
# The round numbers this replaced were badly optimistic -- 33 g of structure
# turned into 65 g the moment the geometry existed.
BOM = [
    ("MG90S x6 (hip roll/pitch, ankles)", 6 * 0.0134),
    ("MG92B x2 (knees)", 2 * 0.0138),
    ("micro servo x2 (neck, head yaw)", 2 * 0.0037),
    ("2S 450 mAh LiHV", 0.028),
    ("control PCB (ESP32-S3-MINI + driver + ADC + BEC)", 0.012),
    ("wiring", 0.008),
    ("trunk chassis (estimate -- not yet drawn)", 0.012),
    ("printed leg structure x2 (yoke, thigh, shin, foot)", 2 * 0.01037),
    ("printed neck link", 0.00167),
    ("cosmetic head: shells, face, jaw, eyes (MicroDuck at 0.62)", 0.02108),
    ("cosmetic body panels (MicroDuck at 0.85)", 0.0159),
    ("cosmetic soles x2 (MicroDuck at 0.75)", 2 * 0.00301),
]


def table(assign, mass, leg, title):
    print("\n%s   (%.0f g, %.0f mm legs)" % (title, 1000 * mass, 1000 * leg))
    print("   %-16s %-10s %8s %10s %8s" % ("joint", "servo", "tau", "speed", "used"))
    rows = sorted(((utilisation(j, mass, leg, SERVOS[s]), j, s) for j, s in assign.items()),
                  reverse=True)
    for u, j, s in rows:
        tau, w = demand(j, mass, leg)
        print("   %-16s %-10s %6.3f Nm %6.2f rad/s %6.0f%%%s"
              % (j, s, np.percentile(tau, QUANTILE), np.percentile(np.abs(w), QUANTILE),
                 100 * u, "   <-- binding" if u == rows[0][0] else ""))
    return rows[0]


def main() -> None:
    print("MicroDuck: %.0f g, %.1f mm legs. Its own knee sits at %.0f%% of the XL330"
          % (1000 * M_MICRO, 1000 * L_MICRO,
             100 * utilisation("left_knee", M_MICRO, L_MICRO, SERVOS["XL330@7V4"])))
    print("envelope, so ~75-80%% is the utilisation a working robot actually lives at.\n")

    print("=" * 78)
    print("HOW MUCH ROBOT CAN THE LEGS CARRY?   (worst joint <= 100%)")
    print("=" * 78)
    print("%-12s %22s %22s" % ("leg length", "all MG90S", "MG92B knees"))
    for leg_mm in (50, 60, 70, 80, 90, 100):
        a = max_mass(ALL_MG90S, leg_mm / 1000.0)
        b = max_mass(ASSIGN, leg_mm / 1000.0)
        print("%9d mm %19s   %19s"
              % (leg_mm,
                 "-" if a is None else "%.0f g" % (1000 * a),
                 "-" if b is None else "%.0f g" % (1000 * b)))
    print("\nUpgrading two servos moves the mass budget more than any plausible")
    print("airframe diet would.  That is where the money goes.")

    total = sum(m for _, m in BOM)
    print("\n" + "=" * 78)
    print("NANODUCK v2 BOM MASS")
    print("=" * 78)
    for label, m in BOM:
        print("   %-50s %6.1f g" % (label, 1000 * m))
    print("   %-50s %6.1f g" % ("TOTAL", 1000 * total))

    leg = LEG_MM / 1000.0
    u, j = worst(ASSIGN, total, leg)
    table(ASSIGN, total, leg, "per-joint check")
    budget = max_mass(ASSIGN, leg)
    print("\n   worst joint: %s at %.0f%%   |   mass budget %.0f g, margin %+.0f g"
          % (j, 100 * u, 1000 * budget, 1000 * (budget - total)))

    print("\n" + "=" * 78)
    print("SENSITIVITY")
    print("=" * 78)
    for extra in (0, 10, 20, 30):
        m = total + extra / 1000.0
        uu, jj = worst(ASSIGN, m, leg)
        print("   +%2d g (%3.0f g)  worst %-16s %3.0f%%%s"
              % (extra, 1000 * m, jj, 100 * uu, "   <-- over" if uu > 1 else ""))
    for leg_mm in (60, 80):
        uu, jj = worst(ASSIGN, total, leg_mm / 1000.0)
        print("   legs %d mm     worst %-16s %3.0f%%" % (leg_mm, jj, 100 * uu))
    print("\n   A 2S 850 mAh pack instead of 450 mAh is +20 g: battery capacity is a")
    print("   joint-torque decision on this robot, not a runtime decision.")


if __name__ == "__main__":
    main()
