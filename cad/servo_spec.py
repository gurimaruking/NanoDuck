"""MG90S / MG92B mechanical interface, in one place.

Every printed part in `parts.py` is cut from these numbers, so if your servos
measure differently you change this file and reprint -- nothing else.

VERIFY THESE WITH CALIPERS BEFORE PRINTING A FULL SET.
These are the commonly published Tower Pro figures, and MG90S clones vary by
several tenths of a millimetre in exactly the dimensions that matter (case
width, ear thickness, hole spacing).  A pocket 0.3 mm too small will not close;
0.5 mm too large and the servo rocks in its mount, which shows up as backlash
the policy was never trained for.

`parts.py --coupon` prints a 12 g test piece carrying one pocket, one pair of
ear holes and one bearing seat.  Print that first.  It takes ten minutes and
saves a whole set.

Coordinates for a servo, as used throughout:
    +X  along the case length (the 22.8 mm dimension)
    +Y  across the case width (the 12.2 mm dimension, the thin one)
    +Z  up the case height, output shaft pointing +Z
    origin at the centre of the OUTPUT SHAFT, on the bottom face of the case
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServoSpec:
    name: str
    case_l: float          # case length, without the mounting ears
    case_w: float          # case width -- the thin axis
    case_h: float          # bottom of case to top of case, no output boss
    ear_span: float        # overall length across both mounting ears
    ear_t: float           # ear thickness
    ear_z: float           # height of the ear underside above the case bottom
    hole_pitch: float      # centre-to-centre of the two mounting holes
    hole_d: float          # mounting hole diameter (M2 clearance)
    shaft_from_end: float  # output shaft axis, measured from the shaft-side end
    boss_d: float          # diameter of the round boss the shaft sits on
    boss_h: float          # boss height above the case top
    shaft_d: float         # spline outside diameter
    horn_screw_d: float    # the self-tapper that holds a horn on
    mass: float            # kg
    cable_w: float         # cable exit width, for the relief slot

    @property
    def shaft_to_far_end(self) -> float:
        return self.case_l - self.shaft_from_end

    @property
    def total_h(self) -> float:
        return self.case_h + self.boss_h


# Tower Pro MG90S.
MG90S = ServoSpec(
    name="MG90S",
    case_l=22.8, case_w=12.2, case_h=22.7,
    ear_span=32.2, ear_t=2.5, ear_z=16.0,
    hole_pitch=28.0, hole_d=2.2,
    shaft_from_end=5.9,
    boss_d=7.4, boss_h=4.0, shaft_d=4.8,
    horn_screw_d=1.8,
    mass=0.0134,
    cable_w=6.0,
)

# MG92B: same envelope, marginally taller. Knees only.
MG92B = ServoSpec(
    name="MG92B",
    case_l=22.8, case_w=12.2, case_h=24.7,
    ear_span=32.2, ear_t=2.5, ear_z=16.0,
    hole_pitch=28.0, hole_d=2.2,
    shaft_from_end=5.9,
    boss_d=7.4, boss_h=4.0, shaft_d=4.8,
    horn_screw_d=1.8,
    mass=0.0138,
    cable_w=6.0,
)

# Generic 3.7 g micro servo (neck pitch, head yaw).
MICRO = ServoSpec(
    name="micro3.7g",
    case_l=20.0, case_w=8.4, case_h=17.0,
    ear_span=27.5, ear_t=2.0, ear_z=11.5,
    hole_pitch=23.5, hole_d=2.2,
    shaft_from_end=5.0,
    boss_d=5.4, boss_h=3.0, shaft_d=3.8,
    horn_screw_d=1.5,
    mass=0.0037,
    cable_w=4.5,
)

SERVOS = {"MG90S": MG90S, "MG92B": MG92B, "micro": MICRO}

# --- Print and hardware settings ---------------------------------------------
# Clearances are for a well-tuned 0.4 mm nozzle on PETG.  PETG swells more than
# PLA, so these are on the generous side; drop CLEAR to 0.15 for PLA.
CLEAR = 0.25            # added to every pocket that has to receive a part
WALL = 1.6              # structural wall thickness: 4 perimeters at 0.4 mm
FLOOR = 1.2             # thickness of a load-bearing floor: 3 layers at 0.4

M2_CLEAR = 2.2          # through-hole for an M2 screw
M2_TAP = 1.7            # hole to drive an M2 self-tapper straight into plastic
M2_HEAD_D = 4.0         # M2 cap-head diameter, for counterbores
M2_NUT_AF = 4.0         # M2 nut across-flats
M2_NUT_T = 1.6          # M2 nut thickness

# MR63ZZ, the idler bearing that sits opposite the horn on every joint.  The
# MG90S output shaft is single-sided, so without this the joint is a cantilever
# on a 4.8 mm plastic spline and the leg wags visibly under load.
BEARING_OD = 6.0
BEARING_ID = 3.0
BEARING_T = 2.5
PIVOT_D = 3.0           # the 3 mm shoulder that runs in the bearing

# PETG. Change if you print in PLA (1.24) or ABS (1.04).
DENSITY = 1.27e-3       # g/mm^3
INFILL = 0.35           # what a 4-perimeter, 35% infill print actually weighs
                        # relative to a solid model of the same shape
