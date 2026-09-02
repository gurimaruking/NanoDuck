#!/usr/bin/env python3
"""Generate the NanoDuck MJCF from the design parameters.

    python build_nanoduck.py

Writes `sim/nanoduck.xml` (the robot) and `sim/scene_nanoduck.xml` (floor,
lighting, keyframes).

Why generated rather than hand-written
--------------------------------------
Every number in this file traces back to a decision recorded elsewhere in the
repo -- servo masses to bom/bom.csv, leg length and total mass to
analysis/design_point.py, the servo orientation to cad/check_servo_fit.py.
Generating the XML keeps those links live: change LEG or a servo mass here and
the model, the keyframe and the actuator limits all move together.

MicroDuck's own MJCF is an Onshape export full of quaternions like
`0.5 -0.5 0.5 -0.5`, because the CAD frames are whatever SolidWorks made them.
This one is built the other way round: every body frame is axis-aligned with
the world in the home pose, so joint axes are literally "1 0 0" / "0 1 0" /
"0 0 1" and the model can be read.

Layout
------
10 DOF, in the left-leg / head / right-leg order MicroDuck uses, so the
observation and action vectors keep the same shape of layout:

    0-3   left  hip_roll, hip_pitch, knee, ankle
    4-5   neck_pitch, head_yaw
    6-9   right hip_roll, hip_pitch, knee, ankle

Servos are mounted TRANSVERSE: the 12.2 mm thin axis runs along the leg and the
32.2 mm long axis sticks out fore-aft.  That is what makes a 35 mm thigh
possible at all (cad/check_servo_fit.py), and it is why the thighs are visibly
chunky front-to-back.  On a duck that reads as drumsticks.

Each servo's mass sits on the link that CARRIES it, at the joint it DRIVES --
so the knee servo is part of the thigh, sitting at the knee end.  That places
mass distally, which is pessimistic and correct: it is where the real one goes.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "sim"))

MM = 0.001

# --- Servos (bom/bom.csv) -----------------------------------------------------
# box is (fore-aft, along-joint-axis, along-link) for the transverse mounting.
SERVO = {
    "MG90S": dict(mass=0.0134, box=(32.2, 28.5, 12.2), tau=0.216, kp=2.48),
    "MG92B": dict(mass=0.0138, box=(32.2, 30.5, 12.2), tau=0.310, kp=3.55),
    "micro": dict(mass=0.0037, box=(20.0, 20.0, 8.4), tau=0.060, kp=0.69),
}

# --- Geometry [mm] ------------------------------------------------------------
TRUNK = (62.0, 44.0, 34.0)      # electronics box: 2S 450mAh pack + 40x30 PCB,
                                # sized longer than tall so it reads as a body
HIP_Y = 20.0                    # half stance width at the hip
HIP_ROLL_TO_PITCH = 22.0        # roll and pitch axes are near-coincident
# LEG = 50 mm, down from a first cut at 70 mm.  Shorter legs are better on every
# axis that matters here: the mass budget rises from 245 g to 315 g
# (analysis/design_point.py), the CoM drops, and the proportions stop looking
# like a robot on stilts and start looking like a duck.  The cost is stride
# length and ground clearance, which a 170 mm robot indoors can afford.
# 25 mm still clears the 20.2 mm a transverse servo needs (cad/check_servo_fit.py).
THIGH = 25.0                    # hip_pitch -> knee
SHIN = 25.0                     # knee -> ankle
ANKLE_TO_SOLE = 12.0
FOOT = (42.0, 26.0, 4.0)        # ducks have big webbed feet, and a wide sole is
                                # free stability on a robot with no ankle roll
FOOT_X = 3.0                    # sole offset ahead of the ankle; centres the
                                # support polygon on the CoM (see check_static_balance)
NECK_X, NECK_Z = 16.0, 14.0     # neck_pitch axis, well forward on the trunk lid
NECK_LEN = 20.0                 # neck_pitch -> head_yaw
HEAD = (34.0, 26.0, 24.0)
HEAD_Z = 12.0
BEAK = (22.0, 13.0, 4.0)       # flat and forward, the one unmistakably duck part

# --- Masses [kg] not accounted for by a servo box -----------------------------
# These are no longer estimates.  cad/parts.py generates the real printed parts
# and reports their volume; these are those volumes at PETG density and 35%
# infill (cosmetic shells are thin-walled CAD and print near solid, so they are
# counted at 90%).  Run `python parts.py --check` to regenerate the numbers.
#
# The round-number guesses this replaced were badly optimistic: 33 g of
# structure became 65 g once the geometry existed, mostly because MicroDuck's
# shells are bigger relative to a 162 mm robot than they look.
M_BATTERY = 0.028
M_PCB = 0.012
M_WIRING = 0.008
M_CHASSIS = 0.012               # trunk frame -- still an estimate, not yet drawn
M_TRUNK_SHELL = 0.0159          # cosmetic left_shell + right_shell at 0.85
M_HIP_YOKE = 0.00289
M_THIGH = 0.00314
M_SHIN = 0.00297
M_FOOT_MOUNT = 0.00137
M_SOLE = 0.00301                # cosmetic sole at 0.75
M_HEAD = 0.02108                # top + bottom shell + face + jaw + 2 eyes at 0.62
M_NECK_LINK = 0.00167

# --- Home pose [rad] ----------------------------------------------------------
# A symmetric crouch, derived rather than copied.  For a two-link leg with equal
# segments, the ankle sits directly under the hip when
#
#     THIGH*sin(hp) + SHIN*sin(hp + knee) = 0   ->   knee = -2*hp
#
# and the sole is horizontal when the joint angles sum to zero, i.e.
#
#     ankle = -(hp + knee) = hp
#
# The first attempt at this file copied MicroDuck's home pose numerically
# (hip -26.2 deg, knee ~0, ankle +26.2 deg).  That is a *leaning* leg, and it
# only balances on MicroDuck because its three-servo hip cluster carries the
# foot forward to meet it.  Dropped into a plain two-axis hip it puts the feet
# 31 mm ahead of the centre of mass, and the robot sits down backwards -- which
# is exactly what the first stand test did.
#
# The crouch is kept shallow (11.5 deg).  Knee torque in stance is mass times
# the horizontal offset from the knee axis, so a straight leg is cheapest; but
# a perfectly straight knee is a singularity it cannot push out of.  20 deg of
# knee bend costs ~7 mm of offset, about 13 mNm -- affordable, and it leaves the
# joint somewhere useful.
# Zero, not MicroDuck's 5 deg. There is no ankle ROLL joint on this leg, so any
# hip roll offset stands the robot on the outside edges of its soles. MicroDuck
# gets away with it on a compliant PU sole; a printed one would just rock.
HIP_ROLL_0 = 0.0
HIP_PITCH_0 = 0.20              # 11.5 deg
KNEE_0 = -2.0 * HIP_PITCH_0     # -22.9 deg, keeps the ankle under the hip
ANKLE_0 = HIP_PITCH_0           # levels the sole
NECK_PITCH_0 = 0.35             # leans the neck forward: a duck, not a meerkat

# Leg pitch limits are MEASURED, not chosen: cad/parts.py sweeps each printed
# joint and finds where the child's yoke fouls the parent's carrier
# (`python parts.py --check`). They are asymmetric because the servo case
# sticks out one way -- the drumstick -- and the knee's carrier is mirrored so
# its wide side faces the direction the knee actually bends.
#
# The ankle is the tight one at 32 deg of total travel, and it is the clearest
# thing to fix next in the mechanics.
HIP_PITCH_RANGE = (-0.567, 1.745)     # -32.5 .. +100.0 deg
KNEE_RANGE = (-1.789, 0.567)          # -102.5 .. +32.5 deg
ANKLE_RANGE = (-0.175, 0.393)         # -10.0 .. +22.5 deg

JOINTS = [
    # name,             servo,   range (rad),      home
    ("left_hip_roll",   "MG90S", (-0.384, 0.384), -HIP_ROLL_0),
    ("left_hip_pitch",  "MG90S", HIP_PITCH_RANGE, HIP_PITCH_0),
    ("left_knee",       "MG92B", KNEE_RANGE,      KNEE_0),
    ("left_ankle",      "MG90S", ANKLE_RANGE,     ANKLE_0),
    ("neck_pitch",      "micro", (-1.571, 1.047), NECK_PITCH_0),
    ("head_yaw",        "micro", (-2.967, 2.967), 0.0),
    ("right_hip_roll",  "MG90S", (-0.384, 0.384), HIP_ROLL_0),
    ("right_hip_pitch", "MG90S", HIP_PITCH_RANGE, HIP_PITCH_0),
    ("right_knee",      "MG92B", KNEE_RANGE,      KNEE_0),
    ("right_ankle",     "MG90S", ANKLE_RANGE,     ANKLE_0),
]


def m(*v):
    """mm -> m, formatted for XML."""
    return " ".join("%.5f" % (x * MM) for x in v)


def half(box):
    return m(*(x / 2.0 for x in box))


def servo_geom(name, kind, pos_mm, group=3):
    s = SERVO[kind]
    return ('        <geom name="%s" type="box" size="%s" pos="%s" mass="%.5f" '
            'class="viz" rgba="0.15 0.15 0.17 1"/>\n'
            % (name, half(s["box"]), m(*pos_mm), s["mass"]))


def link_geom(name, size_mm, pos_mm, mass, rgba="0.85 0.72 0.25 1"):
    return ('        <geom name="%s" type="box" size="%s" pos="%s" mass="%.5f" '
            'class="viz" rgba="%s"/>\n'
            % (name, half(size_mm), m(*pos_mm), mass, rgba))


def leg(side, sign):
    """One leg subtree. `sign` is +1 for left (+Y), -1 for right."""
    roll0 = -HIP_ROLL_0 if side == "left" else HIP_ROLL_0
    x = []
    a = x.append
    a('      <body name="%s_hip" pos="%s">\n' % (side, m(0, sign * HIP_Y, -TRUNK[2] / 2)))
    a('        <joint name="%s_hip_roll" axis="1 0 0" range="%.3f %.3f"/>\n'
      % (side, -0.384, 0.384))
    # The hip_pitch servo rides on the roll link, at the pitch axis.
    a(servo_geom("%s_hip_pitch_servo" % side, "MG90S", (-6, 0, -HIP_ROLL_TO_PITCH)))
    a(link_geom("%s_hip_bracket" % side, (14, 20, HIP_ROLL_TO_PITCH),
                (0, 0, -HIP_ROLL_TO_PITCH / 2), M_HIP_YOKE))

    a('        <body name="%s_thigh" pos="%s">\n' % (side, m(0, 0, -HIP_ROLL_TO_PITCH)))
    a('          <joint name="%s_hip_pitch" axis="0 1 0" range="%.3f %.3f"/>\n'
      % (side, -1.571, 1.571))
    a(servo_geom("%s_knee_servo" % side, "MG92B", (-6, 0, -THIGH)))
    a(link_geom("%s_thigh_link" % side, (12, 16, THIGH), (0, 0, -THIGH / 2), M_THIGH))

    a('          <body name="%s_shin" pos="%s">\n' % (side, m(0, 0, -THIGH)))
    a('            <joint name="%s_knee" axis="0 1 0" range="%.3f %.3f"/>\n'
      % (side, -1.571, 1.571))
    a(servo_geom("%s_ankle_servo" % side, "MG90S", (-6, 0, -SHIN)))
    a(link_geom("%s_shin_link" % side, (10, 14, SHIN), (0, 0, -SHIN / 2), M_SHIN))

    a('            <body name="%s_foot" pos="%s">\n' % (side, m(0, 0, -SHIN)))
    a('              <joint name="%s_ankle" axis="0 1 0" range="%.3f %.3f"/>\n'
      % (side, -1.571, 1.571))
    a(link_geom("%s_ankle_bracket" % side, (12, 16, ANKLE_TO_SOLE),
                (0, 0, -ANKLE_TO_SOLE / 2), M_FOOT_MOUNT))
    a('              <geom name="%s_sole" type="box" size="%s" pos="%s" mass="%.5f" '
      'class="collision" rgba="0.95 0.55 0.10 1"/>\n'
      % (side, half(FOOT), m(FOOT_X, 0, -ANKLE_TO_SOLE - FOOT[2] / 2), M_SOLE))
    a('              <site name="%s_foot" group="3" pos="%s"/>\n'
      % (side, m(FOOT_X, 0, -ANKLE_TO_SOLE - FOOT[2])))
    a('            </body>\n')
    a('          </body>\n')
    a('        </body>\n')
    a('      </body>\n')
    return "".join(x), roll0


def build_robot() -> str:
    x = []
    a = x.append
    a('<mujoco model="nanoduck">\n')
    a('  <compiler angle="radian" autolimits="true"/>\n')
    a('  <option timestep="0.005" iterations="10" ls_iterations="10"/>\n\n')
    a('  <default>\n')
    a('    <!-- Only the soles collide, as in MicroDuck robot_walk.xml: falling is\n')
    a('         cheap during velocity training and the contacts cost solver time. -->\n')
    a('    <default class="viz">\n')
    a('      <geom contype="0" conaffinity="0" group="2"/>\n')
    a('    </default>\n')
    a('    <default class="collision">\n')
    a('      <geom contype="1" conaffinity="1" group="3" condim="3" friction="1.0 0.005 0.0001"/>\n')
    a('    </default>\n')
    a('    <!-- Position gains are the MG90S/MG92B closed-loop stiffness. They are\n')
    a('         overwritten anyway when sim/mg90s.py ServoBank.install() takes over,\n')
    a('         but they make the bare model behave sensibly on its own. -->\n')
    a('    <joint damping="0.005" frictionloss="0.012" armature="0.003"/>\n')
    a('  </default>\n\n')

    a('  <worldbody>\n')
    a('    <body name="trunk_base" pos="0 0 %.4f">\n' % (trunk_z() * MM))
    a('      <freejoint name="root"/>\n')
    a('      <site name="imu" group="3" pos="0 0 0"/>\n')
    # Trunk contents. Battery low and central; PCB above it.
    a(link_geom("trunk_shell", TRUNK, (0, 0, 0),
                M_CHASSIS + M_TRUNK_SHELL + M_WIRING, "0.95 0.93 0.86 1"))
    a('      <geom name="battery" type="box" size="%s" pos="%s" mass="%.5f" class="viz" '
      'rgba="0.2 0.3 0.6 1"/>\n' % (half((45, 20, 11)), m(0, 0, -8), M_BATTERY))
    a('      <geom name="pcb" type="box" size="%s" pos="%s" mass="%.5f" class="viz" '
      'rgba="0.1 0.4 0.2 1"/>\n' % (half((40, 30, 5)), m(0, 0, 6), M_PCB))
    # The hip_roll servos are bolted into the trunk and drive the hip links.
    a(servo_geom("left_hip_roll_servo", "MG90S", (0, HIP_Y, -TRUNK[2] / 2 + 6)))
    a(servo_geom("right_hip_roll_servo", "MG90S", (0, -HIP_Y, -TRUNK[2] / 2 + 6)))
    a(servo_geom("neck_pitch_servo", "micro", (NECK_X, 0, TRUNK[2] / 2 - 5)))
    a('\n')

    left, _ = leg("left", +1)
    right, _ = leg("right", -1)
    a(left)

    # Head
    a('      <body name="neck" pos="%s">\n' % m(NECK_X, 0, TRUNK[2] / 2))
    a('        <joint name="neck_pitch" axis="0 1 0" range="-1.571 1.047"/>\n')
    a(servo_geom("head_yaw_servo", "micro", (0, 0, NECK_LEN)))
    a(link_geom("neck_link", (10, 12, NECK_LEN), (0, 0, NECK_LEN / 2), M_NECK_LINK))
    a('        <body name="head" pos="%s">\n' % m(0, 0, NECK_LEN))
    a('          <joint name="head_yaw" axis="0 0 1" range="-2.967 2.967"/>\n')
    a(link_geom("head_shell", HEAD, (2, 0, HEAD_Z), M_HEAD, "0.95 0.93 0.86 1"))
    a(link_geom("beak", BEAK, (22, 0, HEAD_Z - 5), 0.0005, "0.95 0.65 0.10 1"))
    a('          <site name="head_camera" group="3" pos="%s"/>\n' % m(16, 0, HEAD_Z + 4))
    a('          <site name="mouth_tip" group="3" pos="%s"/>\n' % m(30, 0, HEAD_Z - 5))
    a('        </body>\n')
    a('      </body>\n')

    a(right)
    a('    </body>\n')
    a('  </worldbody>\n\n')

    # Actuators, in JOINTS order -> this defines the action vector.
    a('  <actuator>\n')
    for name, kind, (lo, hi), _ in JOINTS:
        s = SERVO[kind]
        a('    <position name="%s" joint="%s" kp="%.3f" kv="0" '
          'forcerange="%.3f %.3f" ctrlrange="%.3f %.3f"/>\n'
          % (name, name, s["kp"], -s["tau"], s["tau"], lo, hi))
    a('  </actuator>\n\n')

    # Sensor names match MicroDuck's so scripts/infer_policy.py works unchanged.
    a('  <sensor>\n')
    a('    <gyro name="imu_ang_vel" site="imu"/>\n')
    a('    <accelerometer name="imu_accel" site="imu"/>\n')
    a('    <velocimeter name="imu_lin_vel" site="imu"/>\n')
    a('    <framequat name="orientation" objtype="site" objname="imu"/>\n')
    a('  </sensor>\n')
    a('</mujoco>\n')
    return "".join(x)


def trunk_z() -> float:
    """Height of the trunk centre above the ground in the home pose [mm]."""
    import math
    drop = THIGH * math.cos(HIP_PITCH_0) + SHIN * math.cos(HIP_PITCH_0 + KNEE_0)
    return ANKLE_TO_SOLE + FOOT[2] + drop + HIP_ROLL_TO_PITCH + TRUNK[2] / 2


def check_static_balance(scene_path: str) -> None:
    """Is the centre of mass inside the support polygon in the home pose?

    Cheap, and it catches the single most common way a hand-built biped model is
    wrong: a pose that looks plausible but stands the robot on the edge of its
    own feet.  Nothing downstream -- not the sim, not the policy -- can rescue a
    home pose that is statically unbalanced, so it is checked at build time.
    """
    import mujoco
    import numpy as np

    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    com = d.subtree_com[1].copy()
    soles = [g for g in range(m.ngeom)
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_sole")]
    corners = []
    for g in soles:
        c, half = d.geom_xpos[g], m.geom_size[g]
        corners += [c[:2] + np.array([sx * half[0], sy * half[1]])
                    for sx in (-1, 1) for sy in (-1, 1)]
    corners = np.array(corners)
    lo, hi = corners.min(0), corners.max(0)
    clearance = d.geom_xpos[soles[0]][2] - m.geom_size[soles[0]][2]

    print("\nstatic balance in the home pose:")
    print("   CoM              x %+6.1f  y %+6.1f mm" % (1000 * com[0], 1000 * com[1]))
    print("   support polygon  x [%+6.1f, %+6.1f]  y [%+6.1f, %+6.1f] mm"
          % (1000 * lo[0], 1000 * hi[0], 1000 * lo[1], 1000 * hi[1]))
    mx = min(com[0] - lo[0], hi[0] - com[0])
    my = min(com[1] - lo[1], hi[1] - com[1])
    print("   margin           x %+6.1f  y %+6.1f mm  %s"
          % (1000 * mx, 1000 * my, "OK" if min(mx, my) > 0 else "<-- UNBALANCED"))
    print("   sole clearance   %+.2f mm relative to the floor" % (1000 * clearance))
    print("   total mass       %.1f g" % (1000 * m.body_subtreemass[1]))


def build_scene() -> str:
    home = " ".join("%.6f" % h for _, _, _, h in JOINTS)
    qpos = "0 0 %.4f 1 0 0 0 %s" % (trunk_z() * MM, home)
    return """<mujoco model="nanoduck_scene">
  <include file="nanoduck.xml"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <!-- offwidth/offheight size the offscreen framebuffer; the 640x480
         default is too small for a readable render of a tall biped. -->
    <global azimuth="160" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"/>
  </worldbody>

  <keyframe>
    <key name="STAND" qpos="%s" ctrl="%s"/>
  </keyframe>
</mujoco>
""" % (qpos, home)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for name, text in (("nanoduck.xml", build_robot()),
                       ("scene_nanoduck.xml", build_scene())):
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s (%d bytes)" % (path, len(text)))
    print("\nhome-pose trunk height: %.1f mm" % trunk_z())
    print("leg (thigh + shin):     %.1f mm" % (THIGH + SHIN))
    print("DOF:                    %d" % len(JOINTS))
    check_static_balance(os.path.join(OUT, "scene_nanoduck.xml"))


if __name__ == "__main__":
    main()
