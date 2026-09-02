#!/usr/bin/env python3
"""How small must a MicroDuck get before MG90S servos can drive it?

Input `microduck_demand.npz` is the per-joint (torque, angular velocity) time
series recorded from the shipping walking policy (alpha_walking.onnx) running
in CPU MuJoCo for 30 s at a 0.4 m/s command, first 2 s dropped.

Torque and speed are evaluated as SIMULTANEOUS pairs, not as independent
percentiles: a knee reaches peak torque in stance (slow) and peak speed in
swing (unloaded), so pairing the two worst numbers would reject designs that
are actually fine.

Scaling laws (geometric similarity, same materials).  Mass is split THREE ways,
not two, because two of the three terms do not shrink with the robot:
    length         L         -> s * L
    structure      m_struct  -> s^3 * m_struct   (shells, brackets: volume)
    servos         m_servo   -> FIXED, and set by which servo you buy
    electronics    m_elec    -> FIXED, and set by the BOM (battery dominates)
    joint torque   tau       -> tau * (M(s)/M0) * s        (tau ~ M g L)

Lumping electronics into the s^3 term (an earlier version of this file did)
flatters small scales badly: a 500 mAh LiPo weighs what it weighs.

Gait speed has two defensible conventions and they disagree, so both are run:
    "froude"  omega -> omega / sqrt(s)   Froude-similar gait: the small robot
                                         steps faster (pendulum period ~ sqrt(L/g))
                                         and walks at v = v0*sqrt(s).  Pessimistic.
    "period"  omega -> omega             Same gait period in seconds, shorter
                                         strides.  RL is free to find such a gait,
                                         so this is the optimistic bound.

CAVEAT on the source data: infer_policy.py drives plain MuJoCo position
actuators with a flat force limit, NOT the BAM voltage model used in training.
The recorded torques are therefore what an *ideal* actuator delivered.  The
XL330 column is the calibration: it says how much demand is known to work on
real hardware, and everything else is read relative to it.
"""
import numpy as np

Z = np.load("microduck_demand.npz", allow_pickle=True)
TAU = np.abs(Z["tau"])             # (T, 14) Nm
W = Z["w"]                         # (T, 14) rad/s, signed
SGN = np.sign(Z["tau"])
NAMES = [str(x) for x in Z["names"]]
M0 = float(Z["mass_kg"])
N_SERVO = len(NAMES)
HEIGHT0_MM = 250.0
XL330_MASS = 0.018

# --- MicroDuck mass breakdown -------------------------------------------------
# Not guessed: measured off the CAD meshes shipped in microduck_rl
# (src/mjlab_microduck/robot/microduck/assets/, dimensions in metres).
#   np_f970.stl                     38.6 x 20.6 x 70.8 mm -- despite the name that
#                                   is an NP-F550 footprint to 0.2 mm, so ~76 g
#   pcb__raspberry_pi_zero_2_w.stl  65 x 30 mm  -> Pi Zero 2 W, 11 g
#   elec_rpi_robot_hat_pcb.stl      65 x 30 mm  -> carrier board, ~15 g populated
#   speaker.stl                     35 x 25 x 7 mm, ~8 g
#   xl330.stl                       29 x 20 x 34 mm -- confirms the 18 g servo
# plus camera + lens + ToF (~10 g) and servo wiring (~20 g).
M_ELEC_MICRODUCK = 0.140           # battery + compute + audio + sensors + wiring
M_SERVO_MICRODUCK = N_SERVO * XL330_MASS                       # 0.252 kg
M_STRUCT = M0 - M_SERVO_MICRODUCK - M_ELEC_MICRODUCK            # 0.345 kg

# NanoDuck's electronics budget is a BOM decision, so it is an input here.
# 75 g = ESP32-S3 + PCA9685 + ADC + 6 V BEC on one small board (~25 g),
# a 2S 450-500 mAh LiPo (~30 g), IMU, and 14 servo leads (~18 g).
M_ELEC_NANODUCK = 0.075

# --- Torque-speed envelopes ---------------------------------------------------
#     tau_max(w) = min(tau_cap, tau_stall * (1 - w/w_noload))
#
# XL330 comes from the BAM m6 fit actually used in training (kt = 0.3660 Nm/A,
# R = 2.8114 ohm) at the pack voltage the robot really runs: microduck_constants
# sets vin_range 6.5-8.2 V, i.e. a 2S LiPo, NOT the 5 V on the datasheet.
# At 7.4 V that gives tau_stall = kt*V/R = 0.963 Nm -- exactly the MJCF
# `chosen_actuator` forcerange of +/-0.96, which confirms this is the envelope
# the shipped policy was trained against.  The 1.75 A firmware limit caps
# delivered torque at kt*1.75 = 0.641 Nm, so the curve is flat, then falls.
#
# MG90S has no current limit and no feedback; its line is fixed by the two
# catalogue points (stall torque, and 60 deg travel time at no load).  Clones
# vary a lot -- bench-measure before committing (the repo already has a
# testbench: scripts/validate_bam_testbench.py, robot/xl330_test_bench/).
KT, R_OHM, I_MAX = 0.36601, 2.81139, 1.75


def dxl(vin):
    return dict(tau=KT * vin / R_OHM, w=vin / KT, cap=KT * I_MAX, m=0.018, vol=17.7)


SERVOS = {
    "XL330 @7.4V (baseline)": dxl(7.4),
    "XL330 @6.5V (pack low)": dxl(6.5),
    "MG90S @4.8V":            dict(tau=0.177, w=10.47, cap=None, m=0.0134, vol=7.9),
    "MG90S @6.0V":            dict(tau=0.216, w=13.09, cap=None, m=0.0134, vol=7.9),
}

# Fraction of samples allowed to exceed the servo capability.  Brief clipping is
# survivable (the policy trains with domain randomisation); sustained is not.
QUANTILE = 99.0


def total_mass(s, sv):
    """Robot mass at scale s: only the structure shrinks."""
    elec = M_ELEC_MICRODUCK if sv["m"] == XL330_MASS else M_ELEC_NANODUCK
    return N_SERVO * sv["m"] + elec + s ** 3 * M_STRUCT


def envelope(sv, w):
    """Available torque at speed w (w >= 0 driving, w < 0 braking)."""
    avail = sv["tau"] * np.clip(1.0 - w / sv["w"], 0.0, None)
    if sv.get("cap") is not None:
        avail = np.minimum(avail, sv["cap"])
    return avail


def utilisation(s, sv, gait):
    """Per-sample tau_demand / tau_available at that instant speed.

    Speed is signed relative to the torque direction: braking (speed opposing
    torque) is credited only up to the flat part of the envelope, never more.
    """
    mass = total_mass(s, sv)
    tau = TAU * (mass / M0) * s
    w = (W / np.sqrt(s)) if gait == "froude" else W
    w_aligned = np.abs(w) * np.where(np.sign(w) * SGN >= 0, 1.0, -1.0)
    return tau / np.maximum(envelope(sv, w_aligned), 1e-9), mass


def worst(s, sv, gait):
    u, mass = utilisation(s, sv, gait)
    return np.percentile(u, QUANTILE, axis=0), mass


def max_scale(sv, gait, target):
    grid = np.linspace(0.20, 1.30, 111)
    ok = [s for s in grid if worst(s, sv, gait)[0].max() <= target]
    return max(ok) if ok else None


def peak_power(sv):
    w = np.linspace(0.0, sv["w"], 20001)
    return float((w * envelope(sv, w)).max())


print("MicroDuck: %.0f mm, %.3f kg, %d samples of paired (tau, omega)\n"
      % (HEIGHT0_MM, M0, TAU.shape[0]))
print("Per-joint demand (p%.0f of |tau|, p%.0f of |omega|):" % (QUANTILE, QUANTILE))
for i, n in enumerate(NAMES):
    print("   %-16s %.3f Nm   %.2f rad/s" % (n, np.percentile(TAU[:, i], QUANTILE),
                                             np.percentile(np.abs(W[:, i]), QUANTILE)))
print("\nServo envelopes:")
for name, sv in SERVOS.items():
    print("   %-24s stall %.3f Nm, no-load %5.2f rad/s, cap %-11s peak %.2f W"
          % (name, sv["tau"], sv["w"],
             "-" if sv.get("cap") is None else "%.3f Nm" % sv["cap"], peak_power(sv)))
print()

for gait in ("froude", "period"):
    print("#" * 78)
    print("# gait: %s" % (
        "Froude-similar, omega x 1/sqrt(s)  [pessimistic]" if gait == "froude"
        else "same gait period, omega unchanged  [optimistic]"))
    print("#" * 78)
    for name, sv in SERVOS.items():
        row = "%-24s" % name
        for target, label in ((1.00, "feasible"), (0.80, "20% margin")):
            s = max_scale(sv, gait, target)
            if s is None:
                row += "  %s: none" % label
            else:
                _, mass = worst(s, sv, gait)
                row += "  %s: s<=%.2f (%3.0fmm %3.0fg)" % (label, s, s * HEIGHT0_MM, mass * 1000)
        print(row)
        s = max_scale(sv, gait, 1.00) or 0.20
        u, _ = worst(s, sv, gait)
        hot = np.argsort(-u)[:3]
        print("      at s=%.2f, tightest: %s" % (s, ",  ".join(
            "%s %s" % (NAMES[i], ">no-load speed" if u[i] > 100 else "%.0f%%" % (100 * u[i]))
            for i in hot)))
    print()

print("=" * 78)
print("WHY GEARING DOES NOT RESCUE THE KNEE: it is power-limited, not torque-limited")
print("=" * 78)
knee = NAMES.index("left_knee")
p_knee = np.percentile(np.abs(TAU[:, knee] * W[:, knee]), QUANTILE)
print("  MicroDuck left_knee p%.0f mechanical power = %.2f W" % (QUANTILE, p_knee))
for name, sv in SERVOS.items():
    print("     %-24s can make %.2f W" % (name, peak_power(sv)))
print()
print("  Under a Froude-similar gait, power scales as (M(s)/M0)*sqrt(s):")
mg = SERVOS["MG90S @6.0V"]
for sc in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
    mass = total_mass(sc, mg)
    print("     s=%.1f (%3.0f mm, %3.0f g = %3.0f fixed + %3.0f structure)  knee needs %.2f W"
          % (sc, sc * HEIGHT0_MM, mass * 1000,
             1000 * (N_SERVO * mg["m"] + M_ELEC_NANODUCK), 1000 * sc ** 3 * M_STRUCT,
             p_knee * (mass / M0) * np.sqrt(sc)))
print()
print("  Note the mass FLOOR: servos (%.0f g) + electronics (%.0f g) = %.0f g never"
      % (1000 * N_SERVO * mg["m"], 1000 * M_ELEC_NANODUCK,
         1000 * (N_SERVO * mg["m"] + M_ELEC_NANODUCK)))
print("  shrinks, so below s~0.5 the robot stops getting meaningfully lighter and")
print("  only the moment arm keeps improving.  That is where the returns flatten.")
print()
print("  A reduction ratio trades torque against speed and leaves the product")
print("  alone, so it cannot buy power.  A 2:1 knee gear would need 13.3 rad/s")
print("  at the servo -- past MG90S no-load speed.  The lever that DOES work is")
print("  retraining the gait against an MG90S actuator model, so the policy")
print("  finds a slower, more crouched walk inside the power envelope.")
