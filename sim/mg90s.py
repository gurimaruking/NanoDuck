"""An MG90S hobby-servo actuator model for MuJoCo.

Why this exists
---------------
MicroDuck's servos are modelled two different ways upstream:

* training (`FrictionDRBamActuator`) uses BAM's XL330 *voltage* control law --
  back-EMF, Coulomb/Stribeck/load-dependent friction, the lot;
* the CPU viewer (`infer_policy.py`) uses a plain MuJoCo ``position`` actuator
  with a flat ``forcerange``.

Neither describes an MG90S, and the flat ``forcerange`` in particular hides the
one thing that decides whether NanoDuck can walk: a brushed servo makes **less
torque the faster it spins**.  MicroDuck's knee runs at 6.6 rad/s while pulling
0.49 Nm; on an MG90S, 6.6 rad/s is half of no-load speed, where barely half the
stall torque is left.  A box-shaped force limit cannot express that, so a sim
built on it will happily "walk" a robot that would collapse on the bench.

The model
---------
An MG90S is a brushed DC motor behind a ~1:300 gearbox, closed by a small
analogue chip that compares a PWM pulse width against a potentiometer on the
output shaft.  So:

    err  = deadband(q_target - q_measured)      # pot reads the OUTPUT shaft
    u    = clip(kp * err / tau_stall, -1, +1)   # analogue P loop -> PWM duty
    tau  = tau_stall * u  -  (tau_stall/w_noload) * qd

The loop is parameterised by the *joint stiffness* ``kp`` in Nm/rad rather than
by an internal gain, because kp is the number that can be read straight off
`joints_properties.xml` (``chosen_actuator`` uses kp = 0.55 Nm/rad) and
compared between servos.  Getting this wrong is not subtle: a first cut of this
file used "full drive beyond 1 deg of error", i.e. 55 Nm/rad, a hundred times
stiffer than the model the policy was trained on.  The result was bang-bang
control that fell over in 2.4 s -- with the *XL330* numbers.  If the control
case does not walk, the model is wrong, not the servo.

The second term is back-EMF: it both caps speed at ``w_noload`` when u=1 and
provides the damping you feel when backdriving an unpowered servo.  Together
the two terms reproduce the whole torque-speed triangle rather than a box.

Three behaviours that bite in sim2real and are modelled here:

* **deadband** -- the analogue comparator ignores small errors, so the servo
  parks a degree or two off target and never trims it out.  This is the main
  reason a hobby-servo biped jitters where a Dynamixel one does not.
* **command latency** -- standard PWM framing is 50 Hz, so a new target waits
  up to 20 ms before the servo even sees it, plus internal loop lag.
* **gearbox friction** -- set via ``dof_frictionloss`` on the joints, not here,
  because MuJoCo solves stiction implicitly (an explicit friction torque in a
  Python loop oscillates at the timestep).

Parameter provenance
--------------------
``tau_stall`` and ``w_noload`` are the two catalogue points of a Tower Pro
MG90S.  ``K``, ``deadband_rad`` and the friction numbers are engineering
estimates, NOT measurements -- MG90S clones vary enormously.  Measure them on
the bench before trusting any sim2real result; microduck_rl already carries the
rig for exactly this (``scripts/validate_bam_testbench.py``,
``src/mjlab_microduck/robot/xl330_test_bench/``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ServoSpec:
    """Torque-speed envelope and control behaviour of one servo type."""

    name: str
    tau_stall: float      # Nm at zero speed, full duty
    w_noload: float       # rad/s at zero torque, full duty
    kp: float             # Nm/rad, closed-loop joint stiffness before saturation
    deadband_rad: float   # errors below this produce no drive at all
    frictionloss: float   # Nm, applied to the joint via dof_frictionloss
    damping: float        # Nm/(rad/s), residual gearbox viscous term
    armature: float       # kg m^2, rotor inertia reflected through the gearbox
    latency_s: float      # command transport delay (PWM framing + loop lag)

    @property
    def peak_power(self) -> float:
        """Best mechanical watts the servo can make, at half no-load speed."""
        return self.tau_stall * self.w_noload / 4.0

    @property
    def back_emf(self) -> float:
        """Nm per rad/s. Slope of the torque-speed line; acts as damping."""
        return self.tau_stall / self.w_noload

    @property
    def saturation_error_deg(self) -> float:
        """Position error at which the servo goes to full drive."""
        return math.degrees(self.tau_stall / self.kp)


# Tower Pro MG90S. Catalogue: 1.8 kgf-cm / 0.10 s per 60 deg at 4.8 V,
# 2.2 kgf-cm / 0.08 s per 60 deg at 6.0 V, 13.4 g, metal gears.
# The servo MicroDuck actually ships with -- the control case.  Every number is
# traceable, which is what makes it usable as a calibration:
#
#   tau_stall  kt*V/R with the BAM m6 fit (kt 0.36601 Nm/A, R 2.81139 ohm) at the
#              7.4 V 2S pack microduck_constants.vin_range describes.  0.963 Nm,
#              which is exactly the MJCF `chosen_actuator` forcerange of +/-0.96.
#   kp         joints_properties.xml `chosen_actuator`, position kp = 0.55.
#   damping    the MJCF says 0.053, but that was fitted for a position actuator
#              with NO back-EMF term, so it absorbs one.  Here back-EMF is
#              explicit (0.963/20.2 = 0.0477), leaving 0.053 - 0.0477 = 0.0054
#              of real viscous friction -- and BAM's independently fitted
#              friction_viscous is 0.00536.  The two decompositions agree to 1%,
#              which is a strong sign this split is the physical one.
#   frictionloss / armature  MJCF 0.0048 / 0.0018, BAM 0.00477 / 0.00181.
_KT, _R = 0.36601, 2.81139
XL330_7V4 = ServoSpec(
    name="XL330 @7.4V",
    tau_stall=_KT * 7.4 / _R,             # 0.963 Nm
    w_noload=7.4 / _KT,                   # 20.2 rad/s
    kp=0.55,
    deadband_rad=0.0,                     # magnetic encoder, no deadband
    frictionloss=0.0048,
    damping=0.0054,
    armature=0.0018,
    latency_s=0.008,
)

# Tower Pro MG90S. Catalogue: 1.8 kgf-cm / 0.10 s per 60 deg at 4.8 V,
# 2.2 kgf-cm / 0.08 s per 60 deg at 6.0 V, 13.4 g, metal gears.
#
# `kp` is the soft number here.  A hobby servo's analogue comparator drives to
# full duty within a few degrees, so it is STIFFER, relative to its own much
# smaller torque, than a Dynamixel running kp_fw=200.  5 deg to saturation gives
# 0.216/0.0873 = 2.5 Nm/rad, roughly 4.5x the XL330 joint stiffness the policy
# was trained against -- a sim2real gap in its own right, independent of torque.
MG90S_6V = ServoSpec(
    name="MG90S @6.0V",
    tau_stall=0.216,
    w_noload=math.radians(60.0) / 0.08,   # 13.09 rad/s
    kp=0.216 / math.radians(5.0),         # full drive beyond ~5 deg of error
    deadband_rad=math.radians(1.0),
    frictionloss=0.012,                   # metal gears, ~2.5x the XL330 number
    damping=0.005,
    armature=0.003,
    latency_s=0.020,                      # one 50 Hz PWM frame
)

MG90S_4V8 = ServoSpec(
    name="MG90S @4.8V",
    tau_stall=0.177,
    w_noload=math.radians(60.0) / 0.10,   # 10.47 rad/s
    kp=0.177 / math.radians(5.0),
    deadband_rad=math.radians(1.0),
    frictionloss=0.012,
    damping=0.005,
    armature=0.003,
    latency_s=0.020,
)

# MG92B: NanoDuck's knees, and only its knees.  Same 22.8 x 12.2 footprint as an
# MG90S for +0.4 g, but ~45% more stall torque -- which moves the whole robot's
# mass budget from 205 g to 245 g (analysis/design_point.py).
# The stall figure is quoted inconsistently across vendors; this is the
# conservative end and it MUST be bench-measured before the design leans on it.
MG92B_6V = ServoSpec(
    name="MG92B @6.0V",
    tau_stall=0.310,
    w_noload=math.radians(60.0) / 0.08,
    kp=0.310 / math.radians(5.0),
    deadband_rad=math.radians(1.0),
    frictionloss=0.014,
    damping=0.005,
    armature=0.003,
    latency_s=0.020,
)

# Generic 3.7 g micro servo for the neck and head yaw.  Cheap ones have a wider
# deadband than an MG90S, which is fine on a head and fatal on a leg.
MICRO_6V = ServoSpec(
    name="micro 3.7g @6.0V",
    tau_stall=0.060,
    w_noload=math.radians(60.0) / 0.10,
    kp=0.060 / math.radians(5.0),
    deadband_rad=math.radians(1.5),
    frictionloss=0.004,
    damping=0.002,
    armature=0.001,
    latency_s=0.020,
)

SERVOS = {"mg90s-6v": MG90S_6V, "mg90s-4v8": MG90S_4V8, "mg92b-6v": MG92B_6V,
          "micro-6v": MICRO_6V, "xl330": XL330_7V4}


class ServoBank:
    """Drives every actuated joint of a model with one ServoSpec.

    Usage is deliberately non-invasive: :meth:`install` rewrites the model's
    ``position`` actuators into direct-torque actuators and returns a callable
    that turns the position targets a policy writes to ``data.ctrl`` into the
    torque a real servo would produce.  Nothing else in the sim has to change.
    """

    def __init__(self, model: mujoco.MjModel, spec: ServoSpec,
                 current_limit: float | None = None, joints=None):
        """`joints` selects a subset of actuator indices; None means all of them.

        A robot can mix servo types (NanoDuck puts MG92B in the knees and micro
        servos in the head), so a bank owns a subset and several banks coexist.
        """
        self.spec = spec
        self.joints = np.arange(model.nu) if joints is None else np.asarray(joints, dtype=int)
        self.n = len(self.joints)
        self.current_limit = current_limit
        # qpos/qvel addresses of the joint behind each of our actuators.
        self.qpos_adr = np.empty(self.n, dtype=int)
        self.qvel_adr = np.empty(self.n, dtype=int)
        for k, i in enumerate(self.joints):
            jid = model.actuator_trnid[i, 0]
            self.qpos_adr[k] = model.jnt_qposadr[jid]
            self.qvel_adr[k] = model.jnt_dofadr[jid]
        self._delay_buf: list[np.ndarray] = []
        self._delay_steps = 0
        self.clipped_fraction = 0.0
        self._n_calls = 0
        self._n_clipped = 0

    def install(self, model: mujoco.MjModel) -> None:
        """Convert position actuators to torque actuators and apply joint properties.

        A MuJoCo ``position`` actuator computes ``kp*(ctrl - q) - kv*qd``
        internally and clamps it to a *constant* forcerange.  We want the
        torque limit to depend on speed, so we take the control law into
        Python: gaintype FIXED with gain 1 and biastype NONE makes ``ctrl``
        the joint torque directly.
        """
        for k, i in enumerate(self.joints):
            model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            model.actuator_gainprm[i, :] = 0.0
            model.actuator_gainprm[i, 0] = 1.0
            model.actuator_biasprm[i, :] = 0.0
            model.actuator_ctrlrange[i] = (-self.spec.tau_stall, self.spec.tau_stall)
            model.actuator_ctrllimited[i] = 1
            model.actuator_forcerange[i] = (-self.spec.tau_stall, self.spec.tau_stall)
            model.actuator_forcelimited[i] = 1
            dof = self.qvel_adr[k]
            model.dof_frictionloss[dof] = self.spec.frictionloss
            model.dof_damping[dof] = self.spec.damping
            model.dof_armature[dof] = self.spec.armature
        self._delay_steps = max(0, int(round(self.spec.latency_s / model.opt.timestep)))
        self._delay_buf = []

    def torque(self, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
        """Torque each servo produces this step, given the commanded positions.

        `target` is the FULL ``data.ctrl``-shaped vector of joint angles [rad];
        the returned torques correspond to ``self.joints``, in that order.
        """
        s = self.spec
        # Command transport delay: the servo acts on a target from latency_s ago.
        self._delay_buf.append(np.asarray(target, dtype=float)[self.joints].copy())
        if len(self._delay_buf) > self._delay_steps + 1:
            self._delay_buf.pop(0)
        cmd = self._delay_buf[0]

        q = data.qpos[self.qpos_adr]
        qd = data.qvel[self.qvel_adr]

        err = cmd - q
        # Analogue comparator deadband: shrink toward zero, do not just clip,
        # so torque is continuous as the error crosses the band edge.
        err = np.sign(err) * np.maximum(np.abs(err) - s.deadband_rad, 0.0)
        duty = np.clip(err * s.kp / s.tau_stall, -1.0, 1.0)

        tau = s.tau_stall * duty - s.back_emf * qd
        if self.current_limit is not None:
            tau = np.clip(tau, -self.current_limit, self.current_limit)
        tau = np.clip(tau, -s.tau_stall, s.tau_stall)

        # Bookkeeping: how often is the servo asking for more than it can give?
        self._n_calls += self.n
        self._n_clipped += int(np.sum(np.abs(duty) >= 0.999))
        self.clipped_fraction = self._n_clipped / max(self._n_calls, 1)
        return tau

    def patch_mj_step(self, model: mujoco.MjModel):
        """Wrap ``mujoco.mj_step`` so any existing sim loop drives this bank.

        Callers write joint *position* targets to ``data.ctrl`` exactly as
        before; this reads them, substitutes the servo torque, and steps.
        Returns the original function so it can be restored.
        """
        real_step = mujoco.mj_step

        def stepped(m, d, nstep=1):
            target = d.ctrl.copy()
            d.ctrl[:] = self.torque(d, target)
            out = real_step(m, d, nstep)
            d.ctrl[:] = target      # leave ctrl as the caller left it
            return out

        mujoco.mj_step = stepped
        return real_step
