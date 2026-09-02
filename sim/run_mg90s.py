#!/usr/bin/env python3
"""Run MicroDuck's shipped walking policy on MG90S servos instead of XL330s.

This drives microduck_rl's own `scripts/infer_policy.py` -- same policy, same
observation assembly, same 50 Hz decimation -- and swaps only the actuator, so
any difference in the result is the servo and nothing else.

    # headless benchmark, prints a verdict
    python run_mg90s.py --servo xl330            # control: should walk
    python run_mg90s.py --servo mg90s-6v         # the question

    # watch it in the viewer
    python run_mg90s.py --servo mg90s-6v --view

The actuator swap is installed lazily, on the first `mj_step`, so it lands
*after* infer_policy has finished configuring the model (timestep, current
limit, foot friction) and cannot be clobbered by it.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import os
import runpy
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg90s import SERVOS, ServoBank  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "microduck_rl"))
POLICIES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Microduck", "policies"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servo", default="mg90s-6v", choices=sorted(SERVOS),
                    help="which servo to put in every joint")
    ap.add_argument("--policy", default=os.path.join(POLICIES, "alpha_walking.onnx"))
    ap.add_argument("--lin-vel-x", type=float, default=0.3, help="forward velocity command [m/s]")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds to run headless")
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer instead")
    ap.add_argument("--verbose", action="store_true", help="show infer_policy's own output")
    # Ablations: an MG90S differs from an XL330 in two independent ways -- how
    # much power it can make, and how its analogue loop behaves.  Override the
    # control-law fields to hold one constant and vary the other, so a fall can
    # be attributed to the right cause.
    ap.add_argument("--kp", type=float, default=None,
                    help="override joint stiffness [Nm/rad] (XL330 is 0.55)")
    ap.add_argument("--deadband-deg", type=float, default=None, help="override deadband")
    ap.add_argument("--latency-ms", type=float, default=None, help="override command latency")
    args = ap.parse_args()

    spec = SERVOS[args.servo]
    overrides = {}
    if args.kp is not None:
        overrides["kp"] = args.kp
    if args.deadband_deg is not None:
        overrides["deadband_rad"] = np.radians(args.deadband_deg)
    if args.latency_ms is not None:
        overrides["latency_s"] = args.latency_ms / 1000.0
    if overrides:
        tag = ", ".join(
            "kp %.2f" % v if k == "kp"
            else "deadband %.1f deg" % np.degrees(v) if k == "deadband_rad"
            else "latency %.0f ms" % (1000 * v)
            for k, v in overrides.items())
        spec = dataclasses.replace(spec, name="%s [%s]" % (spec.name, tag), **overrides)
    os.chdir(REPO)   # infer_policy.py resolves its scene XML relative to the repo root

    import mujoco
    import mujoco.viewer

    state = {"bank": None, "traj": [], "model": None, "data": None}
    real_step = mujoco.mj_step

    def stepped(m, d, nstep=1):
        # Lazy install: by the first step, infer_policy has finished editing the model.
        if state["bank"] is None:
            bank = ServoBank(m, spec)
            bank.install(m)
            state["bank"], state["model"], state["data"] = bank, m, d
        target = d.ctrl.copy()
        d.ctrl[:] = state["bank"].torque(d, target)
        out = real_step(m, d, nstep)
        d.ctrl[:] = target
        state["traj"].append((d.time, d.qpos[0], d.qpos[1], d.qpos[2]))
        return out

    mujoco.mj_step = stepped

    if not args.view:
        class FakeViewer:
            def __init__(self):
                self.t0 = time.time()

            def sync(self):
                pass

            def is_running(self):
                return time.time() - self.t0 < args.duration

            def __getattr__(self, _):
                return lambda *a, **k: None

        @contextlib.contextmanager
        def fake_launch(model, data, **kw):
            yield FakeViewer()

        mujoco.viewer.launch_passive = fake_launch

    sys.argv = [
        "infer_policy.py", "--new-cmd-obs",
        "--walking", args.policy,
        "--lin-vel-x", str(args.lin_vel_x),
        # Let the ServoBank own the torque limit; infer_policy's XL330 current
        # clip would otherwise re-impose a flat +/-0.641 Nm forcerange.
        "--current-limit", "0",
    ]
    # infer_policy prints its whole key-binding help every run, which buries the
    # result when comparing servos back to back.
    sink = contextlib.redirect_stdout(io.StringIO()) if not (args.verbose or args.view) \
        else contextlib.nullcontext()
    try:
        with sink:
            runpy.run_path(os.path.join(REPO, "scripts", "infer_policy.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        mujoco.mj_step = real_step
    print(">>> %s: stall %.3f Nm, no-load %.2f rad/s, peak %.2f W, "
          "kp %.2f Nm/rad (saturates at %.1f deg), deadband %.1f deg, latency %.0f ms"
          % (spec.name, spec.tau_stall, spec.w_noload, spec.peak_power, spec.kp,
             spec.saturation_error_deg, np.degrees(spec.deadband_rad), 1000 * spec.latency_s))

    verdict(state, spec, args)


def verdict(state, spec, args) -> None:
    traj = np.array(state["traj"])
    if traj.size == 0 or state["data"] is None:
        print("no simulation ran")
        return
    t, x, y, z = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]

    # Standing trunk height is ~120 mm; treat a sustained drop below 70 mm as a fall.
    FALLEN_M = 0.070
    fallen = z < FALLEN_M
    first_fall = t[np.argmax(fallen)] if fallen.any() else None

    # Measure speed over the interval the robot was still upright.  Use path
    # length, not world-X displacement: the policy yaw-drifts (no heading
    # command), so a robot that walks a wide arc would score near zero on X.
    end = np.searchsorted(t, first_fall) if first_fall is not None else len(t)
    end = max(end, 2)
    dt = t[end - 1] - t[0]
    path = float(np.sum(np.hypot(np.diff(x[:end]), np.diff(y[:end]))))
    speed = path / dt if dt > 0 else 0.0

    print("=" * 70)
    print("SERVO   %s" % spec.name)
    print("POLICY  %s   command %.2f m/s" % (os.path.basename(args.policy), args.lin_vel_x))
    print("-" * 70)
    print("  sim time                %.1f s" % t[-1])
    print("  walking speed           %.3f m/s along path   (commanded %.2f)"
          % (speed, args.lin_vel_x))
    print("  distance walked         %.2f m in %.1f s" % (path, dt))
    print("  trunk height  mean/min  %.0f / %.0f mm" % (1000 * z[:end].mean(), 1000 * z.min()))
    print("  servo at full duty      %.0f%% of samples" % (100 * state["bank"].clipped_fraction))
    if first_fall is not None:
        print("  FELL at t = %.1f s" % first_fall)
        print("\n  VERDICT: does not walk on %s." % spec.name)
    elif speed < 0.3 * args.lin_vel_x:
        print("\n  VERDICT: stays up on %s but barely moves (%.0f%% of commanded speed)."
              % (spec.name, 100 * speed / max(args.lin_vel_x, 1e-6)))
    else:
        print("\n  VERDICT: walks on %s." % spec.name)
    print("=" * 70)


if __name__ == "__main__":
    main()
