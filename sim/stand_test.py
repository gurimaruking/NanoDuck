#!/usr/bin/env python3
"""Can NanoDuck stand on MG90S servos, holding only its home pose?

    python stand_test.py                 # MG90S/MG92B as specified
    python stand_test.py --servo xl330   # control: should be rock solid
    python stand_test.py --view

No policy is involved -- the servos are simply commanded to the STAND keyframe
and gravity is switched on.  That makes this a clean test of the static half of
the problem: is there enough torque to hold the robot up, and does the analogue
loop hold still or hunt?

It is deliberately the easy half.  A robot that cannot stand certainly cannot
walk, but standing says nothing about the swing-phase power that
analysis/design_point.py is actually worried about.  Walking needs the policy,
which needs retraining for 10 DOF.
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mg90s import SERVOS, ServoBank  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene_nanoduck.xml")

# Which servo sits in which joint (bom/bom.csv, analysis/design_point.py).
ASSIGN = {"left_knee": "mg92b-6v", "right_knee": "mg92b-6v",
          "neck_pitch": "micro-6v", "head_yaw": "micro-6v"}
DEFAULT = "mg90s-6v"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--servo", default=None,
                    help="force one servo everywhere (default: the per-joint assignment)")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--push", type=float, default=0.0,
                    help="lateral shove [m/s] applied at t=2s")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    picks = [args.servo or ASSIGN.get(n, DEFAULT) for n in names]

    # One ServoBank per distinct servo type, each owning its own joints.
    banks = []
    for kind in sorted(set(picks)):
        idx = [i for i, p in enumerate(picks) if p == kind]
        b = ServoBank(model, SERVOS[kind], joints=idx)
        b.install(model)
        banks.append((kind, b))

    print("NanoDuck stand test")
    print("  mass         %.1f g" % (1000 * model.body_subtreemass[1]))
    print("  trunk height %.1f mm at STAND" % (1000 * data.qpos[2]))
    for kind, b in banks:
        print("  %-10s -> %s" % (kind, ", ".join(names[i] for i in b.joints)))
    print()

    target = data.ctrl.copy()          # the keyframe's ctrl == the home pose
    hold = target.copy()
    n_steps = int(args.duration / model.opt.timestep)
    trace, sat = [], []

    viewer = None
    if args.view:
        # NB: `import mujoco.viewer` here would rebind `mujoco` as a local.
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data, show_left_ui=False,
                                          show_right_ui=False)

    for k in range(n_steps):
        t = k * model.opt.timestep
        if args.push and abs(t - 2.0) < model.opt.timestep:
            data.qvel[1] += args.push
        for _, b in banks:
            data.ctrl[b.joints] = b.torque(data, hold)
        mujoco.mj_step(model, data)
        data.ctrl[:] = hold
        trace.append((t, data.qpos[2], data.qpos[0], data.qpos[1]))
        sat.append(max(b.clipped_fraction for _, b in banks))
        if viewer is not None:
            viewer.sync()
    if viewer is not None:
        viewer.close()

    tr = np.array(trace)
    z = tr[:, 1]
    z0 = z[0]
    settle = z[int(0.6 * len(z)):]
    print("  trunk height  start %.1f mm -> settled %.1f +/- %.2f mm"
          % (1000 * z0, 1000 * settle.mean(), 1000 * settle.std()))
    print("  drift         %.1f mm forward, %.1f mm lateral"
          % (1000 * (tr[-1, 2] - tr[0, 2]), 1000 * (tr[-1, 3] - tr[0, 3])))
    print("  servos at full duty: %.0f%% of samples" % (100 * max(sat)))
    print()
    if settle.mean() < 0.5 * z0:
        # A fixed pose has no balance controller, so anything that walks the CoM
        # outside the support polygon topples the robot no matter what servo is
        # fitted.  Saturation is what separates the two failure modes, and it is
        # worth saying which one happened: a torque problem is fixed by a better
        # servo, a balance problem only by a policy.
        if max(sat) < 0.02:
            print("  VERDICT: collapses -- but no servo ever reached full duty, so this")
            print("           is a BALANCE failure, not a torque failure.  Holding a fixed")
            print("           pose cannot recover from a push; that needs the policy.")
        else:
            print("  VERDICT: collapses, and servos saturated %.0f%% of the time -- out of torque."
                  % (100 * max(sat)))
    elif settle.std() > 0.002:
        print("  VERDICT: stands but hunts (%.1f mm rms) -- deadband limit cycle."
              % (1000 * settle.std()))
    else:
        print("  VERDICT: stands, %.1f mm sag, %.2f mm rms."
              % (1000 * (z0 - settle.mean()), 1000 * settle.std()))


if __name__ == "__main__":
    main()
