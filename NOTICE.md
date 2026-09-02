# Licensing and attribution

This repository is licensed in two parts, mirroring
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl),
because part of it is derived from that project.

## Code — Apache License 2.0

Everything in `analysis/`, `sim/*.py`, `cad/*.py`, and the documentation is
original work, licensed under [Apache 2.0](LICENSE).

## 3D model files — Creative Commons BY-NC-SA 4.0

**`cad/microduck_src/` is not our work.** It is a verbatim copy of
`src/mjlab_microduck/robot/microduck/assets/` (47 STL meshes) plus
`robot_walk.xml`, `joints_properties.xml` and `config_mjcf_walk.json` from
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl),
whose README states:

> 3D model files are licensed under Creative Commons BY-SA-NC.

i.e. [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Those terms carry over to this copy and to anything derived from it:

* **BY** — attribution to Pollen Robotics, as given here.
* **NC** — non-commercial use only.
* **SA** — derivatives must be shared under the same licence.

### Why these files are kept here

`cad/check_servo_fit.py` measures the XL330's real envelope from `xl330.stl`
and MicroDuck's link lengths from `robot_walk.xml`, and `analysis/` transfers
joint demand recorded from MicroDuck's own gait.  Every conclusion in this
repository traces back to those measurements, so removing the source files
would leave the analysis unreproducible.

### What is derived from them

`analysis/microduck_demand.npz` is measured data — per-joint torque and
angular velocity recorded from MicroDuck's shipping walking policy running in
MuJoCo on the model above.  Treat it as covered by the same CC BY-NC-SA terms.

`sim/nanoduck.xml` is **not** derived from those meshes.  It is generated from
scratch by `cad/build_nanoduck.py` out of primitive boxes and the parameters in
that file; it shares no geometry with MicroDuck.  Its kinematic layout was of
course informed by MicroDuck, and the sensor and body names deliberately match
so that microduck_rl's tooling runs against it.

## Not affiliated with Pollen Robotics

NanoDuck is an independent redesign. If you want the real thing — which walks,
sees, and is supported — buy a MicroDuck: https://pollen-robotics.com/microduck
