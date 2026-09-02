#!/usr/bin/env python3
"""Scale MicroDuck's CAD meshes down to NanoDuck size.

    python scale_meshes.py --scale 0.50

Reads every binary STL in `microduck_src/` and writes a uniformly scaled copy
to `nanoduck_s<NNN>/`.  Under uniform scaling the facet normals are unchanged
(they are unit vectors), so only the nine vertex floats of each triangle are
touched.

What this is and is not
-----------------------
This gets you a printable, correctly proportioned NanoDuck shell in one command,
which is enough to check reach, clearance and mass, and to start the sim.  It is
NOT a finished mechanical design, because three things do not scale:

* **Servo pockets.**  Every mount here is cut for an XL330 (29 x 20 x 34 mm).
  An MG90S is 22.8 x 12.2 x 28.5 mm with mounting ears on the sides and a single
  output shaft -- a completely different interface.  Every servo pocket has to
  be redrawn.
* **Wall thickness.**  Scaling 0.5x halves a 2 mm wall to 1 mm, below what most
  FDM printers do well and far below what the joint loads want.  Walls must be
  put back to absolute thickness, not scaled.
* **Bought parts.**  Bearings, screws and the battery keep their real sizes, so
  their pockets need redrawing too (M2 stays M2; MicroDuck's 22x16x4 bearing
  becomes an MR63 or MR83, not an 11x8x2 that does not exist).

The real route is the Onshape source (see cad/README.md), where these are
parametric.  This script is the fast path to something you can look at and
simulate today.
"""

from __future__ import annotations

import argparse
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def scale_stl(src: str, dst: str, s: float) -> tuple[int, np.ndarray]:
    """Uniformly scale one binary STL. Returns (triangle count, bbox in mm)."""
    with open(src, "rb") as f:
        data = bytearray(f.read())
    if data[:5] == b"solid" and b"facet" in data[:512]:
        raise ValueError("%s is ASCII STL; this script handles binary only" % src)

    n = struct.unpack("<I", bytes(data[80:84]))[0]
    body = np.frombuffer(memoryview(data)[84:84 + 50 * n], dtype=np.uint8).reshape(n, 50).copy()

    # bytes 0:12 normal, 12:48 three vertices, 48:50 attribute count.
    verts = body[:, 12:48].copy().view(np.float32).reshape(n, 3, 3)
    verts *= s
    body[:, 12:48] = verts.reshape(n, 9).view(np.uint8)

    data[84:84 + 50 * n] = body.tobytes()
    with open(dst, "wb") as f:
        f.write(bytes(data))

    flat = verts.reshape(-1, 3)
    return n, (flat.max(0) - flat.min(0)) * 1000.0   # STL units are metres


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=0.50,
                    help="linear scale factor (0.50 = the NanoDuck design point)")
    ap.add_argument("--src", default=os.path.join(HERE, "microduck_src"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, "nanoduck_s%03d" % round(args.scale * 100))
    os.makedirs(out, exist_ok=True)

    names = sorted(f for f in os.listdir(args.src) if f.endswith(".stl"))
    print("scaling %d meshes by %.3f  ->  %s\n" % (len(names), args.scale, out))
    interesting = {"trunk_base.stl", "leg.stl", "upper_leg_left.stl", "ankle_left.stl",
                   "sole_left.stl", "top_head_shell.stl", "xl330.stl"}
    for name in names:
        n, bbox = scale_stl(os.path.join(args.src, name), os.path.join(out, name), args.scale)
        if name in interesting:
            print("   %-24s %6.1f x %6.1f x %6.1f mm" % (name, *bbox))
    print("\ndone. %d files written." % len(names))
    print("Reminder: servo pockets are still XL330-shaped and walls are now %.0f%% "
          "of their original thickness." % (100 * args.scale))


if __name__ == "__main__":
    main()
