#!/usr/bin/env python3
"""Run the Python controller on sim/orbi/model.xml and write what the browser engine must reproduce.

USAGE: python export_run.py [--out runs/compare]
Writes one JSON per scenario: the start state, the target, the control sent at every 1 ms step and the
state every 10 steps. sim/compare_wasm.mjs replays it in the vendored WebAssembly build.
"""
import argparse, hashlib, json
from pathlib import Path
import mujoco, numpy as np
import build_model as bm
from controller import Gait, Terrain

HERE = Path(__file__).resolve().parent
FT = bm.FT
SCEN = {  # name: (start page x,z ; target page x,z ; origin page x,z or None ; stop_at ; seconds)
    "flat_stop": ((0, 0), (5, 5), (0, 0), True, 10.0),         # open floor, click-style stop
    "stairs": ((1.5, 0), (12, 0), (1.5, 0), False, 12.0),       # the 4-step staircase, autopilot style
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=HERE / "runs" / "compare")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    xml = (HERE / "orbi" / "model.xml").read_bytes()
    for name, (start, tgt, org, stop_at, seconds) in SCEN.items():
        m = mujoco.MjModel.from_xml_string(xml.decode()); d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("start").id)
        d.qpos[0], d.qpos[1] = start[0] * FT, -start[1] * FT
        mujoco.mj_forward(m, d)
        g = Gait(m, d); g.terrain = Terrain(m)
        g.target, g.stop_at = (tgt[0] * FT, -tgt[1] * FT), stop_at
        g.origin = (org[0] * FT, -org[1] * FT) if org else None
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        n = round(seconds / m.opt.timestep); ctrl, rows = [], []
        for i in range(n):
            g.step(); ctrl.append(d.ctrl.tolist())
            mujoco.mj_step(m, d)
            if (i + 1) % 10 == 0:
                rows.append([d.time] + d.qpos.tolist())
        out = a.out / f"{name}.json"
        out.write_text(json.dumps(dict(scenario=name, mujoco=mujoco.__version__, model_sha256=hashlib.sha256(xml).hexdigest(),
                                       timestep=m.opt.timestep, steps=n, qpos0=q0.tolist(), qvel0=v0.tolist(), target=g.target,
                                       origin=g.origin, stop_at=stop_at, every=10, ctrl=ctrl, rows=rows)))
        print(f"{name}: python mujoco {mujoco.__version__}, {n} steps, final page pos = ({d.qpos[0]/FT:.4f}, {d.qpos[2]/FT:.4f}, {-d.qpos[1]/FT:.4f}) ft, "
              f"pushes {g.pushes}, standing {g.holding and g.latched} -> {out}")
