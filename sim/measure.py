#!/usr/bin/env python3
"""Measure Orbi's gait in MuJoCo. USAGE: python measure.py [settle reach sprint stop steps] [--out runs]

Prints one block per scenario; writes runs/<name>/trace.json for sprint, stop and the tallest step climbed.
"""
import argparse, json, os, sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
KIT = Path(os.environ.get("ORBI_SIM_KIT", "scripts"))  # folder holding the simulation kit's simlib.py
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(HERE))
import simlib  # noqa: E402
import build_model as bm  # noqa: E402
from controller import Gait, Terrain  # noqa: E402

G = 9.81


def make(kind, step_h=0.122):
    simlib.check_engine()
    m = mujoco.MjModel.from_xml_string(bm.build(kind, step_h))
    d = mujoco.MjData(m)
    simlib.reset_to_start(m, d)
    return m, d


def run(m, d, gait, seconds, ground=lambda p: 0.0, record_every=0, until=None):
    """Feedback loop with the kit's observation timing: ctrl -> forward -> record -> step."""
    rows, log = [], dict(t=[], pos=[], vel=[], touch=[], frc=[], ext=[])
    n = round(seconds / m.opt.timestep)
    tid = [m.sensor(f"touch{i+1:02d}").adr[0] for i in range(14)]
    for i in range(n):
        if gait:
            gait.step(ground(d.qpos[:3]))
        mujoco.mj_forward(m, d)
        if record_every and i % record_every == 0:
            rows.append(simlib.observe(m, d))
        log["t"].append(d.time); log["pos"].append(d.qpos[:3].copy()); log["vel"].append(d.qvel[:3].copy())
        log["touch"].append(d.sensordata[tid].copy()); log["frc"].append(d.actuator_force.copy()); log["ext"].append(d.qpos[7:21].copy())
        if until and until(d):
            break
        mujoco.mj_step(m, d)
        if simlib.bad_state_count(d) or not np.isfinite(d.qpos).all():
            raise SystemExit(f"DIVERGED at t={d.time:.3f}")
    return {k: np.array(v) for k, v in log.items()}, rows


def save(out, name, m, rows, every):
    if rows:
        simlib.write_json(out / name / "trace.json", {"meta": {"model": "orbi", "timestep": m.opt.timestep,
                          "record_every": every, "mujoco": mujoco.__version__}, "rows": rows})


def settle(out):
    m, d = make("flat")
    L, _ = run(m, d, None, 2.0)
    mujoco.mj_forward(m, d)
    fn = 0.0
    for i in range(d.ncon):
        f6 = np.zeros(6); mujoco.mj_contactForce(m, d, i, f6); fn += f6[0]
    print(f"settle: weight m*g = {bm.MASS_TOTAL*G:.2f} N, contact normal force = {fn:.2f} N, "
          f"centre height = {d.qpos[2]*1000:.3f} mm (R = {bm.R*1000:.1f}), speed = {np.linalg.norm(d.qvel[:3]):.2e} m/s")


def reach(out):
    rng = np.random.default_rng(0)
    q = rng.normal(size=(200000, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    row = np.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], 1)   # world-Z component of each body axis
    down = -(row @ np.array(bm.AXES).T)
    cnt = (bm.contact_ext(down) <= bm.STROKE).sum(1)      # rim contact, the model's own panel shape
    print(f"reach: over 200000 random orientations, panels that can touch flat ground within the stroke: "
          f"min {cnt.min()}, mean {cnt.mean():.2f}, max {cnt.max()}")


def sprint(out):
    m, d = make("flat")
    g = Gait(m, d); g.target, g.stop_at = (8.0, 0.0), False
    L, rows = run(m, d, g, 14.0, record_every=10, until=lambda d: d.qpos[0] > 6.0)
    x, t, sp = L["pos"][:, 0], L["t"], np.hypot(L["vel"][:, 0], L["vel"][:, 1])
    if x[-1] < 5:
        print(f"sprint: FAILED to reach 5 m in {t[-1]:.1f} s (got {x[-1]:.2f} m, {g.pushes} pushes)")
    else:
        t1, t5 = t[np.argmax(x > 1)], t[np.argmax(x > 5)]
        print(f"sprint: 0->5 m in {t5:.2f} s; mean speed over 1-5 m = {4/(t5-t1):.2f} m/s; peak speed = {sp.max():.2f} m/s; "
              f"pushes = {g.pushes}; max hop (centre above R) = {(L['pos'][:,2].max()-bm.R)*1000:.0f} mm; "
              f"sideways drift at 5 m = {L['pos'][np.argmax(x>5),1]*1000:.0f} mm")
    print(f"        peak panel contact force = {L['touch'].max():.0f} N ({L['touch'].max()/(bm.MASS_TOTAL*G):.1f} x weight); "
          f"actuator at force limit {100*(np.abs(L['frc']).max(1) >= bm.ACT_FORCE*0.999).mean():.1f} % of the time")
    save(out, "sprint", m, rows, 10)


def stop(out):
    m, d = make("flat")
    g = Gait(m, d); g.target = (2.0, 0.0)
    L, rows = run(m, d, g, 10.0, record_every=10)
    err = np.hypot(L["pos"][:, 0] - 2.0, L["pos"][:, 1])
    sp = np.hypot(L["vel"][:, 0], L["vel"][:, 1])
    still = np.where(sp > 0.02)[0]
    t_still = L["t"][still[-1]] if len(still) else 0
    print(f"stop: target 2.00 m; final error = {err[-1]*1000:.0f} mm; at rest (<0.02 m/s) from t = {t_still:.2f} s; "
          f"overshoot past target = {max(0, L['pos'][:,0].max()-2.0)*1000:.0f} mm; panels down at end = {(L['ext'][-1] > 0.01).sum()}")
    save(out, "stop", m, rows, 10)


def steps(out):
    best = None
    for h_mm in (40, 60, 80, 100, 122, 150, 180, 213):
        h = h_mm / 1000
        m, d = make("step", h)
        g = Gait(m, d); g.target, g.stop_at = (6.0, 0.0), False
        g.terrain, g.origin, g.rise_max = Terrain(m), (0.0, 0.0), 0.25   # rise_max raised here only, to find the limit
        edge = 1.5
        ok = lambda d: d.qpos[0] > edge + bm.R + 0.3 and abs(d.qpos[2] - (h + bm.R)) < 0.03   # past the edge and resting on top
        L, rows = run(m, d, g, 15.0, ground=lambda p: h if p[0] > edge else 0.0, record_every=10, until=ok)
        up = ok(d)
        print(f"step {h_mm:3d} mm: {'CLIMBED in %.2f s' % L['t'][-1] if up else 'not climbed in 15 s'} "
              f"(x = {d.qpos[0]:.2f} m, centre z = {d.qpos[2]*1000:.0f} mm, pushes = {g.pushes}, peak panel force = {L['touch'].max():.0f} N)")
        if up:
            best = (h_mm, m, rows)
    if best:
        save(out, f"step_{best[0]}mm", best[1], best[2], 10)
    print(f"steps: tallest single step climbed = {best[0] if best else 0} mm (page claims 213 mm; page stairs are 122 mm each)")


# the page's autopilot route (page feet, x and z); the page switches to the next point within 0.9 ft
WAY_PAGE_ORIGINAL = [[3,0],[12,0],[12.2,6.5],[12.2,9.3],[5,9.3],[2,7.8],[2,4.6],[-3.2,7.6],[-3.2,11.4],[-7.5,10.4],[-7.5,4.2],[-4,3],[-1,1.5],[-6,-2],[-2,-5],[5.5,-4.2],[5,-1.5],[0,0]]
# the route the physics can do: round the east end of the 274 mm pipe instead of over it, and past
# the 213 mm block instead of onto it (both are taller than the 180 mm the rover climbs)
WAY = [[3,0],[12,0],[12.2,6.5],[12.2,9.3],[5,9.3],[2,7.8],[2,4.6],[0.4,7.6],[0.4,10.2],[-3.2,11.4],[-7.5,10.4],[-7.5,4.2],[-4,3],[-1,1.5],[-6,-2],[-2,-2.9],[5.5,-4.2],[5,-1.5],[0,0]]


def follow(way, seconds=400.0, stuck=45.0, gait_kw=None, verbose=True, start=None, quat=None, use_terrain=True, line=True, recover=25.0, hook=None, kind="course", corner=True):
    """Drive the waypoint list once on the full course. Returns (waypoints reached, time, log)."""
    m, d = make(kind)
    if start is not None:
        d.qpos[0], d.qpos[1] = start[0] * bm.FT, -start[1] * bm.FT
    if quat is not None:
        d.qpos[3:7] = quat
    g = Gait(m, d); g.stop_at = False
    for k, v in (gait_kw or {}).items():
        setattr(g, k, v)
    ter = Terrain(m)
    g.terrain = ter if use_terrain else None
    wp, t_wp, zmax, reached, backing, retries = 0, 0.0, 0.0, 0, False, 0
    n = round(seconds / m.opt.timestep)
    pos = []
    for i in range(n):
        w = way[wp]
        tx, ty = w[0] * bm.FT, -w[1] * bm.FT
        g.target = (tx, ty)
        prev = way[wp - 1] if wp else start if start is not None else (0.0, 0.0)
        g.origin = (prev[0] * bm.FT, -prev[1] * bm.FT) if line else None
        if corner:
            # ease into corners as the page does (orbi-physics.js route): carry less speed through a sharper turn
            nxt = way[(wp + 1) % len(way)]
            a1, a2 = np.array(w, float) - np.array(prev, float), np.array(nxt, float) - np.array(w, float)
            n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
            c = float(a1 @ a2 / (n1 * n2)) if n1 > 1e-6 and n2 > 1e-6 else 1.0
            g.v_end = g.cruise * max(0.3, (1 + c) / 2)
        ox, oy = prev[0] * bm.FT, -prev[1] * bm.FT
        L = float(np.hypot(tx - ox, ty - oy))
        along = ((d.qpos[0] - ox) * (tx - ox) + (d.qpos[1] - oy) * (ty - oy)) / max(L, 1e-9)
        across = abs((d.qpos[0] - ox) * (ty - oy) - (d.qpos[1] - oy) * (tx - ox)) / max(L, 1e-9)
        # arrival as on the page (orbi-physics.js route): within 0.9 ft, or past the leg's finish line
        if np.hypot(tx - d.qpos[0], ty - d.qpos[1]) < 0.9 * bm.FT or (line and not backing and L > 1e-6 and along >= L and across < 2.5 * bm.FT):
            if verbose:
                print(f"  wp {wp:2d} {w} reached at t = {d.time:6.2f} s (leg {d.time - t_wp:5.2f} s, pushes {g.pushes}, levers {g.levers})")
            reached += 0 if backing else 1; wp += 1; t_wp = d.time; backing = False
            if wp == len(way):
                break
            continue
        g.step(ter.height(d.qpos[0], d.qpos[1]))
        mujoco.mj_step(m, d)
        if hook:
            hook(m, d, g, ter)
        if i % 20 == 0:
            pos.append(d.qpos[:3].copy())
        zmax = max(zmax, d.qpos[2])
        if not np.isfinite(d.qpos).all():
            raise SystemExit(f"DIVERGED at t={d.time:.3f}")
        if recover and d.time - t_wp > recover and wp > 0 and not backing:
            # the page's autopilot rule: no arrival in `recover` seconds -> go back one waypoint and retry
            if verbose:
                print(f"  no arrival at wp {wp} {w} after {recover:.0f} s at page ({d.qpos[0]/bm.FT:.2f}, {-d.qpos[1]/bm.FT:.2f}): back to wp {wp-1}")
            wp -= 1; t_wp = d.time; backing = True; retries += 1
        if d.time - t_wp > stuck:
            if verbose:
                print(f"  STUCK heading for wp {wp} {w}: page pos ({d.qpos[0]/bm.FT:.2f}, {-d.qpos[1]/bm.FT:.2f}), centre z = {d.qpos[2]*1000:.0f} mm")
            break
    return reached, d.time, zmax, np.array(pos)


def course(out):
    r, t, zmax, pos = follow(WAY)
    print(f"course: {r} / {len(WAY)} waypoints in {t:.1f} s; highest centre = {zmax*1000:.0f} mm "
          f"(platform top + R = {(1.6*bm.FT+bm.R)*1000:.0f} mm)")


class Bounce:
    """Bounce numbers for one drive. `hop` = centre height above its resting height on open floor
    (no terrain within 1.5 R); `vz_rms` = RMS vertical speed of the centre there; `peak` = largest
    total contact normal force (all contacts of the rover summed), i.e. the hardest landing or shove."""

    def __init__(self):
        self.hop, self.vz2, self.n, self.peak, self.peak_t, self.air = 0.0, 0.0, 0, 0.0, 0.0, 0
        self.N, self.air_all, self.vz2_all, self.hard, self.hard_t, self.vdown = 0, 0, 0.0, 0, -9.0, 0.0
        self.ring = [(np.cos(a) * 1.5 * bm.R, np.sin(a) * 1.5 * bm.R) for a in np.arange(8) * np.pi / 4] + [(0.0, 0.0)]

    def __call__(self, m, d, g, ter):
        fn, f6 = 0.0, np.zeros(6)
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, f6); fn += f6[0]
        if fn > self.peak:
            self.peak, self.peak_t = fn, d.time
        self.N += 1; self.air_all += d.ncon == 0; self.vz2_all += d.qvel[2] ** 2; self.vdown = max(self.vdown, -d.qvel[2])
        if fn > 5 * bm.MASS_TOTAL * G and d.time - self.hard_t > 0.3:
            self.hard += 1; self.hard_t = d.time
        x, y = d.qpos[0], d.qpos[1]
        if d.time > 0.3 and all(ter.height(x + a, y + b) == 0.0 for a, b in self.ring):
            self.hop = max(self.hop, d.qpos[2] - bm.R)
            self.vz2 += d.qvel[2] ** 2; self.n += 1
            self.air += d.ncon == 0

    def line(self):
        return (f"max hop on open floor = {self.hop*1000:.1f} mm; RMS vertical speed there = {np.sqrt(self.vz2/max(self.n,1)):.3f} m/s; "
                f"airborne {100*self.air/max(self.n,1):.1f} % of open-floor time; peak total contact force = {self.peak:.0f} N "
                f"({self.peak/(bm.MASS_TOTAL*G):.1f} x weight, at t = {self.peak_t:.1f} s)\n"
                f"               whole drive: airborne {100*self.air_all/max(self.N,1):.1f} % of the time; RMS vertical speed = {np.sqrt(self.vz2_all/max(self.N,1)):.3f} m/s; "
                f"fastest descent = {self.vdown:.2f} m/s; contacts over 5 x weight = {self.hard}")


FLAT_TOUR = [[13, 0], [13, 10], [3, 10], [3, 0], [9, 5]]   # page feet, on the flat model: straights and square turns


def bounce(out, gait_kw=None, do_course=True):
    b = Bounce()
    r, t, zmax, pos = follow(FLAT_TOUR, seconds=90.0, verbose=False, hook=b, kind="flat", gait_kw=gait_kw)
    print(f"bounce flat:   {r} / {len(FLAT_TOUR)} waypoints in {t:.1f} s; {b.line()}")
    if not do_course:
        return
    b = Bounce()
    r, t, zmax, pos = follow(WAY, verbose=False, hook=b, gait_kw=gait_kw)
    print(f"bounce course: {r} / {len(WAY)} waypoints in {t:.1f} s; highest centre = {zmax*1000:.0f} mm; {b.line()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["settle", "reach", "sprint", "stop", "steps"])
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    a = ap.parse_args()
    print(f"mujoco {mujoco.__version__}; ASSUMED mass {bm.MASS_TOTAL} kg, actuator {bm.ACT_FORCE} N, panel speed 1.0 m/s, friction {bm.FRICTION}")
    for w in a.which:
        globals()[w](a.out)
