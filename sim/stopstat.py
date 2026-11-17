"""Stop behaviour over random start orientations on flat ground. USAGE: python stopstat.py [N] key=value..."""
import sys, numpy as np, mujoco, measure as M
from controller import Gait
N = int(sys.argv[1]) if len(sys.argv) > 1 and "=" not in sys.argv[1] else 30
kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:] if "=" in a)}
rng = np.random.default_rng(2); res = []
for i in range(N):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    m, d = M.make("flat"); d.qpos[3:7] = q
    g = Gait(m, d)
    for k, v in kw.items(): setattr(g, k, v)
    g.target = (2.0, 0.0)
    t_rest = 0.0
    for _ in range(12000):
        g.step(); mujoco.mj_step(m, d)
        if np.linalg.norm(d.qvel[:3]) > 0.02: t_rest = d.time
    err = np.hypot(d.qpos[0] - 2, d.qpos[1]); sp = np.linalg.norm(d.qvel[:3]); w = np.linalg.norm(d.qvel[3:6])
    down = int((d.qpos[7:21] > 0.01).sum())
    res.append((err, sp, down, g.latched, d.qpos[2], t_rest))
    if not g.latched or sp > 0.02 or err > 0.3:
        print(f"  run {i}: err {err*1000:.0f} mm, speed {sp:.3f} m/s, spin {w:.3f} rad/s, panels down {down}, latched {g.latched}, holding {g.holding}, z {d.qpos[2]*1000:.0f}")
r = np.array(res, float)
print(f"stop x{N} {kw}: latched {int(r[:,3].sum())}/{N}; error mean {r[:,0].mean()*1000:.0f} max {r[:,0].max()*1000:.0f} mm; "
      f"final speed max {r[:,1].max():.4f} m/s; at rest from t mean {r[:,5].mean():.1f} max {r[:,5].max():.1f} s; centre z {r[:,4].min()*1000:.0f}-{r[:,4].max()*1000:.0f} mm; panels down min {int(r[:,2].min())} max {int(r[:,2].max())}")
