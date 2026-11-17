"""Success rate of a route over random start orientations. USAGE: python tune.py key=value ..."""
import sys, numpy as np, measure as M
kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:] if "=" in a and not a.startswith("_"))}
opt = dict(a.split("=") for a in sys.argv[1:] if a.startswith("_"))
way = eval(opt.get("_way", "[[12,0]]")); start = eval(opt.get("_start", "(1.5,0)")); N = int(opt.get("_n", 8))
rng = np.random.default_rng(1); ok = 0; ts = []; zs = []
for i in range(N):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    r, t, z, pos = M.follow(way, seconds=float(opt.get("_T", 40)), stuck=float(opt.get("_T", 40)), gait_kw=kw, verbose=False, start=start, quat=q)
    ok += r == len(way); ts.append(t); zs.append(z)
    print(f"  run {i}: {r}/{len(way)} t={t:5.1f} zmax={z*1000:4.0f}mm end=({pos[-1][0]/M.bm.FT:.1f},{-pos[-1][1]/M.bm.FT:.1f})")
print(f"{kw}: {ok}/{N} ok, mean t {np.mean(ts):.1f}s, max centre {max(zs)*1000:.0f} mm")
