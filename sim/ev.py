import sys, json, measure as M, numpy as np, mujoco
kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
last=[-9,-9]
def hook(m,d,g,ter):
    f6=np.zeros(6); fn=0
    for i in range(d.ncon):
        mujoco.mj_contactForce(m,d,i,f6); fn+=f6[0]
    x,y=d.qpos[0],d.qpos[1]; gz=ter.height(x,y); h=d.qpos[2]-gz-M.bm.R
    if fn>600 and d.time-last[0]>0.3:
        last[0]=d.time; print("  F %5.0f N t %.2f page (%.1f,%.1f) h+%.0f mm vz %.2f sp %.2f climb %s holding %s nExt %d"%(fn,d.time,x/.3048,-y/.3048,h*1000,d.qvel[2],np.hypot(*d.qvel[:2]),getattr(g,'climbing',None),g.holding,(d.qpos[7:]>0.02).sum()))
    if h>0.03 and d.ncon==0 and d.time-last[1]>0.3:
        last[1]=d.time; print("  AIR t %.2f page (%.1f,%.1f) h+%.0f mm vz %.2f sp %.2f"%(d.time,x/.3048,-y/.3048,h*1000,d.qvel[2],np.hypot(*d.qvel[:2])))
M.follow(M.WAY, gait_kw=kw, verbose=False, hook=hook)
