#!/usr/bin/env node
// Does the browser engine match Python? USAGE: node sim/compare_wasm.mjs [runs/compare]
// For every JSON written by export_run.py, in the vendored wasm build (../vendor/mujoco):
//   1. replay  : Python's recorded controls, step for step -> engine agreement
//   2. closed  : the JS port of Gait (../orbi-physics.js) drives it -> controller + engine agreement
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { FT, Gait, Terrain, PINNED_MUJOCO } from '../orbi-physics.js';

const here = dirname(fileURLToPath(import.meta.url));
const dir = resolve(process.argv[2] || join(here, 'runs', 'compare'));
const xml = readFileSync(join(here, 'orbi', 'model.xml'));
const sha = createHash('sha256').update(xml).digest('hex');
const mj = await loadMujoco();
const version = mj.mj_versionString();
if (version !== PINNED_MUJOCO) { console.error(`wasm engine ${version}, pinned ${PINNED_MUJOCO}`); process.exit(1); }
let bad = 0;
for (const f of readdirSync(dir).filter((f) => f.endsWith('.json')).sort()) {
  const run = JSON.parse(readFileSync(join(dir, f), 'utf8'));
  if (run.model_sha256 !== sha || run.mujoco !== version) { console.error(`${f}: model or engine differs from the run; rerun export_run.py`); process.exit(1); }
  const load = () => {
    const vfs = new mj.MjVFS(); vfs.addBuffer('model.xml', new Uint8Array(xml));
    const model = mj.MjModel.mj_loadXML('model.xml', vfs); vfs.delete();
    const data = new mj.MjData(model);
    mj.mj_resetData(model, data); data.qpos.set(run.qpos0); data.qvel.set(run.qvel0); mj.mj_forward(model, data);
    return [model, data];
  };
  const fmt = (q) => `(${(q[0] / FT).toFixed(4)}, ${(q[2] / FT).toFixed(4)}, ${(-q[1] / FT).toFixed(4)})`;
  for (const mode of ['replay', 'closed']) {
    const [model, data] = load();
    const gait = new Gait(mj, model, data); gait.terrain = new Terrain(model);
    gait.target = run.target.slice(); gait.origin = run.origin ? run.origin.slice() : null; gait.stop_at = run.stop_at;
    let maxPos = 0, maxAll = 0, maxCtrl = 0, firstCtrl = -1, at1 = 0, at3 = 0;
    const t0 = performance.now();
    for (let i = 0; i < run.steps; i++) {
      if (mode === 'replay') data.ctrl.set(run.ctrl[i]);
      else {
        gait.step();
        const c = data.ctrl, u = run.ctrl[i];
        for (let k = 0; k < 14; k++) { const e = Math.abs(c[k] - u[k]); if (e > maxCtrl) maxCtrl = e; if (e > 1e-9 && firstCtrl < 0) firstCtrl = i; }
      }
      mj.mj_step(model, data);
      if ((i + 1) % run.every === 0) {
        const row = run.rows[(i + 1) / run.every - 1], q = data.qpos;
        const e = Math.hypot(q[0] - row[1], q[1] - row[2], q[2] - row[3]);
        if (e > maxPos) maxPos = e;
        for (let k = 0; k < q.length; k++) maxAll = Math.max(maxAll, Math.abs(q[k] - row[k + 1]));
        if (i + 1 === 1000) at1 = maxPos; if (i + 1 === 3000) at3 = maxPos;
      }
    }
    const ms = performance.now() - t0, last = run.rows[run.rows.length - 1], q = data.qpos;
    const fin = Math.hypot(q[0] - last[1], q[1] - last[2], q[2] - last[3]);
    console.log(`${run.scenario} ${mode.padEnd(6)}: wasm ${version}, ${run.steps} steps in ${ms.toFixed(0)} ms (${(ms / run.steps * 1000).toFixed(0)} us/step); ` +
      `final page pos wasm ${fmt(q)} vs python ${fmt(last.slice(1))} ft; final diff ${(fin * 1000).toExponential(2)} mm; ` +
      `max centre diff ${(maxPos * 1000).toExponential(2)} mm (first 1 s ${(at1 * 1000).toExponential(2)}, first 3 s ${(at3 * 1000).toExponential(2)}); max |dqpos| ${maxAll.toExponential(2)}` +
      (mode === 'closed' ? `; max |dctrl| ${maxCtrl.toExponential(2)} m, first step with |dctrl| > 1e-9: ${firstCtrl < 0 ? 'none' : firstCtrl}` : ''));
    if (maxPos > 1e-4) bad++;   // 0.1 mm over the whole run
    data.delete(); model.delete();
  }
}
process.exit(bad ? 1 : 0);
