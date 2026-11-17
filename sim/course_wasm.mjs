#!/usr/bin/env node
// The page's drive mode without the page: orbi-physics.js + the vendored wasm, 60 fps frames,
// the autopilot route from index.html. USAGE: node sim/course_wasm.mjs [runs=8] [laps=1] [--trace]
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import loadMujoco from '../vendor/mujoco/mujoco.js';
import { createEngine } from '../orbi-physics.js';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, '..', 'index.html'), 'utf8');
const WAY = JSON.parse(html.match(/const WAY=(\[\[.*?\]\]);/)[1]);
const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const runs = +(args[0] || 8), laps = +(args[1] || 1), trace = process.argv.includes('--trace');
const mj = await loadMujoco();
const eng = createEngine(mj, new Uint8Array(readFileSync(join(here, 'orbi', 'model.xml'))));
let seed = 12345; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
console.log(`route from index.html: ${WAY.length} waypoints; mujoco ${eng.version} wasm; frames of 1/60 s`);
let okAll = 0;
for (let r = 0; r < runs; r++) {
  const q = [gauss(), gauss(), gauss(), gauss()];
  eng.reset({ pos: [0, 1, 0], quat: q });
  let wp = 0, t = 0, vmax = 0, hmax = 0, legMax = 0, legStart = 0, lastReached = 0;
  const limit = 200 * laps;
  while (eng.routeInfo.reached < WAY.length * laps && t < limit) {
    wp = eng.route(WAY, wp, 1 / 60);
    eng.step(1 / 60); t += 1 / 60;
    const s = eng.state;
    vmax = Math.max(vmax, s.speed); hmax = Math.max(hmax, s.pos[1]);
    if (eng.routeInfo.reached !== lastReached) {
      if (trace) console.log(`   wp ${(wp + WAY.length - 1) % WAY.length} reached t=${t.toFixed(1)} leg ${(t - legStart).toFixed(1)} s`);
      legMax = Math.max(legMax, t - legStart); legStart = t; lastReached = eng.routeInfo.reached;
    }
  }
  const ok = eng.routeInfo.reached >= WAY.length * laps; okAll += ok;
  console.log(`run ${r}: ${ok ? 'COMPLETED' : 'NOT completed'} ${eng.routeInfo.reached}/${WAY.length * laps} waypoints in ${t.toFixed(1)} s sim; ` +
    `retries ${eng.routeInfo.retries}; longest leg ${legMax.toFixed(1)} s; peak speed ${vmax.toFixed(2)} m/s; highest centre ${(hmax * 304.8).toFixed(0)} mm; ` +
    `end page (${eng.state.pos[0].toFixed(1)}, ${eng.state.pos[2].toFixed(1)})`);
}
console.log(`${okAll}/${runs} runs completed ${laps} lap(s)`);
