import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const COMMON = { sigma: 6.8, noise: 0.0079, band: 1.105, ptcTc: 18, minTemp: -6, maxTemp: 25 };
const COMBOS = {
  baseline:      { powerBase: 0.47, beta: 0.94, tOpt: 25, ptcSharp: 4.1, ...COMMON },
  tOpt22:        { powerBase: 0.47, beta: 0.94, tOpt: 22, ptcSharp: 4.1, ...COMMON },
  tOpt28:        { powerBase: 0.47, beta: 0.94, tOpt: 28, ptcSharp: 4.1, ...COMMON },
  ptcSharp3:     { powerBase: 0.47, beta: 0.94, tOpt: 25, ptcSharp: 3.0, ...COMMON },
  ptcSharp6:     { powerBase: 0.47, beta: 0.94, tOpt: 25, ptcSharp: 6.0, ...COMMON },
  beta080:       { powerBase: 0.47, beta: 0.80, tOpt: 25, ptcSharp: 4.1, ...COMMON },
  beta098:       { powerBase: 0.47, beta: 0.98, tOpt: 25, ptcSharp: 4.1, ...COMMON },
  extremo_combo: { powerBase: 0.30, beta: 0.80, tOpt: 28, ptcSharp: 6.0, ...COMMON },
};

fs.mkdirSync('muestras_topes', { recursive: true });
const procs = [];
for (const [name, params] of Object.entries(COMBOS)) {
  const outPath = path.join('muestras_topes', `${name}.json`);
  const logPath = path.join('muestras_topes', `${name}.log`);
  const logFd = fs.openSync(logPath, 'w');
  const child = spawn('node', ['muestra_topes_worker.mjs', outPath, JSON.stringify(params)], {
    stdio: ['ignore', logFd, logFd],
    detached: true,
  });
  child.unref();
  console.log(`${name}: PID ${child.pid}`);
  procs.push(child.pid);
}
fs.writeFileSync('muestras_topes/pids.json', JSON.stringify(procs));
