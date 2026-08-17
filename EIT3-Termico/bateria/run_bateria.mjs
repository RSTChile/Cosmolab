import fs from 'node:fs';
import path from 'node:path';
import { fork } from 'node:child_process';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PARAMS_FIJOS = { noise: 0.0079, band: 1.105, tOpt: 25, ptcTc: 18, ptcSharp: 4.1 };
const EJE = { from: 0.25, to: 1.95, steps: 60, settle: 300, measure: 120 };

function buildJobsA() {
  const jobs = [];
  const params = { powerBase: 0.47, beta: 0.94, sigma: 6.8, ...PARAMS_FIJOS };
  for (let seed = 1; seed <= 30; seed++) jobs.push({ seed, ...EJE, params });
  return jobs;
}

function buildJobsB() {
  const jobs = [];
  const betas = [0.80, 0.88, 0.94, 0.98], sigmas = [3.0, 5.0, 6.8, 8.0], powerBases = [0.30, 0.47, 0.65];
  for (const beta of betas) for (const sigma of sigmas) for (const powerBase of powerBases) {
    const params = { powerBase, beta, sigma, ...PARAMS_FIJOS };
    for (let seed = 1; seed <= 10; seed++) jobs.push({ seed, ...EJE, params });
  }
  return jobs;
}

const exp = process.argv[2];
const outCsv = process.argv[3];
if (!exp || !outCsv) { console.error('uso: node run_bateria.mjs A|B salida.csv'); process.exit(1); }
const jobs = exp === 'A' ? buildJobsA() : buildJobsB();
console.log(`Experimento ${exp}: ${jobs.length} barridos, ~${jobs.length * 30000} pasos de física.`);

const N_WORKERS = Math.max(1, Math.min(14, os.cpus().length - 1, jobs.length));
const shards = Array.from({ length: N_WORKERS }, () => []);
jobs.forEach((j, i) => shards[i % N_WORKERS].push(j));
console.log(`Repartiendo en ${N_WORKERS} procesos.`);

const tmpDir = path.join(__dirname, '.tmp_shards');
fs.mkdirSync(tmpDir, { recursive: true });

const t0 = Date.now();
const promises = shards.map((shard, i) => {
  const inPath = path.join(tmpDir, `in_${exp}_${i}.json`);
  const outPath = path.join(tmpDir, `out_${exp}_${i}.json`);
  fs.writeFileSync(inPath, JSON.stringify(shard));
  return new Promise((resolve, reject) => {
    const child = fork(path.join(__dirname, 'worker_shard.mjs'), [inPath, outPath], { stdio: 'inherit' });
    child.on('exit', code => code === 0 ? resolve(outPath) : reject(new Error(`worker ${i} exit ${code}`)));
    child.on('error', reject);
  });
});

Promise.all(promises).then(outPaths => {
  const allRows = [];
  for (const p of outPaths) allRows.push(...JSON.parse(fs.readFileSync(p, 'utf8')));
  allRows.sort((a, b) => a.semilla - b.semilla || a.beta - b.beta || a.sigma - b.sigma || a.potencia_base - b.potencia_base || a.luminosidad - b.luminosidad);
  const header = Object.keys(allRows[0]);
  const csvLines = [header.join(',')];
  for (const r of allRows) csvLines.push(header.map(h => r[h]).join(','));
  fs.writeFileSync(outCsv, csvLines.join('\n'));
  console.log(`Listo: ${allRows.length} filas en ${(Date.now() - t0) / 1000}s -> ${outCsv}`);
}).catch(err => { console.error('ERROR en batería:', err); process.exit(1); });
