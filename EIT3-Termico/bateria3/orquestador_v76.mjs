// Orquestador de Experimento D + A' + B' (v7.6.1). Mismo patrón que
// bateria2/orquestador_v75.mjs: cola dinámica sobre N workers (IPC), CSV
// incremental, progreso.json pollable. Columnas nuevas de v7.6.1: asentamiento
// hasta equilibrio + las 5 repeticiones individuales de recuperación.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fork } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { buildAllJobs } from './jobs_v76.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const N_WORKERS = Number(process.argv[2]) || Math.max(1, os.cpus().length - 2);
const MODO_SMOKE = process.argv[3] === 'smoke';

const HEADER = [
  'semilla','modo','potencia_base','beta','sigma','ruido','banda','t_optima','tc_ptc','exponente_ptc','dia_noche','luminosidad',
  'asent_pasos','asent_ok',
  'pasos_recuperacion','rec_mediana','rec_topes','rec_1','rec_2','rec_3','rec_4','rec_5','recuperacion_completa','tasa_recuperacion',
  'varianza_potencia_viva','autocorrelacion1_potencia_viva',
  'entropia_abs_local','entropia_piso_local','entropia_abs_global','entropia_piso_global','entropia_rel',
  'huella','lambda','acoplamiento','tasa_error','potencia_viva','rango_potencia_viva','banda_potencia_viva',
  'piso_ruido','valores_distintos','multiplicidad','diagnostico','saturacion_sensor',
];

function filaCSV(r) {
  return [
    r.semilla, r.modo, r.potencia_base, r.beta, r.sigma, r.ruido, r.banda, r.t_optima, r.tc_ptc, r.exponente_ptc, r.dia_noche,
    r.luminosidad.toFixed(4),
    r.asent_pasos, r.asent_ok,
    r.pasos_recuperacion.toFixed(1), r.rec_mediana, r.rec_topes, r.rec_1, r.rec_2, r.rec_3, r.rec_4, r.rec_5, r.recuperaron_todos, r.tasa_recuperacion.toFixed(6),
    r.varianza_pl.toExponential(4), r.autocorr1_pl.toFixed(5),
    r.entropia_abs_local.toFixed(4), r.entropia_piso_local.toFixed(4), r.entropia_abs_global.toFixed(4), r.entropia_piso_global.toFixed(4), r.entropia_rel.toFixed(4),
    r.huella.toFixed(4), r.lambda.toFixed(5), r.acoplamiento.toFixed(4), r.tasa_error.toFixed(6), r.potencia_viva.toFixed(4),
    r.rango_potencia_viva.toFixed(5), r.banda_potencia_viva.toFixed(5), r.piso_ruido.toFixed(5), r.valores_distintos, r.multiplicidad.toFixed(4),
    r.diagnostico, r.saturacion_sensor,
  ].join(',');
}

const ARCHIVOS = {
  D: path.join(__dirname, 'experimento_D_reinicio.csv'),
  Aprima: path.join(__dirname, 'experimento_Aprima_repeticion.csv'),
  Bprima: path.join(__dirname, 'experimento_Bprima_multivariable.csv'),
};
for (const p of Object.values(ARCHIVOS)) fs.writeFileSync(p, HEADER.join(',') + '\n');

const LOG_PATH = path.join(__dirname, 'orquestador_v76.log');
const PROGRESO_PATH = path.join(__dirname, 'progreso_v76.json');
const MARCADOR_PATH = path.join(__dirname, 'TERMINADO_v76');
function log(msg) {
  const linea = `[${new Date().toISOString()}] ${msg}`;
  console.log(linea);
  fs.appendFileSync(LOG_PATH, linea + '\n');
}

function buildJobsSmoke() {
  const eje = { from: 0.9, to: 1.4, steps: 4, settle: 20, measure: 20 };
  const base = { powerBase: 0.47, beta: 0.94, sigma: 6.8, noise: 0.0079, band: 1.105, tOpt: 25, ptcTc: 18, ptcSharp: 4.1 };
  const jobs = [];
  for (const tag of ['D', 'Aprima', 'Bprima']) {
    for (let seed = 1; seed <= 2; seed++) jobs.push({ tag, seed, modo: 'parada', params: { ...base }, ...eje });
  }
  return jobs;
}

const jobs = MODO_SMOKE ? buildJobsSmoke() : buildAllJobs();
const total = jobs.length;
let siguiente = 0;
let completados = 0;
let fallidos = 0;
const tiemposJob = [];
const porTag = { D: { done: 0, total: 0 }, Aprima: { done: 0, total: 0 }, Bprima: { done: 0, total: 0 } };
for (const j of jobs) porTag[j.tag].total++;

const t0 = Date.now();

function escribirProgreso() {
  const transcurrido = (Date.now() - t0) / 1000;
  const promedioJob = tiemposJob.length ? tiemposJob.reduce((a, b) => a + b, 0) / tiemposJob.length : null;
  const restantes = total - completados;
  const etaSeg = (promedioJob && N_WORKERS > 0) ? (restantes * promedioJob) / N_WORKERS : null;
  fs.writeFileSync(PROGRESO_PATH, JSON.stringify({
    completados, fallidos, total, porTag,
    transcurrido_s: Math.round(transcurrido),
    promedio_job_s: promedioJob ? Number(promedioJob.toFixed(1)) : null,
    eta_restante_s: etaSeg ? Math.round(etaSeg) : null,
    n_workers: N_WORKERS,
  }, null, 2));
}

log(`Arrancando: ${total} jobs (D=${porTag.D.total}, Aprima=${porTag.Aprima.total}, Bprima=${porTag.Bprima.total}), ${N_WORKERS} workers.`);

let workersActivos = 0;
const workers = [];

function siguienteJob() {
  if (siguiente >= jobs.length) return null;
  return jobs[siguiente++];
}

function lanzarWorker(idx) {
  const w = fork(path.join(__dirname, 'worker_v76_ipc.mjs'), [], { stdio: ['ignore', 'ignore', 'inherit', 'ipc'] });
  workersActivos++;
  w.on('message', (msg) => {
    if (msg.tipo === 'listo') {
      const job = siguienteJob();
      if (job) {
        w.send({ tipo: 'job', job });
      } else {
        w.send({ tipo: 'fin' });
      }
    } else if (msg.tipo === 'resultado') {
      completados++;
      porTag[msg.tag].done++;
      tiemposJob.push(msg.seg);
      if (msg.error) {
        fallidos++;
        log(`ERROR job tag=${msg.tag} seed=${msg.seed} modo=${msg.modo}: ${msg.error}`);
      } else {
        const stream = fs.createWriteStream(ARCHIVOS[msg.tag], { flags: 'a' });
        for (const r of msg.rows) stream.write(filaCSV(r) + '\n');
        stream.end();
      }
      if (completados % 5 === 0 || completados === total) {
        escribirProgreso();
        log(`progreso: ${completados}/${total} (D=${porTag.D.done}/${porTag.D.total} A'=${porTag.Aprima.done}/${porTag.Aprima.total} B'=${porTag.Bprima.done}/${porTag.Bprima.total}) último job ${msg.seg.toFixed(1)}s`);
      }
    }
  });
  w.on('exit', (code) => {
    workersActivos--;
    if (siguiente < jobs.length) {
      log(`worker ${idx} murió (code=${code}) con trabajo pendiente — relanzando.`);
      lanzarWorker(idx);
    } else if (workersActivos === 0) {
      terminar();
    }
  });
  workers.push(w);
}

let terminando = false;
function terminar() {
  if (terminando) return;
  terminando = true;
  escribirProgreso();
  fs.writeFileSync(MARCADOR_PATH, JSON.stringify({ fin: new Date().toISOString(), completados, fallidos, total }, null, 2));
  log(`TERMINADO: ${completados}/${total} jobs, ${fallidos} fallidos, ${((Date.now() - t0) / 1000 / 3600).toFixed(2)} h.`);
  process.exit(fallidos > 0 ? 1 : 0);
}

for (let i = 0; i < N_WORKERS; i++) lanzarWorker(i);
