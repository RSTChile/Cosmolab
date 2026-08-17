// Worker: recibe una lista de jobs (uno por barrido) vía argv (índice de
// shard) y un archivo de trabajo JSON compartido, corre su porción, escribe
// su propio CSV parcial. El orquestador concatena al final.
import { correrBarridoV77 } from './motor_v77.mjs';
import fs from 'node:fs';

const [, , jobsPath, shardIdxStr, outDir] = process.argv;
const shardIdx = Number(shardIdxStr);
const todos = JSON.parse(fs.readFileSync(jobsPath, 'utf8'));
const misJobs = todos.filter((j, i) => i % Number(process.env.N_WORKERS) === shardIdx);

const HEADER = ['experimento','semilla','modo','potencia_base','beta','sigma','ruido','banda',
  't_optima','tc_ptc','exponente_ptc','dia_noche','luminosidad',
  'entropia_abs_local','entropia_piso_local','entropia_abs_global','entropia_piso_global','entropia_rel',
  'huella','lambda','acoplamiento','tasa_error','potencia_viva','rango_potencia_viva','banda_potencia_viva',
  'piso_ruido','valores_distintos','multiplicidad','diagnostico','saturacion_sensor',
  'pasos_recuperacion','recuperacion_completa','recuperacion_mediana','recuperacion_topes',
  'asentamiento_pasos','asentamiento_ok','tasa_recuperacion','varianza_potencia_viva','autocorrelacion1_potencia_viva'];

function filaCSV(job, r){
  return [job.experimento, job.seed, job.modo, job.powerBase, job.beta, job.sigma, job.noise, job.band,
    job.tOpt, job.ptcTc, job.ptcSharp, 0, r.x.toFixed(4),
    r.H_absLocal.toFixed(4), r.H_noiseLocal.toFixed(4), r.H_absGlobal.toFixed(4), r.H_noiseGlobal.toFixed(4), r.H_rel.toFixed(4),
    r.footprint.toFixed(4), r.Lambda.toFixed(5), r.A_sys_env.toFixed(4), r.err_rate.toFixed(6),
    r.powerLive.toFixed(4), r.plRange.toFixed(5), r.plBand.toFixed(5), r.noiseFloor.toFixed(5),
    r.distinct, r.mult.toFixed(4), r.diag, r.ptcSat,
    r.pasos_recuperacion.toFixed(1), r.convergio, r.rec_mediana, r.rec_topes,
    r.asent_pasos, r.asent_ok, r.tasa_recuperacion.toFixed(6), r.varianza_pl.toExponential(4), r.autocorr1_pl.toFixed(5),
  ].join(',');
}

// Escritura INCREMENTAL: cada job se vuelca al CSV apenas termina, no al
// final del shard completo. La corrida anterior murió a mitad de camino (el
// proceso se cortó por fuera) y como esto escribía todo junto al final, se
// perdieron ~135 jobs ya calculados — no vuelve a pasar.
//
// RESUME: si progreso_shard{i}.log ya tiene N líneas, este shard ya completó
// los primeros N jobs de misJobs (el orden es determinístico dado el mismo
// jobs.json) — se saltean, para poder relanzar sin repetir trabajo.
const progresoPath = `${outDir}/progreso_shard${shardIdx}.log`;
const yaHechos = fs.existsSync(progresoPath)
  ? fs.readFileSync(progresoPath,'utf8').split('\n').filter(Boolean).length
  : 0;
const pendientes = misJobs.slice(yaHechos);

const csvPath = `${outDir}/parcial_shard${shardIdx}.csv`;
if (!fs.existsSync(csvPath)) fs.writeFileSync(csvPath, '');
for (const job of pendientes) {
  const rows = correrBarridoV77(job);
  const lineas = rows.map(r => filaCSV(job, r));
  fs.appendFileSync(csvPath, lineas.join('\n')+'\n');
  fs.appendFileSync(progresoPath,
    `[${new Date().toISOString()}] job ${job.experimento} seed=${job.seed} modo=${job.modo} listo\n`);
}
process.exit(0);
