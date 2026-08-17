import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';

const DIR = import.meta.dirname;
const N_WORKERS = 14;

// Parámetros fijos comunes (calibración v7.7: tc_ptc=20, exponente_ptc=16 default)
const FIJOS = { axis:'luminosity', from:0.6, to:1.4, steps:60, settle:300, measure:120,
  sigma:6.8, noise:0.0079, band:1.105, ptcTc:20 };

const jobs = [];
// D: 10 semillas x 2 modos
for (let seed=1; seed<=10; seed++){
  for (const modo of ['parada','inicio']){
    jobs.push({ experimento:'D', seed, modo, ...FIJOS, powerBase:0.47, beta:0.94, tOpt:25, ptcSharp:16 });
  }
}
// A': 30 semillas, modo=parada
for (let seed=1; seed<=30; seed++){
  jobs.push({ experimento:'Aprima', seed, modo:'parada', ...FIJOS, powerBase:0.47, beta:0.94, tOpt:25, ptcSharp:16 });
}
// B': grilla completa 4x3x3x3 x 10 semillas, modo=parada
const BETAS=[0.80,0.88,0.94,0.98], TOPTS=[22,25,28], SHARPS=[8,12,16], PBASES=[0.30,0.47,0.65];
for (const beta of BETAS) for (const tOpt of TOPTS) for (const ptcSharp of SHARPS) for (const powerBase of PBASES){
  for (let seed=1; seed<=10; seed++){
    jobs.push({ experimento:'Bprima', seed, modo:'parada', ...FIJOS, powerBase, beta, tOpt, ptcSharp });
  }
}

console.log(`Total jobs: ${jobs.length} (D=20, A'=30, B'=${jobs.length-50})`);
fs.writeFileSync(`${DIR}/jobs.json`, JSON.stringify(jobs));

// NO se limpia work/ al arrancar: worker_shard.mjs ahora resume desde
// progreso_shard{i}.log si ya existe (escritura incremental por job). Si de
// verdad se necesita una corrida 100% desde cero, borrar work/ a mano antes.
const OUT = `${DIR}/work`;
fs.mkdirSync(OUT, { recursive: true });

const t0 = Date.now();
let terminados = 0;
const totalPorShard = jobs.map((_,i)=>i%N_WORKERS);
const countPerShard = {};
for (const s of totalPorShard) countPerShard[s]=(countPerShard[s]||0)+1;

function logProgreso(){
  let hechos = 0;
  for (let i=0;i<N_WORKERS;i++){
    const p = `${OUT}/progreso_shard${i}.log`;
    if (fs.existsSync(p)) hechos += fs.readFileSync(p,'utf8').split('\n').filter(Boolean).length;
  }
  const dt = (Date.now()-t0)/1000;
  fs.appendFileSync(`${DIR}/orquestador_v77.log`, `[${new Date().toISOString()}] progreso: ${hechos}/${jobs.length} (${dt.toFixed(0)}s)\n`);
}
const iv = setInterval(logProgreso, 30000);

const procesos = [];
for (let i=0;i<N_WORKERS;i++){
  const p = spawn(process.execPath, [`${DIR}/worker_shard.mjs`, `${DIR}/jobs.json`, String(i), OUT], {
    env: { ...process.env, N_WORKERS: String(N_WORKERS) },
    stdio: 'inherit',
  });
  procesos.push(new Promise((resolve)=>{ p.on('exit', (code)=>{ terminados++; resolve(code); }); }));
}

Promise.all(procesos).then((codigos)=>{
  clearInterval(iv);
  logProgreso();
  const dt=(Date.now()-t0)/1000;
  const fallidos = codigos.filter(c=>c!==0).length;
  fs.appendFileSync(`${DIR}/orquestador_v77.log`, `[${new Date().toISOString()}] TERMINADO: ${jobs.length} jobs, ${fallidos} fallidos, ${(dt/3600).toFixed(2)} h.\n`);

  // Concatenar CSVs por experimento
  const HEADER = 'experimento,semilla,modo,potencia_base,beta,sigma,ruido,banda,t_optima,tc_ptc,exponente_ptc,dia_noche,luminosidad,entropia_abs_local,entropia_piso_local,entropia_abs_global,entropia_piso_global,entropia_rel,huella,lambda,acoplamiento,tasa_error,potencia_viva,rango_potencia_viva,banda_potencia_viva,piso_ruido,valores_distintos,multiplicidad,diagnostico,saturacion_sensor,pasos_recuperacion,recuperacion_completa,recuperacion_mediana,recuperacion_topes,asentamiento_pasos,asentamiento_ok,tasa_recuperacion,varianza_potencia_viva,autocorrelacion1_potencia_viva';
  const porExperimento = { D: [], Aprima: [], Bprima: [] };
  for (let i=0;i<N_WORKERS;i++){
    const p = `${OUT}/parcial_shard${i}.csv`;
    if (!fs.existsSync(p)) continue;
    const lineas = fs.readFileSync(p,'utf8').split('\n').filter(Boolean);
    for (const linea of lineas){
      const exp = linea.split(',')[0];
      if (porExperimento[exp]) porExperimento[exp].push(linea);
    }
  }
  const nombreArchivo = { D:'experimento_D_reinicio.csv', Aprima:'experimento_Aprima_repeticion.csv', Bprima:'experimento_Bprima_multivariable.csv' };
  for (const exp of Object.keys(porExperimento)){
    fs.writeFileSync(`${DIR}/${nombreArchivo[exp]}`, HEADER+'\n'+porExperimento[exp].join('\n')+'\n');
  }
  fs.appendFileSync(`${DIR}/orquestador_v77.log`, `[${new Date().toISOString()}] CSVs escritos: D=${porExperimento.D.length} A'=${porExperimento.Aprima.length} B'=${porExperimento.Bprima.length} filas\n`);
  console.log('LISTO');
});
