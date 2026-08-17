import { correrBarridoV77 } from './motor_v77.mjs';

function medir(nombre, params){
  const t0 = Date.now();
  const rows = correrBarridoV77(params);
  const dt = (Date.now()-t0)/1000;
  const noConverg = rows.filter(r=>r.convergio===0).length;
  const noAsent = rows.filter(r=>r.asent_ok===0).length;
  console.log(`${nombre}: ${dt.toFixed(1)}s, ${rows.length} puntos, ${noConverg} sin converger recuperación, ${noAsent} sin asentar`);
  return dt;
}

const base = { seed:1, modo:'parada', from:0.6, to:1.4, steps:60, settle:300, measure:120,
  powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:20, ptcSharp:16 };

const t1 = medir('baseline (tOpt=25,ptcSharp=16,beta=0.94,pB=0.47) semilla=1', base);
const t2 = medir('baseline semilla=2', {...base, seed:2});
const esquina = { ...base, seed:1, beta:0.80, tOpt:28, ptcSharp:16, powerBase:0.30 };
const t3 = medir('esquina grilla (beta=0.80,tOpt=28,ptcSharp=16,pB=0.30) semilla=1', esquina);
const esquina2 = { ...base, seed:1, beta:0.98, tOpt:22, ptcSharp:8, powerBase:0.65 };
const t4 = medir('esquina2 (beta=0.98,tOpt=22,ptcSharp=8,pB=0.65) semilla=1', esquina2);

console.log('\npromedio baseline:', ((t1+t2)/2).toFixed(1), 's/barrido');
console.log('esquinas:', t3.toFixed(1), 's y', t4.toFixed(1), 's');
