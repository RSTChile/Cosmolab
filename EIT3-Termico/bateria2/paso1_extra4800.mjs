import { correrBarrido2 } from './correr_barrido2.mjs';
const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const t0=Date.now();
const rows = correrBarrido2({ seed:1, modo:'parada', from:0.25, to:1.95, steps:20, settle:4800, measure:120, params:PARAMS });
console.log('tiempo:', ((Date.now()-t0)/1000).toFixed(1), 's');
console.log('k  x       huella   H_absLocal');
rows.forEach((r,k)=>console.log(String(k).padStart(2), r.luminosidad.toFixed(3), r.huella.toFixed(4).padStart(8), r.entropia_abs_local.toFixed(4)));
