import { correrBarridoV75 } from './correr_barrido_v75.mjs';
const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const t0 = Date.now();
const rows = correrBarridoV75({ seed:3, modo:'parada', from:0.25, to:1.95, steps:60, settle:300, measure:120, params:PARAMS });
const seg = (Date.now()-t0)/1000;
const noConvergen = rows.filter(r=>r.recuperaron_todos===0);
console.log(`semilla=3: ${seg.toFixed(1)}s | no convergen: ${noConvergen.length}/60`);
