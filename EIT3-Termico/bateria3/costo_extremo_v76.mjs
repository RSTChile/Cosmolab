import { correrBarridoV76 } from './correr_barrido_v76.mjs';
const PARAMS = { powerBase:0.47, beta:0.80, sigma:6.8, noise:0.0079, band:1.105, tOpt:28, ptcTc:18, ptcSharp:6.0, minTemp:-6, maxTemp:25 };
const t0 = Date.now();
const rows = correrBarridoV76({ seed:1, modo:'parada', from:0.60, to:1.40, steps:60, settle:300, measure:120, params:PARAMS });
const seg = (Date.now()-t0)/1000;
const noConvergenRec = rows.filter(r=>r.recuperaron_todos===0);
const noAsentaron = rows.filter(r=>r.asent_ok===0);
console.log(`combo extremo (beta=0.80,tOpt=28,ptcSharp=6.0): ${seg.toFixed(1)}s | rec_no_convergen=${noConvergenRec.length}/60 | asent_no_ok=${noAsentaron.length}/60`);
