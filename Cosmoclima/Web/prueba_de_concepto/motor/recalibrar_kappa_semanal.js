// Ronda 12 (11-ago-2026) -- helper reutilizable: toma un
// regimen_<sufijo>_percentiles.json (salida de calcular_percentiles_y_regimen.js)
// y emite las 4 líneas `let KAPPA_*_POR_SEMANA = [...]` listas para pegar en
// el HTML y en generar_motor_node.py (SENTENCIAS), con la misma receta usada
// en las rondas anteriores: mediana (p50) para κ_LF/κ_Δ/κ_V, p90 de e_R con
// piso 0.0005 para κ_O (evita división por casi-cero cuando una semana no
// tuvo caídas de acoplamiento en casi ningún año).
// Uso: node recalibrar_kappa_semanal.js regimen_<sufijo>_percentiles.json
'use strict';
const fs = require('fs');
const ruta = process.argv[2];
if (!ruta) { console.error('Uso: node recalibrar_kappa_semanal.js regimen_<sufijo>_percentiles.json'); process.exit(1); }
const j = JSON.parse(fs.readFileSync(ruta, 'utf-8'));
const ps = j.percentilesPorSemana;

const kLF = ps.map(s => +s.LF.p50.toFixed(4));
const kDelta = ps.map(s => +s.deltaStruct.p50.toFixed(4));
const kV = ps.map(s => +s.A_sys_env.p50.toFixed(4));
const kO = ps.map(s => +Math.max(s.err.p90, 0.0005).toFixed(4));

console.log(`let KAPPA_LF_POR_SEMANA = [${kLF.join(',')}];`);
console.log(`let KAPPA_DELTA_POR_SEMANA = [${kDelta.join(',')}];`);
console.log(`let KAPPA_V_POR_SEMANA = [${kV.join(',')}];`);
console.log(`let KAPPA_O_POR_SEMANA = [${kO.join(',')}];`);
