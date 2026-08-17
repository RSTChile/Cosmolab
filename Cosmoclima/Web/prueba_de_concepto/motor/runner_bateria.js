// Runner de batería de experimentos (Fase A.4 del plan de granularidad,
// 08-ago-2026). Lee un archivo de config JSON (array de
// {id, resolucionLluvia, overrides, kappa, semilla}), corre cada
// configuración en su PROPIO proceso (child_process.fork -- aislamiento
// real de sistema operativo, no worker_threads), con concurrencia acotada
// a cores-1, y al final junta todo en resultados/bateria_<fecha>_resumen.csv.
//
// Uso: node runner_bateria.js experimentos/mi_bateria.json
'use strict';
const fs = require('fs');
const path = require('path');
const os = require('os');
const { fork } = require('child_process');

const configPath = process.argv[2];
if (!configPath) {
  console.error('Uso: node runner_bateria.js <archivo_config.json>');
  process.exit(1);
}
const configs = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
if (!Array.isArray(configs) || configs.length === 0) {
  console.error('El archivo de config debe ser un array no vacío de configuraciones.');
  process.exit(1);
}

const CONCURRENCIA = Math.max(1, os.cpus().length - 1);
console.log(`${configs.length} configuraciones, concurrencia ${CONCURRENCIA} (${os.cpus().length} núcleos detectados).`);
console.log('Cada corrida completa tarda minutos reales (ver benchmark_motor.js) -- una batería grande puede tardar horas. No es un número optimista, es lo medido en esta máquina.');

const pendientes = [...configs];
const enCurso = new Map(); // id -> proceso
const resumen = [];
const t0 = Date.now();

function lanzarSiguiente() {
  if (pendientes.length === 0) return;
  const config = pendientes.shift();
  const hijo = fork(path.join(__dirname, 'experimentos', 'child_worker.js'), [JSON.stringify(config)], {
    stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
  });
  const tInicio = Date.now();
  enCurso.set(config.id, hijo);
  hijo.on('message', (msg) => {
    const ZONAS = ['JARDIN_FERTIL', 'CIERRE', 'SELVA_HOSTIL', 'COLAPSO'];
    const total = msg.ticksTotales;
    const fila = { id: msg.id };
    ZONAS.forEach((z) => { fila[`${z}_pct`] = (msg.conteoGlobal[z] / total * 100).toFixed(2); });
    resumen.push(fila);
  });
  hijo.on('exit', (code) => {
    const minutos = ((Date.now() - tInicio) / 60000).toFixed(1);
    console.log(`[${config.id}] terminó (código ${code}) en ${minutos} min.`);
    enCurso.delete(config.id);
    if (code !== 0) {
      console.error(`[${config.id}] ATENCIÓN: salió con error -- revisar esta config antes de confiar en su resultado.`);
    }
    lanzarSiguiente();
    if (pendientes.length === 0 && enCurso.size === 0) finalizar();
  });
}

function finalizar() {
  const resultadosDir = path.join(__dirname, 'resultados');
  fs.mkdirSync(resultadosDir, { recursive: true });
  const fecha = new Date().toISOString().slice(0, 10);
  const archivoResumen = path.join(resultadosDir, `bateria_${fecha}_resumen.csv`);
  if (resumen.length > 0) {
    const header = Object.keys(resumen[0]);
    const csv = [header.join(','), ...resumen.map((r) => header.map((h) => r[h]).join(','))].join('\n');
    fs.writeFileSync(archivoResumen, csv);
  }
  const minutosTotal = ((Date.now() - t0) / 60000).toFixed(1);
  console.log(`\nBatería completa en ${minutosTotal} min. Resumen: ${archivoResumen}`);
  console.log(`Resultados individuales (JSON + CSV por año) en: ${resultadosDir}`);
}

for (let i = 0; i < CONCURRENCIA; i++) lanzarSiguiente();
