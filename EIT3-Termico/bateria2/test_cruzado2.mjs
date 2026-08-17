// Tarea 5: valida motor2.mjs (correrBarrido2) contra el script REAL corrido en
// el shim (runSweep() real), comparando fila por fila en precisión completa
// (no a través del CSV con .toFixed truncado, sino leyendo sweepRows crudo del
// sandbox vía getSweepRows()).
import { buildSandbox } from './shim_html2.mjs';
import { correrBarrido2 } from './correr_barrido2.mjs';

const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const BARRIDO = { desde:0.6, hasta:1.4, puntos:8, settle:40, measure:30, trazas:0 };

function fijar(api, seed, modo) {
  api.els.powerBase.value = String(PARAMS.powerBase);
  api.els.beta.value = String(PARAMS.beta);
  api.els.sigma.value = String(PARAMS.sigma);
  api.els.noise.value = String(PARAMS.noise);
  api.els.band.value = String(PARAMS.band);
  api.els.tOpt.value = String(PARAMS.tOpt);
  api.els.ptcTc.value = String(PARAMS.ptcTc);
  api.els.ptcSharp.value = String(PARAMS.ptcSharp);
  api.els.minTemp.value = '-6'; api.els.maxTemp.value = '25'; api.els.dayNightToggle.checked = false;
  api.getEl('sweepAxis').value = 'luminosity';
  api.getEl('sweepFrom').value = String(BARRIDO.desde);
  api.getEl('sweepTo').value = String(BARRIDO.hasta);
  api.getEl('sweepSteps').value = String(BARRIDO.puntos);
  api.getEl('sweepSettle').value = String(BARRIDO.settle);
  api.getEl('sweepMeasure').value = String(BARRIDO.measure);
  api.getEl('sweepNoise').value = '0';
  api.getEl('sweepTraceN').value = '0';
  api.getEl('sweepReset').value = modo;
  api.getEl('seedInput').value = String(seed);
}

function casi(a, b, eps = 1e-9) { return Math.abs(a - b) <= eps * Math.max(1, Math.abs(a), Math.abs(b)); }

const MAPA = {
  H_absLocal: 'entropia_abs_local', H_noiseLocal: 'entropia_piso_local',
  H_absGlobal: 'entropia_abs_global', H_noiseGlobal: 'entropia_piso_global',
  H_rel: 'entropia_rel', footprint: 'huella', Lambda: 'lambda', A_sys_env: 'acoplamiento',
  err_rate: 'tasa_error', powerLive: 'potencia_viva', plRange: 'rango_potencia_viva',
  plBand: 'banda_potencia_viva', noiseFloor: 'piso_ruido', distinct: 'valores_distintos',
  mult: 'multiplicidad', diag: 'diagnostico', ptcSat: 'saturacion_sensor',
};

async function caso(seed, modo) {
  const api = buildSandbox();
  fijar(api, seed, modo);
  await api.runSweep();
  const filasShim = api.getSweepRows();

  const filasMotor = correrBarrido2({
    seed, modo,
    from: BARRIDO.desde, to: BARRIDO.hasta, steps: BARRIDO.puntos,
    settle: BARRIDO.settle, measure: BARRIDO.measure, params: PARAMS,
  });

  if (filasShim.length !== filasMotor.length) {
    console.log(`  seed=${seed} modo=${modo}: FALLÓ — distinto número de filas (${filasShim.length} vs ${filasMotor.length})`);
    return false;
  }
  let diffs = 0;
  for (let i = 0; i < filasShim.length; i++) {
    const rs = filasShim[i], rm = filasMotor[i];
    if (!casi(rs.x, rm.luminosidad, 1e-12)) { console.log(`  fila ${i}: x ${rs.x} vs ${rm.luminosidad}`); diffs++; }
    for (const [kShim, kMotor] of Object.entries(MAPA)) {
      const a = rs[kShim], b = rm[kMotor];
      const ok = (typeof a === 'string') ? a === b : casi(a, b, 1e-9);
      if (!ok) { console.log(`  fila ${i} campo ${kShim}/${kMotor}: ${a} vs ${b}`); diffs++; }
    }
  }
  console.log(`  seed=${seed} modo=${modo}: ${diffs === 0 ? 'IDÉNTICO' : `${diffs} DIFERENCIAS`}`);
  return diffs === 0;
}

async function main() {
  console.log('=== Tarea 5: motor2.mjs vs shim (script real v7.4.1), runSweep completo ===');
  const r1 = await caso(1, 'parada');
  const r2 = await caso(1, 'inicio');
  const r3 = await caso(17, 'parada');
  const ok = r1 && r2 && r3;
  console.log(ok ? '\nTarea 5 PASÓ: motor2.mjs reproduce runSweep() real en los 3 casos.' : '\nTarea 5 FALLÓ.');
  process.exit(ok ? 0 : 1);
}
main().catch(e => { console.error('ERROR', e); process.exit(2); });
