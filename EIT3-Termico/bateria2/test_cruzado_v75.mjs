// Valida MotorV75/correrBarridoV75 contra el script real (shim_v75.mjs),
// usando SOLO 2 puntos del eje: uno en zona lenta (lum=0.6, cerca de donde la
// referencia de Alexis dice que NO converge) y uno en zona rápida (lum=1.4,
// 70 pasos según la referencia). Con steps=2, runSweep evalúa exactamente
// from y to, así que un barrido de 2 puntos alcanza para ejercitar ambas ramas
// (convergio=0 con tope=20000 Y convergio=1 rápido) sin pagar el costo de
// correr los 24 puntos completos por el shim (lento, ~150-250 pasos/s).
import { buildSandbox } from './shim_v75.mjs';
import { correrBarridoV75 } from './correr_barrido_v75.mjs';

const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const BARRIDO = { desde:0.6, hasta:1.4, puntos:2, settle:40, measure:30, trazas:0 };

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
  pasos_recuperacion: 'pasos_recuperacion', convergio: 'recuperaron_todos',
  varianza_pl: 'varianza_pl', autocorr1_pl: 'autocorr1_pl',
  H_absLocal: 'entropia_abs_local', H_noiseLocal: 'entropia_piso_local',
  H_absGlobal: 'entropia_abs_global', H_noiseGlobal: 'entropia_piso_global',
  H_rel: 'entropia_rel', footprint: 'huella', Lambda: 'lambda', A_sys_env: 'acoplamiento',
  err_rate: 'tasa_error', powerLive: 'potencia_viva', plRange: 'rango_potencia_viva',
  plBand: 'banda_potencia_viva', noiseFloor: 'piso_ruido', distinct: 'valores_distintos',
  mult: 'multiplicidad', diag: 'diagnostico', ptcSat: 'saturacion_sensor',
};

async function caso(seed, modo) {
  console.log(`  corriendo shim seed=${seed} modo=${modo}... (puede tardar por la zona lenta)`);
  const t0 = Date.now();
  const api = buildSandbox();
  fijar(api, seed, modo);
  await api.runSweep();
  const filasShim = api.getSweepRows();
  console.log(`  shim listo en ${((Date.now()-t0)/1000).toFixed(1)}s`);

  const filasMotor = correrBarridoV75({
    seed, modo,
    from: BARRIDO.desde, to: BARRIDO.hasta, steps: BARRIDO.puntos,
    settle: BARRIDO.settle, measure: BARRIDO.measure, params: PARAMS,
  });

  let diffs = 0;
  for (let i = 0; i < filasShim.length; i++) {
    const rs = filasShim[i], rm = filasMotor[i];
    console.log(`  fila ${i} (x=${rs.x}): pasos_recuperacion shim=${rs.pasos_recuperacion} motor=${rm.pasos_recuperacion} | convergio shim=${rs.convergio} motor=${rm.recuperaron_todos}`);
    if (!casi(rs.x, rm.luminosidad, 1e-12)) { console.log(`    x distinto`); diffs++; }
    for (const [kShim, kMotor] of Object.entries(MAPA)) {
      const a = rs[kShim], b = rm[kMotor];
      const ok = (typeof a === 'string') ? a === b : casi(a, b, 1e-9);
      if (!ok) { console.log(`    campo ${kShim}/${kMotor}: ${a} vs ${b}`); diffs++; }
    }
  }
  console.log(`  seed=${seed} modo=${modo}: ${diffs === 0 ? 'IDÉNTICO' : `${diffs} DIFERENCIAS`}`);
  return diffs === 0;
}

async function main() {
  console.log('=== motor_v75 vs shim (runSweep real v7.5), 2 puntos: zona lenta + zona rápida ===');
  const ok = await caso(7, 'parada');
  console.log(ok ? '\nPASÓ: MotorV75/correrBarridoV75 reproduce runSweep() real.' : '\nFALLÓ.');
  process.exit(ok ? 0 : 1);
}
main().catch(e => { console.error('ERROR', e); process.exit(2); });
