// Valida MotorV76/correrBarridoV76 contra el script real (shim_v76.mjs), en
// 2 puntos por caso: uno cerca de luminosidad=0.93 (zona lenta: las margaritas
// negras se extinguen ahí, para ejercitar asent_ok=0/cerca del tope y
// rec_topes>0) y uno en zona rápida (lum=1.4).
import { buildSandbox } from './shim_v76.mjs';
import { correrBarridoV76 } from './correr_barrido_v76.mjs';

const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const BARRIDO = { desde:0.93, hasta:1.4, puntos:2, settle:40, measure:30, trazas:0 };

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
  asent_pasos: 'asent_pasos', asent_ok: 'asent_ok',
  pasos_recuperacion: 'pasos_recuperacion', rec_mediana: 'rec_mediana', rec_topes: 'rec_topes',
  convergio: 'recuperaron_todos',
  varianza_pl: 'varianza_pl', autocorr1_pl: 'autocorr1_pl',
  H_absLocal: 'entropia_abs_local', H_noiseLocal: 'entropia_piso_local',
  H_absGlobal: 'entropia_abs_global', H_noiseGlobal: 'entropia_piso_global',
  H_rel: 'entropia_rel', footprint: 'huella', Lambda: 'lambda', A_sys_env: 'acoplamiento',
  err_rate: 'tasa_error', powerLive: 'potencia_viva', plRange: 'rango_potencia_viva',
  plBand: 'banda_potencia_viva', noiseFloor: 'piso_ruido', distinct: 'valores_distintos',
  mult: 'multiplicidad', diag: 'diagnostico', ptcSat: 'saturacion_sensor',
};

async function caso(seed, modo) {
  console.log(`  corriendo shim seed=${seed} modo=${modo}...`);
  const t0 = Date.now();
  const api = buildSandbox();
  fijar(api, seed, modo);
  await api.runSweep();
  const filasShim = api.getSweepRows();
  console.log(`  shim listo en ${((Date.now()-t0)/1000).toFixed(1)}s`);

  const filasMotor = correrBarridoV76({
    seed, modo,
    from: BARRIDO.desde, to: BARRIDO.hasta, steps: BARRIDO.puntos,
    settle: BARRIDO.settle, measure: BARRIDO.measure, params: PARAMS,
  });

  let diffs = 0;
  for (let i = 0; i < filasShim.length; i++) {
    const rs = filasShim[i], rm = filasMotor[i];
    console.log(`  fila ${i} (x=${rs.x}): asent_pasos shim=${rs.asent_pasos} motor=${rm.asent_pasos} | asent_ok shim=${rs.asent_ok} motor=${rm.asent_ok} | rec_mediana shim=${rs.rec_mediana} motor=${rm.rec_mediana} | rec_topes shim=${rs.rec_topes} motor=${rm.rec_topes}`);
    if (!casi(rs.x, rm.luminosidad, 1e-12)) { console.log(`    x distinto`); diffs++; }
    // rec_reps es un array en el shim; comparamos contra rec_1..5 del motor
    for (let j = 0; j < 5; j++) {
      if (rs.rec_reps[j] !== rm[`rec_${j+1}`]) { console.log(`    rec_reps[${j}]: ${rs.rec_reps[j]} vs ${rm[`rec_${j+1}`]}`); diffs++; }
    }
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
  console.log('=== motor_v76 vs shim (runSweep real v7.6.1), zona lenta (0.93) + zona rápida (1.4) ===');
  const r1 = await caso(7, 'parada');
  const r2 = await caso(7, 'inicio');
  const r3 = await caso(17, 'parada');
  const ok = r1 && r2 && r3;
  console.log(ok ? '\nPASÓ: MotorV76/correrBarridoV76 reproduce runSweep() real en los 3 casos.' : '\nFALLÓ.');
  process.exit(ok ? 0 : 1);
}
main().catch(e => { console.error('ERROR', e); process.exit(2); });
