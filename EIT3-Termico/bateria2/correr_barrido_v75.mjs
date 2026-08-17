// Replica exacta de runSweep() de v7.5.html (líneas ~553-689), usando
// MotorV75 (motor_v75.mjs). Idéntica a correr_barrido2.mjs (v7.4.1) salvo por
// lo nuevo: warm-up de min(200,SETTLE) + medirRecuperacion(GOLPE_TF,TOPE_REC)
// ANTES del settle/measure normal, y varianzaYAutocorr(plS) al medir.
// GOLPE_TF=0.03 y TOPE_REC=20000 son constantes fijas del HTML, no dependen
// del settle elegido.
import { MotorV75 } from './motor_v75.mjs';

const GOLPE_TF = 0.03, TOPE_REC = 20000;

export function correrBarridoV75({ seed, modo, from, to, steps, settle, measure, params, bins = 24, margin = 0.05 }) {
  const m = new MotorV75();
  m.setSeed(seed);
  m.setParams(params);
  if (modo !== 'ninguno') m.reiniciarSilencioso();

  const calSteps = Math.min(80, Math.max(20, Math.round(settle / 2) || 40));

  // ── pasada de calibración global (sin cambios respecto a v7.4.1) ──
  let gLo = Infinity, gHi = -Infinity;
  for (let k = 0; k < steps; k++) {
    const v = from + (to - from) * k / (steps - 1);
    if (modo === 'parada') m.reiniciarSilencioso();
    m.resetField(); m.aBuf = []; m.noiseEchoBuf = []; m.state._A_prev = 0; m._Awin = [];
    m.state.luminosity = v;
    for (let s = 0; s < calSteps; s++) {
      m.paso();
      if (s > calSteps * 0.5) {
        if (m.state.powerLive < gLo) gLo = m.state.powerLive;
        if (m.state.powerLive > gHi) gHi = m.state.powerLive;
      }
    }
  }
  if (!isFinite(gLo) || !isFinite(gHi)) { gLo = 0; gHi = 1; }
  const gMargin = ((gHi - gLo) * margin) || 1e-3;
  const gLoCal = gLo - gMargin, gHiCal = gHi + gMargin;

  // ── pasada de medición (con medición de recuperación, v7.5) ──
  const rows = [];
  for (let k = 0; k < steps; k++) {
    const v = from + (to - from) * k / (steps - 1);
    if (modo === 'parada') m.reiniciarSilencioso();
    m.resetField(); m.aBuf = []; m.noiseEchoBuf = []; m.state._A_prev = 0; m._Awin = [];
    m.state.luminosity = v;

    for (let s = 0; s < Math.min(200, settle); s++) m.paso();
    const asent = m.medirRecuperacion(GOLPE_TF, TOPE_REC);
    for (let s = 0; s < settle; s++) { m.paso(); m.errRatePush(m.state.A_sys_env); }

    const plS = [], neS = [];
    let footSum = 0, lamSum = 0, aSum = 0, errSum = 0, c = 0;
    for (let s = 0; s < measure; s++) {
      m.paso(); m.errRatePush(m.state.A_sys_env);
      const eR = m.errRate();
      const lam = (m.state.deltaStruct * m.state.mult) / Math.max(eR, 1e-6) * m.state.A_sys_env;
      plS.push(m.state.powerLive); neS.push(m.passiveNoiseSample());
      footSum += m.state.bioticFootprint; lamSum += lam; aSum += m.state.A_sys_env; errSum += eR; c++;
    }

    const la = m.entropyLocalAbs(plS, neS, bins, margin);
    const H_noiseLocal = m.entropyAtWidth(neS, bins, la.hi - la.lo);
    const H_absGlobal = m.shannonEntropy(plS, bins, gLoCal, gHiCal);
    const H_noiseGlobal = m.entropyAtWidth(neS, bins, gHiCal - gLoCal);
    const H_rel = m.entropyRel(plS, bins);
    let pLo = Infinity, pHi = -Infinity, sum = 0;
    for (const x of plS) { if (x < pLo) pLo = x; if (x > pHi) pHi = x; sum += x; }
    const plRange = pHi - pLo, plMean = sum / plS.length;
    const distinct = new Set(plS.map(x => x.toFixed(5))).size;
    const diag = (plRange <= Math.max(la.floor, 1e-9)) ? 'banda<=ruido' : 'banda>ruido';
    const ptcSat = (m.state.ptcOut >= 1.2 - 1e-9 || m.state.ptcOut <= 0.05 + 1e-9) ? 1 : 0;
    const vac = m.varianzaYAutocorr(plS);

    rows.push({
      semilla: seed, modo,
      potencia_base: params.powerBase, beta: params.beta, sigma: params.sigma, ruido: params.noise,
      banda: params.band, t_optima: params.tOpt, tc_ptc: params.ptcTc, exponente_ptc: params.ptcSharp,
      dia_noche: 0, luminosidad: v,
      pasos_recuperacion: asent.pasos, recuperaron_todos: asent.convergio,
      tasa_recuperacion: asent.pasos > 0 ? 1 / asent.pasos : 0,
      varianza_pl: vac.varianza, autocorr1_pl: vac.autocorr1,
      entropia_abs_local: la.H, entropia_piso_local: H_noiseLocal,
      entropia_abs_global: H_absGlobal, entropia_piso_global: H_noiseGlobal, entropia_rel: H_rel,
      huella: footSum / c, lambda: lamSum / c, acoplamiento: aSum / c, tasa_error: errSum / c,
      potencia_viva: plMean, rango_potencia_viva: plRange, banda_potencia_viva: la.band,
      piso_ruido: la.floor, valores_distintos: distinct, multiplicidad: m.state.mult,
      diagnostico: diag, saturacion_sensor: ptcSat,
    });
  }
  return rows;
}
