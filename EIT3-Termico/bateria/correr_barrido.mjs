// Replica exacta de la lógica de runSweep() (líneas ~466-559 del HTML original)
// para el eje 'luminosidad', usando Motor (motor.mjs) en vez del DOM.
// Reset COMPLETO una vez por barrido (defecto de reproducibilidad corregido,
// ver defectos_encontrados.md) + rng consumido de forma continua a través de
// ambas pasadas y todos los puntos del eje (una sola semilla por barrido).
import { Motor } from './motor.mjs';

export function correrBarrido({ seed, from, to, steps, settle, measure, params, bins = 24, margin = 0.05 }) {
  const m = new Motor();
  m.setSeed(seed);
  m.resetSimulation();
  m.setParams(params);

  const calSteps = Math.min(80, Math.max(20, Math.round(settle / 2) || 40));

  // ── pasada de calibración global (banda global de powerLive) ──
  let gLo = Infinity, gHi = -Infinity;
  for (let k = 0; k < steps; k++) {
    const v = from + (to - from) * k / (steps - 1);
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

  // ── pasada de medición (calibración local por punto) ──
  const rows = [];
  for (let k = 0; k < steps; k++) {
    const v = from + (to - from) * k / (steps - 1);
    m.resetField(); m.aBuf = []; m.noiseEchoBuf = []; m.state._A_prev = 0; m._Awin = [];
    m.state.luminosity = v;
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

    rows.push({
      semilla: seed,
      potencia_base: params.powerBase, beta: params.beta, sigma: params.sigma, ruido: params.noise,
      banda: params.band, t_optima: params.tOpt, tc_ptc: params.ptcTc, exponente_ptc: params.ptcSharp,
      dia_noche: 0, luminosidad: v,
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
