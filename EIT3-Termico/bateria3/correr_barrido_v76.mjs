// Replica exacta de runSweep() de ET3-Termico_v7.6.1.html (líneas ~748-911),
// usando MotorV76. Orden EXACTO de sembrarFase (la parte más fácil de arruinar
// por transcripción, según el propio encargo):
//   pasada de calibración, por punto: sembrarFase(eje,k,'calibracion')
//   pasada de medición, por punto:
//     (reset si modo==='parada')
//     sembrarFase(eje,k,'preasentamiento') -> asentarHastaEquilibrio
//     sembrarFase(eje,k,'recuperacion')    -> medirRecuperacion
//     sembrarFase(eje,k,'asentamiento')    -> SETTLE pasos
//     sembrarFase(eje,k,'medicion')        -> MEASURE pasos
import { MotorV76 } from './motor_v76.mjs';

// v7.6.1-bateria3 — TOPE_EQ/TOPE_REC bajados de 20.000 a 6.000/3.000. Decisión
// DELIBERADA de esta batería (no un cambio al instrumento real / HTML), con
// respaldo de datos: 0 reclasificación medida en 480 puntos de muestra (8
// combinaciones de la grilla de B', incluida la esquina más extrema), con
// márgenes de 50% (TOPE_EQ) y 43% (TOPE_REC) sobre el máximo genuino
// observado. Ver justificación completa y la distribución medida en
// topes_investigacion.md y defectos_encontrados3.md.
const GOLPE_TF = 0.03, TOPE_REC = 3000;
const TOPE_EQ = 6000, TOL_EQ = GOLPE_TF * 0.2 / 10;

export function correrBarridoV76({ seed, modo, from, to, steps, settle, measure, params, bins = 24, margin = 0.05, eje = 'luminosity' }) {
  const m = new MotorV76();
  m.setSeed(seed);
  m.setParams(params);
  if (modo !== 'ninguno') m.reiniciarSilencioso();

  const calSteps = Math.min(80, Math.max(20, Math.round(settle / 2) || 40));

  // ── pasada de calibración global ──
  let gLo = Infinity, gHi = -Infinity;
  for (let k = 0; k < steps; k++) {
    const v = from + (to - from) * k / (steps - 1);
    if (modo === 'parada') m.reiniciarSilencioso();
    m.resetField(); m.aBuf = []; m.noiseEchoBuf = []; m.state._A_prev = 0; m._Awin = [];
    m.state.luminosity = v;
    m.sembrarFase(eje, k, 'calibracion');
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

  // ── pasada de medición ──
  const rows = [];
  for (let k = 0; k < steps; k++) {
    const v = from + (to - from) * k / (steps - 1);
    if (modo === 'parada') m.reiniciarSilencioso();
    m.resetField(); m.aBuf = []; m.noiseEchoBuf = []; m.state._A_prev = 0; m._Awin = [];
    m.state.luminosity = v;

    m.sembrarFase(eje, k, 'preasentamiento');
    const eq = m.asentarHastaEquilibrio(TOPE_EQ, TOL_EQ);
    m.sembrarFase(eje, k, 'recuperacion');
    const asent = m.medirRecuperacion(GOLPE_TF, TOPE_REC);
    m.sembrarFase(eje, k, 'asentamiento');
    for (let s = 0; s < settle; s++) { m.paso(); m.errRatePush(m.state.A_sys_env); }
    m.sembrarFase(eje, k, 'medicion');

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
      asent_pasos: eq.pasos, asent_ok: eq.asentado,
      pasos_recuperacion: asent.pasos, rec_mediana: asent.mediana, rec_topes: asent.topes,
      rec_1: asent.reps[0], rec_2: asent.reps[1], rec_3: asent.reps[2], rec_4: asent.reps[3], rec_5: asent.reps[4],
      recuperaron_todos: asent.convergio,
      tasa_recuperacion: asent.mediana > 0 ? 1 / asent.mediana : 0,
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
