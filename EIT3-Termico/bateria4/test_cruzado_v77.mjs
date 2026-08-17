import { buildSandbox } from 'file:///tmp/daisy_calib/shim_v77.mjs';
import { correrBarridoV77 } from './motor_v77.mjs';

function correrShim({seed, modo, from, to, steps, settle, measure, powerBase, beta, sigma, noise, band, tOpt, ptcTc, ptcSharp}){
  const api = buildSandbox();
  api.document.getElementById('sweepReset').value=modo;
  api.document.getElementById('seedInput').value=String(seed);
  api.document.getElementById('sweepAxis').value='luminosity';
  api.document.getElementById('sweepFrom').value=String(from);
  api.document.getElementById('sweepTo').value=String(to);
  api.document.getElementById('sweepSteps').value=String(steps);
  api.document.getElementById('sweepSettle').value=String(settle);
  api.document.getElementById('sweepMeasure').value=String(measure);
  api.document.getElementById('sweepNoise').value='0';
  api.document.getElementById('sweepTraceN').value='0';
  api.els.powerBase.value=String(powerBase); api.els.beta.value=String(beta); api.els.sigma.value=String(sigma);
  api.els.noise.value=String(noise); api.els.band.value=String(band); api.els.tOpt.value=String(tOpt);
  api.els.ptcTc.value=String(ptcTc); api.els.ptcSharp.value=String(ptcSharp);
  api.els.minTemp.value='-6'; api.els.maxTemp.value='25';
  return api.runSweep().then(()=>api.getSweepRows());
}

const casos = [
  { nombre:'baseline centro', seed:3, modo:'parada', from:0.6, to:1.4, steps:6, settle:60, measure:40, powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:20, ptcSharp:16 },
  { nombre:'borde frio (extincion)', seed:5, modo:'parada', from:0.6, to:0.68, steps:4, settle:60, measure:40, powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:20, ptcSharp:16 },
  { nombre:'modo inicio', seed:7, modo:'inicio', from:0.6, to:1.4, steps:5, settle:60, measure:40, powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:22, ptcTc:20, ptcSharp:16 },
  { nombre:'borde caliente (extincion)', seed:11, modo:'parada', from:1.32, to:1.4, steps:4, settle:60, measure:40, powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:20, ptcSharp:16 },
];

const CAMPOS = ['x','pasos_recuperacion','convergio','rec_mediana','rec_topes','asent_pasos','asent_ok',
  'tasa_recuperacion','varianza_pl','autocorr1_pl','H_absLocal','H_noiseLocal','H_absGlobal','H_noiseGlobal',
  'H_rel','footprint','Lambda','A_sys_env','err_rate','powerLive','plRange','plBand','noiseFloor','distinct','mult','diag','ptcSat'];

(async () => {
  for (const caso of casos) {
    const { nombre, ...params } = caso;
    const filasShim = await correrShim(params);
    const filasMotor = correrBarridoV77(params);
    let diffs = 0;
    for (let i=0;i<filasShim.length;i++){
      for (const campo of CAMPOS){
        const a = filasShim[i][campo], b = filasMotor[i][campo];
        if (Array.isArray(a)) {
          if (JSON.stringify(a) !== JSON.stringify(b)) { diffs++; console.log(`  DIFF fila${i}.${campo}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`); }
        } else if (a !== b) { diffs++; console.log(`  DIFF fila${i}.${campo}: ${a} vs ${b}`); }
      }
    }
    console.log(`${nombre}: ${filasShim.length} filas x ${CAMPOS.length} campos, diffs=${diffs}`, diffs===0?'IDENTICO':'DIFIERE');
  }
})();
