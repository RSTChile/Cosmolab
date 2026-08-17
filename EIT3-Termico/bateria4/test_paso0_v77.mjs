import { buildSandbox } from 'file:///tmp/daisy_calib/shim_v77.mjs';
import crypto from 'node:crypto';

function correr({seed, modo}){
  const api = buildSandbox();
  api.document.getElementById('sweepReset').value=modo;
  api.document.getElementById('seedInput').value=String(seed);
  api.document.getElementById('sweepAxis').value='luminosity';
  api.document.getElementById('sweepFrom').value='0.6';
  api.document.getElementById('sweepTo').value='1.4';
  api.document.getElementById('sweepSteps').value='8';
  api.document.getElementById('sweepSettle').value='40';
  api.document.getElementById('sweepMeasure').value='30';
  api.document.getElementById('sweepNoise').value='0';
  api.document.getElementById('sweepTraceN').value='0';
  api.els.powerBase.value='0.47'; api.els.beta.value='0.94'; api.els.sigma.value='6.8';
  api.els.noise.value='0.0079'; api.els.band.value='1.105'; api.els.tOpt.value='25';
  api.els.ptcTc.value='20'; api.els.ptcSharp.value='16';
  api.els.minTemp.value='-6'; api.els.maxTemp.value='25';
  return api.runSweep().then(()=>{
    const rows = api.getSweepRows();
    const txt = JSON.stringify(rows);
    return crypto.createHash('sha256').update(txt).digest('hex');
  });
}

(async () => {
  const x1 = await correr({seed:7, modo:'parada'});
  const x2 = await correr({seed:7, modo:'parada'});
  const y = await correr({seed:99, modo:'parada'});
  const z = await correr({seed:7, modo:'inicio'});
  console.log('semilla=7/parada  #1:', x1);
  console.log('semilla=7/parada  #2:', x2, x1===x2 ? 'IDENTICO ✅' : 'DIFIERE ❌');
  console.log('semilla=99/parada   :', y, y!==x1 ? 'distinto ✅' : 'IGUAL ❌ (no debería)');
  console.log('semilla=7/inicio    :', z, z!==x1 ? 'distinto ✅' : 'IGUAL ❌ (no debería)');
})();
