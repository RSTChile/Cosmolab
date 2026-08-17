import { correrBarrido2 } from './correr_barrido2.mjs';
const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const NIVELES = [150,300,600,1200,2400];
const porNivel = {};
for (const settle of NIVELES) {
  porNivel[settle] = correrBarrido2({ seed:1, modo:'parada', from:0.25, to:1.95, steps:20, settle, measure:120, params:PARAMS });
}
console.log('k  x       foot@150  foot@300  foot@600  foot@1200 foot@2400');
for (let k=0;k<20;k++){
  const vals = NIVELES.map(s=>porNivel[s][k].huella.toFixed(4));
  console.log(String(k).padStart(2), porNivel[150][k].luminosidad.toFixed(3), vals.join('   '));
}
console.log('\nk  x       H@150   H@300   H@600   H@1200  H@2400');
for (let k=0;k<20;k++){
  const vals = NIVELES.map(s=>porNivel[s][k].entropia_abs_local.toFixed(4));
  console.log(String(k).padStart(2), porNivel[150][k].luminosidad.toFixed(3), vals.join('  '));
}
