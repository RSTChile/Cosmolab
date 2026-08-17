import { correrBarrido2 } from './correr_barrido2.mjs';
const PARAMS = { powerBase:0.47, beta:0.94, sigma:6.8, noise:0.0079, band:1.105, tOpt:25, ptcTc:18, ptcSharp:4.1, minTemp:-6, maxTemp:25 };
const NIVELES = [150,300,600,1200,2400,4800];
const porNivel = {};
for (const settle of NIVELES) {
  porNivel[settle] = correrBarrido2({ seed:1, modo:'parada', from:0.25, to:1.95, steps:20, settle, measure:120, params:PARAMS });
}
const FRONTERA = new Set([6,7,8,9]); // x en [0.787, 1.055], la zona del colapso

function diffZona(prev, cur, keys) {
  let sumAbsF=0,maxAbsF=0,sumRelF=0,maxRelF=0,sumAbsH=0,maxAbsH=0,sumRelH=0,maxRelH=0,n=0;
  for (const k of keys) {
    const fa=prev[k].huella, fb=cur[k].huella, ha=prev[k].entropia_abs_local, hb=cur[k].entropia_abs_local;
    const dF=Math.abs(fb-fa), dH=Math.abs(hb-ha);
    sumAbsF+=dF; maxAbsF=Math.max(maxAbsF,dF);
    sumAbsH+=dH; maxAbsH=Math.max(maxAbsH,dH);
    const rF=dF/Math.max(1e-9,Math.abs(fa)), rH=dH/Math.max(1e-9,Math.abs(ha));
    sumRelF+=rF; maxRelF=Math.max(maxRelF,rF);
    sumRelH+=rH; maxRelH=Math.max(maxRelH,rH);
    n++;
  }
  return {huella_abs_media:sumAbsF/n, huella_abs_max:maxAbsF, huella_rel_max_pct:maxRelF*100,
          H_abs_media:sumAbsH/n, H_abs_max:maxAbsH, H_rel_max_pct:maxRelH*100};
}

const todos = [...Array(20).keys()];
const resto = todos.filter(k=>!FRONTERA.has(k));

console.log('=== RESTO DEL EJE (16 puntos, excluye k=6..9, zona del colapso) ===');
for (let i=1;i<NIVELES.length-1;i++){ // no incluir 4800 acá, solo hasta 2400 (el resto ya converge antes)
  const prev=NIVELES[i-1], cur=NIVELES[i];
  const d = diffZona(porNivel[prev], porNivel[cur], resto);
  console.log(`${prev}->${cur}: huella_abs_media=${d.huella_abs_media.toFixed(4)} huella_abs_max=${d.huella_abs_max.toFixed(4)} huella_rel_max%=${d.huella_rel_max_pct.toFixed(2)} | H_abs_media=${d.H_abs_media.toFixed(4)} H_abs_max=${d.H_abs_max.toFixed(4)} H_rel_max%=${d.H_rel_max_pct.toFixed(2)}`);
}

console.log('\n=== ZONA FRONTERA (k=6..9, x=0.787..1.055, el colapso de la huella) ===');
for (let i=1;i<NIVELES.length;i++){
  const prev=NIVELES[i-1], cur=NIVELES[i];
  const d = diffZona(porNivel[prev], porNivel[cur], [...FRONTERA]);
  console.log(`${prev}->${cur}: huella_abs_media=${d.huella_abs_media.toFixed(4)} huella_abs_max=${d.huella_abs_max.toFixed(4)} huella_rel_max%=${d.huella_rel_max_pct.toFixed(2)} | H_abs_media=${d.H_abs_media.toFixed(4)} H_abs_max=${d.H_abs_max.toFixed(4)} H_rel_max%=${d.H_rel_max_pct.toFixed(2)}`);
}
