// Prueba del mecanismo de recorte silencioso (defecto documentado en v7.6.1):
// el <input id="luminosity" min="0.6" max="1.4"> SIEMPRE tuvo ese rango, en
// TODAS las versiones. Si un barrido pide un valor fuera de [0.6,1.4] via
// setAxisValue('luminosity', v) -> els.luminosity.value = String(v), un
// elemento DOM real de un navegador de verdad recorta el valor al límite más
// cercano cuando se lee de vuelta. Nuestros shims/motores de Node SIEMPRE
// usaron objetos planos (sin ese recorte), así que nunca lo reprodujeron.
//
// Corremos el <script> REAL de v7.3 (versión modificada en la batería 1: ya
// tiene semilla y el bug de arrastre SIGUE ahí, tal como en el HTML real) en
// DOS sandboxes idénticas salvo un detalle: en una, els.luminosity CLAMPEA su
// valor como lo haría un <input type="range"> real; en la otra, es un objeto
// plano (como en TODO nuestro trabajo anterior). Mismo seed, mismo barrido de
// 60 puntos, 0.25->1.95 (el rango de la batería 1 y la referencia -0.756).
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = path.join(__dirname, '..', '..', 'Old', 'EIT3_Termico_kappaH_v7.3.html');

const EPILOGUE = `
;globalThis.__api = {
  getState: () => state,
  setSeed, resetSimulation,
  runSweep, els, document,
  getSweepRows: () => sweepRows,
};
`;

function extractScript() {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const m = html.match(/<script>([\s\S]*)<\/script>/);
  if (!m) throw new Error('no se encontró <script> en el HTML');
  return m[1] + EPILOGUE;
}

function makeFakeElement(id, { clampLuminosity } = {}) {
  const ctx2d = {
    clearRect(){}, fillRect(){}, beginPath(){}, arc(){}, stroke(){}, fill(){},
    moveTo(){}, lineTo(){}, fillText(){}, setLineDash(){},
    fillStyle:'', strokeStyle:'', lineWidth:1, font:'',
  };
  const base = {
    id, textContent: '', checked: false, className: '', style: {},
    disabled: false, hidden: false, dataset: {}, width: 520, height: 520,
    _listeners: {},
    addEventListener(ev, fn){ (this._listeners[ev] ||= []).push(fn); },
    removeEventListener(){},
    setAttribute(){}, getAttribute(){ return null; },
    appendChild(){}, prepend(){}, remove(){},
    closest(){ return null; },
    getContext(){ return ctx2d; },
    getBoundingClientRect(){ return { width: 400, height: 220 }; },
    querySelectorAll(){ return []; },
    click(){}, href:'', download:'',
  };
  if (id === 'luminosity' && clampLuminosity) {
    // Replica el comportamiento real de <input type="range" min="0.6" max="1.4">:
    // el navegador recorta CUALQUIER valor asignado al rango declarado.
    let raw = '0.94';
    Object.defineProperty(base, 'value', {
      get(){ return raw; },
      set(v){ raw = String(Math.max(0.6, Math.min(1.4, Number(v)))); },
    });
  } else {
    base.value = '0';
  }
  return base;
}

function buildSandbox({ clampLuminosity }) {
  const elements = new Map();
  const getEl = (id) => {
    if (!elements.has(id)) elements.set(id, makeFakeElement(id, { clampLuminosity }));
    return elements.get(id);
  };
  const documentStub = {
    getElementById: getEl,
    querySelectorAll(){ return []; },
    addEventListener(){},
    createElement(){ return makeFakeElement('anon', {}); },
  };
  const sandbox = {
    console, Math, Date, Array, Object, Number, String, Set, Map, JSON,
    document: documentStub, window: undefined,
    requestAnimationFrame(){},
    Chart: class { constructor(ctx, config){ this.data = config && config.data ? JSON.parse(JSON.stringify(config.data)) : { labels: [], datasets: [{ data: [] }] }; } update(){} },
    URL: { createObjectURL(){ return 'blob://fake'; } },
    Blob: class {},
    setTimeout: (fn) => { fn && fn(); return 0; },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(extractScript(), context, { filename: 'v7.3-recorte-test.js' });
  return context.__api;
}

function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n;
  let sxy=0, sxx=0, syy=0;
  for (let i=0;i<n;i++){ const dx=xs[i]-mx, dy=ys[i]-my; sxy+=dx*dy; sxx+=dx*dx; syy+=dy*dy; }
  return sxy/Math.sqrt(sxx*syy);
}

async function correr(clampLuminosity, seed) {
  const api = buildSandbox({ clampLuminosity });
  api.setSeed(seed);
  api.resetSimulation();

  // Mismos parámetros que Experimento A de la batería 1 (réplica declarada de
  // "la única corrida limpia"): tc_ptc=18, exponente_ptc=4.1, resto default.
  api.els.powerBase.value = '0.47';
  api.els.beta.value = '0.94';
  api.els.sigma.value = '6.8';
  api.els.noise.value = '0.0079';
  api.els.band.value = '1.105';
  api.els.tOpt.value = '25';
  api.els.ptcTc.value = '18';
  api.els.ptcSharp.value = '4.1';
  api.els.minTemp.value = '-6';
  api.els.maxTemp.value = '25';
  api.els.dayNightToggle.checked = false;

  // Controles del barrido: v7.3 los lee directo de document.getElementById.
  // Escala completa, igual a Experimento A de la batería 1: 60 puntos,
  // settle=300, measure=120, para poder comparar el r resultante directo
  // contra el -0.756 original y el -0.236 de la batería 1.
  api.document.getElementById('sweepAxis').value = 'luminosity';
  api.document.getElementById('sweepFrom').value = '0.25';
  api.document.getElementById('sweepTo').value = '1.95';
  api.document.getElementById('sweepSteps').value = '60';
  api.document.getElementById('sweepSettle').value = '300';
  api.document.getElementById('sweepMeasure').value = '120';
  api.document.getElementById('sweepNoise').value = '0';
  api.document.getElementById('sweepTraceN').value = '0';

  await api.runSweep();
  return api.getSweepRows();
}

const seed = 7;
console.log(`Corriendo SOLO la versión CON recorte (escala completa, 60 puntos, settle=300, measure=120), semilla=${seed}...`);
const filasClamp = await correr(true, seed);
console.log('lista, calculando...');

function resumen(nombre, rows) {
  const xs = rows.map(r=>r.x);
  const foot = rows.map(r=>r.footprint);
  const hAbs = rows.map(r=>r.H_absLocal);
  const r = pearson(foot, hAbs);
  const fueraDeRango = xs.filter(x=>x<0.6||x>1.4).length;
  // ¿cuántas filas con x<0.6 tienen footprint IDÉNTICO (o casi) a la de x=0.6?
  const bajoRango = rows.filter(row=>row.x<0.6);
  const distintosFootprintBajoRango = new Set(bajoRango.map(row=>row.footprint.toFixed(6))).size;
  console.log(`\n=== ${nombre} ===`);
  console.log(`filas: ${rows.length}, puntos fuera de [0.6,1.4] (por el x pedido): ${fueraDeRango}`);
  console.log(`de esos, footprint distintos entre sí (debería ser >1 si NO hay recorte, y muy bajo si SÍ hay recorte): ${distintosFootprintBajoRango} de ${bajoRango.length}`);
  console.log(`correlación huella<->entropía (Pearson) sobre las 60 filas: ${r.toFixed(4)}`);
  console.log(`primeros 5 puntos (x, footprint, H_absLocal):`);
  for (let i=0;i<5;i++) console.log(`  x=${xs[i].toFixed(4)}  footprint=${foot[i].toFixed(4)}  H_absLocal=${hAbs[i].toFixed(4)}`);
}

resumen('CON recorte (como un navegador real de verdad haría), escala completa', filasClamp);
console.log('\ncurva completa (x, footprint):');
for (const r of filasClamp) console.log(`  x=${r.x.toFixed(4)}  footprint=${r.footprint.toFixed(4)}  H_absLocal=${r.H_absLocal.toFixed(4)}`);
