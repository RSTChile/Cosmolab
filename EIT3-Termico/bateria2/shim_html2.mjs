// Shim mínimo de DOM para ejecutar el <script> real de v7.4.1.html dentro de
// Node, sin navegador. Extiende el shim de la batería anterior (bateria/shim_html.mjs)
// agregando: sweepReset/seedInput/ids de barrido (accesibles vía getEl genérico,
// no solo vía `els`), y captura real de los CSV que produce exportSweepCSV()/
// exportTracesCSV() a través de un <a> falso + Blob + URL.createObjectURL.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = path.join(__dirname, '..', 'EIT3_Termico_kappaH_v7.4.1.html');

const EPILOGUE = `
;globalThis.__api = {
  getState: () => state,
  getField: () => field,
  getABuf: () => aBuf,
  getNoiseEchoBuf: () => noiseEchoBuf,
  getAwin: () => _Awin,
  gridSize, HCFG,
  els,
  setSeed, resetSimulation, resetField, reiniciarSilencioso,
  stepHeadless, updateSimulation, pasoFisica,
  runSweep,
  syncStateFromUI, updateLabels,
  entropyLocalAbs, entropyRel, entropyAtWidth, shannonEntropy,
  passiveNoiseSample, errRate, errRatePush,
  computeDaisyworld, computeLFandErr, computeCoupling, computeDeltaStruct,
  abioticTf, evolveField, classifyRegime, ptcResponse, clamp, lerp, pseudoNoise,
  updateBehavioralEntropy,
  getSweepRows: () => sweepRows,
  getSweepTraces: () => sweepTraces,
};
`;

function extractScript() {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const m = html.match(/<script>([\s\S]*)<\/script>/);
  if (!m) throw new Error('no se encontró <script> en el HTML');
  return m[1] + EPILOGUE;
}

function makeFakeElement(id) {
  const ctx2d = {
    clearRect(){}, fillRect(){}, beginPath(){}, arc(){}, stroke(){}, fill(){},
    moveTo(){}, lineTo(){}, fillText(){}, setLineDash(){},
    fillStyle:'', strokeStyle:'', lineWidth:1, font:'',
  };
  return {
    id,
    value: '0',
    textContent: '',
    checked: false,
    className: '',
    style: {},
    disabled: false,
    hidden: false,
    dataset: {},
    width: 520, height: 520,
    _listeners: {},
    addEventListener(ev, fn){ (this._listeners[ev] ||= []).push(fn); },
    removeEventListener(){},
    setAttribute(){}, getAttribute(){ return null; },
    appendChild(){}, prepend(){}, remove(){},
    closest(){ return null; },
    getContext(){ return ctx2d; },
    getBoundingClientRect(){ return { width: 400, height: 220 }; },
    querySelectorAll(){ return []; },
  };
}

class FakeChart {
  constructor(ctx, config){
    this.data = config && config.data ? JSON.parse(JSON.stringify(config.data)) : { labels: [], datasets: [{ data: [] }] };
  }
  update(){}
}

export function buildSandbox() {
  const elements = new Map();
  const getEl = (id) => {
    if (!elements.has(id)) elements.set(id, makeFakeElement(id));
    return elements.get(id);
  };

  // Captura de descargas: exportSweepCSV()/exportTracesCSV()/exportCSV()/exportLog()
  // crean <a>, arman un Blob con el CSV y llaman a.click(). El patrón real es
  // siempre síncrono (new Blob -> URL.createObjectURL -> a.href=url -> a.click()),
  // así que basta con recordar el último Blob creado y leerlo en el click().
  const descargas = [];
  let ultimoBlobTexto = null;

  class FakeBlob {
    constructor(parts){ this.parts = parts; }
  }
  const URLStub = {
    createObjectURL(blob){ ultimoBlobTexto = (blob && blob.parts) ? blob.parts.join('') : null; return 'blob://fake'; },
    revokeObjectURL(){},
  };

  function makeAnchor() {
    const a = makeFakeElement('a');
    a.href = '';
    a.download = '';
    a.click = function(){
      descargas.push({ nombre: a.download, contenido: ultimoBlobTexto, href: a.href });
    };
    return a;
  }

  const documentStub = {
    getElementById: getEl,
    querySelectorAll(){ return []; },
    addEventListener(){},
    createElement(tag){ return tag === 'a' ? makeAnchor() : makeFakeElement('anon'); },
  };

  const sandbox = {
    console,
    Math,
    Date,
    Array,
    Object,
    Number,
    String,
    Set,
    Map,
    JSON,
    document: documentStub,
    window: undefined,
    requestAnimationFrame(){},
    Chart: FakeChart,
    URL: URLStub,
    Blob: FakeBlob,
    setTimeout: (fn) => { fn && fn(); return 0; },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  const code = extractScript();
  vm.runInContext(code, context, { filename: 'sim-script-v741.js' });

  const api = context.__api;
  api.getDescargas = () => descargas;
  api.limpiarDescargas = () => { descargas.length = 0; };
  api.getEl = getEl;
  return api;
}
