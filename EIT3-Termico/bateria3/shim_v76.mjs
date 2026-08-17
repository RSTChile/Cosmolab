// Shim mínimo de DOM para ejecutar el <script> real de ET3-Termico_v7.6.1.html
// dentro de Node, sin navegador. Mismo patrón que bateria2/shim_v75.mjs.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = path.join(__dirname, '..', 'ET3-Termico_v7.6.1.html');

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
  claveSemilla, sembrarFase,
  instantanea, restaurarInstantanea, asentarHastaEquilibrio, medirRecuperacion, varianzaYAutocorr,
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
    min: undefined, max: undefined, step: undefined,
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
  vm.runInContext(code, context, { filename: 'sim-script-v761.js' });

  const api = context.__api;
  api.getDescargas = () => descargas;
  api.limpiarDescargas = () => { descargas.length = 0; };
  api.getEl = getEl;
  return api;
}
