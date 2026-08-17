// Shim mínimo de DOM para ejecutar el <script> real del HTML v7.3 (modificado)
// dentro de Node, sin navegador. Sirve como el "camino 2 — navegador headless"
// del encargo: es literalmente el mismo código fuente del simulador, no una
// reimplementación.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = path.join(__dirname, '..', 'EIT3_Termico_kappaH_v7.3.html');

// El epílogo va CONCATENADO al mismo texto del <script> (mismo top-level scope),
// no en un vm.runInContext separado: así puede ver las const/let del script
// (state, field, els, aBuf...) sin depender de si el contexto comparte el
// lexical environment global entre llamadas separadas a runInContext.
const EPILOGUE = `
;globalThis.__api = {
  getState: () => state,
  getField: () => field,
  getABuf: () => aBuf,
  getNoiseEchoBuf: () => noiseEchoBuf,
  getAwin: () => _Awin,
  gridSize, HCFG,
  els,
  setSeed, resetSimulation, resetField,
  stepHeadless, updateSimulation, pasoFisica,
  syncStateFromUI, updateLabels,
  entropyLocalAbs, entropyRel, entropyAtWidth, shannonEntropy,
  passiveNoiseSample, errRate, errRatePush,
  computeDaisyworld, computeLFandErr, computeCoupling, computeDeltaStruct,
  abioticTf, evolveField, classifyRegime, ptcResponse, clamp, lerp, pseudoNoise,
  updateBehavioralEntropy,
};
`;

function extractScript() {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const m = html.match(/<script>([\s\S]*)<\/script>/);
  if (!m) throw new Error('no se encontró <script> en el HTML');
  return m[1] + EPILOGUE;
}

// Elementos falsos: todo lo que el script pide via $(id) debe soportar
// .value / .textContent / .checked / .className / .style / addEventListener /
// getContext (para el canvas) / getBoundingClientRect (para el quadrantCanvas).
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
    // clona la config real (evoChart trae 5 datasets, phaseChart 1, gauges 1)
    // para que datasets[i].data exista para cualquier índice que el script use.
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

  const documentStub = {
    getElementById: getEl,
    querySelectorAll(){ return []; },
    addEventListener(){},
    createElement(){ return makeFakeElement('anon'); },
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
    requestAnimationFrame(){ /* no-op: nunca se agenda un loop real */ },
    Chart: FakeChart,
    URL: { createObjectURL(){ return 'blob://fake'; } },
    Blob: class {},
    setTimeout: (fn) => { /* no-op síncrono: el sweep real usa await setTimeout(r,0) */ fn && fn(); return 0; },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  const code = extractScript();
  vm.runInContext(code, context, { filename: 'sim-script.js' });
  return context.__api; // getState/getField + setSeed/stepHeadless/updateSimulation/resetSimulation/etc.
}
