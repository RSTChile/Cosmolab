import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = path.join(__dirname, '..', '..', 'Old', 'EIT3_Termico_kappaH_v7.3.html');

const EPILOGUE = `
;globalThis.__api = { getState: () => state, setSeed, resetSimulation, stepHeadless };
`;

function extractScript() {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const m = html.match(/<script>([\s\S]*)<\/script>/);
  return m[1] + EPILOGUE;
}

function makeFakeElement(id) {
  const ctx2d = { clearRect(){}, fillRect(){}, beginPath(){}, arc(){}, stroke(){}, fill(){}, moveTo(){}, lineTo(){}, fillText(){}, setLineDash(){}, fillStyle:'', strokeStyle:'', lineWidth:1, font:'' };
  return {
    id, value:'0', textContent:'', checked:false, className:'', style:{}, disabled:false, hidden:false, dataset:{}, width:520, height:520,
    _listeners:{}, addEventListener(ev,fn){(this._listeners[ev]||=[]).push(fn);}, removeEventListener(){},
    setAttribute(){}, getAttribute(){return null;}, appendChild(){}, prepend(){}, remove(){}, closest(){return null;},
    getContext(){return ctx2d;}, getBoundingClientRect(){return {width:400,height:220};}, querySelectorAll(){return [];},
  };
}
function buildSandbox() {
  const elements = new Map();
  const getEl = (id) => { if(!elements.has(id)) elements.set(id, makeFakeElement(id)); return elements.get(id); };
  const documentStub = { getElementById:getEl, querySelectorAll(){return [];}, addEventListener(){}, createElement(){return makeFakeElement('anon');} };
  const sandbox = {
    console, Math, Date, Array, Object, Number, String, Set, Map, JSON,
    document: documentStub, window: undefined, requestAnimationFrame(){},
    Chart: class { constructor(ctx,config){ this.data = config && config.data ? JSON.parse(JSON.stringify(config.data)) : {labels:[],datasets:[{data:[]}]}; } update(){} },
    URL: { createObjectURL(){return 'blob://fake';} }, Blob: class {},
    setTimeout: (fn) => { fn && fn(); return 0; },
  };
  sandbox.window = sandbox; sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(extractScript(), context, { filename: 'debug.js' });
  return context.__api;
}

console.log('construyendo sandbox...');
const t0 = Date.now();
const api = buildSandbox();
console.log(`sandbox lista en ${Date.now()-t0}ms`);
api.setSeed(7);
console.log('llamando resetSimulation()...');
const t1 = Date.now();
api.resetSimulation();
console.log(`resetSimulation() listo en ${Date.now()-t1}ms`);

console.log('corriendo 20 stepHeadless() con timing individual...');
for (let i=0;i<20;i++){
  const ta = Date.now();
  api.stepHeadless();
  console.log(`paso ${i}: ${Date.now()-ta}ms`);
}
