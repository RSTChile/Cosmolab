// RONDA 16 (13-ago-2026) — ¿cuál es el modelo correcto de e_R?
//
// POR QUÉ UNA SOLA CORRIDA. Se verificó en el motor que state.err NO
// realimenta la física: sólo lo leen state.Lambda (que nadie consume aguas
// arriba) y clasificarCierre(). Y Δ_struct entra en la física por su cuenta,
// no por su umbral. O sea: las cuatro variantes comparten EXACTAMENTE el mismo
// tejido físico y sólo difieren en cómo se lee. Entonces se corre la física una
// vez, se graban las series crudas por tick, y las variantes se calculan
// después en segundos. Cuatro corridas de una hora se vuelven una.
//
// LAS VARIANTES
//   M0  publicado      err = max(0, -ΔA)   κ_Δ = 0.5102 (p10 inventado)
//   M1  |e_R|          err = |ΔA|          κ_Δ = 0.5102
//   M2  + cota inf.    err = |ΔA|          κ_Δ = 0.5102   y err=0 ⇒ no viable
//   M3  canónico       err = |ΔA|          Δ_struct > 0   y err=0 ⇒ no viable
//
// M3 es la lectura literal de C-N2.8.5 ("0 < |e_R| < κ_O") y C-N2.8.4
// ("κ_Δ = inf(Δ_struct^operable) > 0"). M1 y M2 están para separar qué hace
// cada corrección por sí sola: sin ellas no se sabría si un cambio de resultado
// viene del valor absoluto o de la cota de abajo.
//
// AUTOVERIFICACIÓN. La serie de err de M0 se reconstruye desde A_sys_env y se
// compara contra el err que el motor grabó de verdad. Si la reconstrucción no
// es exacta, el script ABORTA: sin eso, las variantes calculadas offline serían
// palabra mía y no medición.
'use strict';
const fs = require('fs');
const path = require('path');
const motor = require('./motor_fisico.generado.js');

const ZONAS = ['JARDIN_FERTIL', 'CIERRE', 'SELVA_HOSTIL', 'COLAPSO'];
const PARAMETROS_FABRICA = {
  powerBase: 0.47, beta: 0.94, noise: 0.0079, tOpt: 13.0, ptcTc: 16.0,
  ptcSharp: 1.0, luminosity: 0.94, umbralGerminacion: 15, rezagoGyriosomus: 30,
};
const KAPPA_LF = 0.35, KAPPA_V = 0.70, KAPPA_O = 0.20, KAPPA_D_VIEJO = 0.5102;

// ---------------------------------------------------------------------------
// 1. La única corrida de física
// ---------------------------------------------------------------------------
// CACHÉ. La física es determinista (semilla fija) y no depende de ninguna de
// las banderas que se están comparando, así que se graba una vez y se reusa.
// Es lo que permite iterar sobre las variantes sin volver a pagar 13 minutos.
// Se invalida sola si cambia el tamaño de la corrida o el motor.
const N = (motor.TOPE_DIA_CALENDARIO + 1) * motor.TICKS_POR_DIA;
const RUTA_CACHE = path.join(__dirname, 'r16_fisica_cache.bin');
const SELLO_MOTOR = require('crypto').createHash('sha256')
  .update(fs.readFileSync(path.join(__dirname, 'motor_fisico.generado.js'))).digest('hex').slice(0, 16);

const sLF = new Float64Array(N), sDelta = new Float64Array(N);
const sA = new Float64Array(N), sErrMotor = new Float64Array(N);
const sAnio = new Int16Array(N);

function guardarCache() {
  const cab = Buffer.from(JSON.stringify({ N, sello: SELLO_MOTOR }).padEnd(256, ' '), 'utf-8');
  fs.writeFileSync(RUTA_CACHE, Buffer.concat([cab, Buffer.from(sLF.buffer), Buffer.from(sDelta.buffer),
    Buffer.from(sA.buffer), Buffer.from(sErrMotor.buffer), Buffer.from(sAnio.buffer)]));
}
function cargarCache() {
  if (!fs.existsSync(RUTA_CACHE)) return false;
  const b = fs.readFileSync(RUTA_CACHE);
  let meta;
  try { meta = JSON.parse(b.subarray(0, 256).toString('utf-8').trim()); } catch { return false; }
  if (meta.N !== N || meta.sello !== SELLO_MOTOR) {
    console.log('  (caché descartada: el motor o el tamaño de la corrida cambiaron)');
    return false;
  }
  let o = 256;
  for (const arr of [sLF, sDelta, sA, sErrMotor, sAnio]) {
    b.copy(Buffer.from(arr.buffer), 0, o, o + arr.byteLength);
    o += arr.byteLength;
  }
  console.log('  física leída de la caché (misma corrida determinista, motor idéntico)');
  return true;
}

const hayCache = cargarCache();
motor.ERR_ABSOLUTO = false;   // se graba el err ORIGINAL, para poder verificar
motor.rngTf = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'Tf'));
motor.rngEco = motor.mulberry32(motor.claveSemilla('regimen1966-2027', 'eco'));
Object.assign(motor.state, PARAMETROS_FABRICA, { dayNightMode: true, seasonMode: true });
motor.state.tick = 0; motor.state.step = 0;
motor.state.Tf = 24.6; motor.state.Tc = 25; motor.state.Th = 28;
motor.state.floracion = 0; motor.state.gyriosomus = 0; motor.state.sueloDesnudo = 1;
motor.state.floracionHistorial = [];
motor.state.powerLive = motor.state.powerBase; motor.state._A_prev = 0;
motor.resetField(); motor.resetBuffers();

if (!hayCache) {
  for (let i = 0; i < 60 * motor.TICKS_POR_DIA; i++) motor.pasoFisica(false);
  motor.state.tick = 0; motor.state.step = 0;

  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    motor.pasoFisica(false);
    sLF[i] = motor.state.LF;
    sDelta[i] = motor.state.deltaStruct;
    sA[i] = motor.state.A_sys_env;
    sErrMotor[i] = motor.state.err;
    sAnio[i] = motor.ANIO_CERO + Math.floor(motor.diaCalendarioActual() / motor.DIAS_POR_ANIO_CAL);
    if (i % 200000 === 0) {
      const seg = (Date.now() - t0) / 1000, ritmo = (i + 1) / seg;
      process.stderr.write(`  ${(i / N * 100).toFixed(1)}% · faltan ~${Math.round((N - i) / ritmo / 60)} min\n`);
    }
  }
  process.stderr.write(`  física lista en ${((Date.now() - t0) / 60000).toFixed(1)} min\n`);
  guardarCache();
}

// ---------------------------------------------------------------------------
// 2. Reconstrucción de |ΔA| y verificación contra lo que grabó el motor
// ---------------------------------------------------------------------------
// A_sys_env sólo cambia en el borde de cada semana real; entre bordes queda
// fijo. Detectar el cambio da ΔA = A_nueva - A_previa, que es exactamente lo
// que calcula actualizarViabilidadSemanal(). El primer borde no cuenta: ahí el
// motor pone err=0 porque todavía no hay semana anterior con qué comparar.
// LA PRIMERA SEMANA SE EXCLUYE, y no por comodidad. Los acumuladores semanales
// del motor son de módulo y SOBREVIVEN a los 60 días de asentamiento térmico:
// cuando se resetea el reloj para arrancar 1966, el motor todavía trae cargada
// la última semana del asentamiento, así que en el primer borde publica un
// error heredado de un tramo que no es parte del experimento. Ese arrastre no
// se puede reconstruir desde A_sys_env porque su signo se perdió. Son 420 ticks
// de 1.360.800 (0,03%, dentro de 1966) y se excluyen IGUAL en las cuatro
// variantes, así que la comparación entre ellas no se ve afectada.
const sErrAbs = new Float64Array(N);
const sErrM0 = new Float64Array(N);
let i0 = -1;
for (let i = 1; i < N; i++) if (sA[i] !== sA[0]) { i0 = i; break; }
if (i0 < 0) { console.error('ABORTA: A_sys_env nunca cambia en toda la corrida.'); process.exit(1); }

let aPrev = sA[0], errAbs = 0, errM0 = 0, bordes = 0;
for (let i = i0; i < N; i++) {
  if (sA[i] !== aPrev) {
    bordes++;
    const dA = sA[i] - aPrev;
    errAbs = Math.abs(dA);
    errM0 = Math.max(0, -dA);
    aPrev = sA[i];
  }
  sErrAbs[i] = errAbs;
  sErrM0[i] = errM0;
}
let discrepancias = 0, peor = 0;
for (let i = i0; i < N; i++) {
  const d = Math.abs(sErrM0[i] - sErrMotor[i]);
  if (d > 1e-12) { discrepancias++; if (d > peor) peor = d; }
}
console.log(`Verificación: ${bordes} bordes de semana reconstruidos, ` +
            `${discrepancias} ticks discrepantes de ${N - i0} (peor ${peor.toExponential(2)})`);
console.log(`Excluidos los primeros ${i0} ticks (arrastre del asentamiento térmico).`);
if (discrepancias > 0) {
  console.error('ABORTA: la reconstrucción de e_R no reproduce lo que grabó el motor.');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// 3. Las variantes
// ---------------------------------------------------------------------------
const VARIANTES = [
  { id: 'M0_publicado', err: sErrM0,  cotaInferior: false, deltaCanonico: false,
    nota: 'lo que está publicado: err recortado a la baja, κ_Δ = p10 inventado' },
  { id: 'M1_abs',       err: sErrAbs, cotaInferior: false, deltaCanonico: false,
    nota: '|e_R| como lo escribe el canon; sin cota inferior todavía' },
  { id: 'M2_abs_cota',  err: sErrAbs, cotaInferior: true,  deltaCanonico: false,
    nota: '|e_R| + la cota de abajo: err = 0 no es viable (C-N2.8.8a)' },
  { id: 'M3_canonico',  err: sErrAbs, cotaInferior: true,  deltaCanonico: true,
    nota: 'lectura literal: 0 < |e_R| < κ_O y Δ_struct > 0' },
];

function percentiles(arr) {
  const c = Float64Array.from(arr); c.sort();
  const p = (q) => c[Math.min(c.length - 1, Math.floor(q * c.length))];
  return { min: c[0], p10: p(0.10), p50: p(0.50), p90: p(0.90), max: c[c.length - 1] };
}

// LA TABLA QUE FALTABA (R16-3). Es el control que no teníamos: cada invariante
// contra su umbral, ANTES de mirar los cinco criterios. Esta tabla habría
// mostrado la mediana cero de e_R en cualquier momento del último mes.
function tablaInvariantes(err) {
  const f = (x) => x.toFixed(4).padStart(8);
  const filas = [
    ['LF', percentiles(sLF.subarray(i0)), KAPPA_LF, 'piso'],
    ['Δ_struct', percentiles(sDelta.subarray(i0)), KAPPA_D_VIEJO, 'piso'],
    ['A_sys_env', percentiles(sA.subarray(i0)), KAPPA_V, 'piso'],
    ['|e_R|', percentiles(err.subarray(i0)), KAPPA_O, 'techo'],
  ];
  console.log('    invariante      mín      p10      p50      p90      máx   umbral   veredicto');
  for (const [nombre, q, umbral, tipo] of filas) {
    let veredicto;
    if (tipo === 'piso') {
      veredicto = q.min >= umbral ? 'NUNCA se viola'
                : q.p90 < umbral ? 'casi siempre violado'
                : 'discrimina';
    } else {
      veredicto = q.max < umbral ? (q.p50 === 0 ? 'holgado, pero MEDIANA CERO' : 'holgado: nunca colapsa')
                : 'llega al techo';
    }
    console.log(`    ${nombre.padEnd(10)}${f(q.min)} ${f(q.p10)} ${f(q.p50)} ${f(q.p90)} ${f(q.max)} ` +
                `${umbral.toFixed(4).padStart(8)}   ${veredicto}`);
  }
  let ceros = 0;
  for (let i = i0; i < N; i++) if (!(err[i] > 0)) ceros++;
  console.log(`    e_R = 0 (sin señal correctora, C-N2.8.8a): ${(100 * ceros / (N - i0)).toFixed(2)}% del tiempo`);
}

const salida = [];
for (const v of VARIANTES) {
  console.log(`\n${'='.repeat(78)}\n${v.id} — ${v.nota}`);
  tablaInvariantes(v.err);

  const conteo = {};
  let mandaLF = 0, mandaDelta = 0, mandaA = 0, mandaE = 0;
  for (let i = i0; i < N; i++) {
    const ratioLF = sLF[i] / KAPPA_LF;
    const ratioDelta = v.deltaCanonico ? (sDelta[i] > 0 ? Infinity : 0) : sDelta[i] / KAPPA_D_VIEJO;
    const ratioA = sA[i] / KAPPA_V;
    let ratioE = 2 - v.err[i] / KAPPA_O;
    if (v.cotaInferior && !(v.err[i] > 0)) ratioE = 0;
    const activo = Math.min(ratioLF, ratioDelta) >= 1;
    const viable = Math.min(ratioA, ratioE) >= 1;
    if (ratioLF <= ratioDelta) mandaLF++; else mandaDelta++;
    if (ratioA <= ratioE) mandaA++; else mandaE++;
    const zona = viable ? (activo ? 'JARDIN_FERTIL' : 'CIERRE')
                        : (activo ? 'SELVA_HOSTIL' : 'COLAPSO');
    const a = sAnio[i];
    if (!conteo[a]) conteo[a] = { JARDIN_FERTIL: 0, CIERRE: 0, SELVA_HOSTIL: 0, COLAPSO: 0 };
    conteo[a][zona]++;
  }
  const pc = (x) => (100 * x / (N - i0)).toFixed(1) + '%';
  console.log(`    quién decide la ACTIVACIÓN: LF ${pc(mandaLF)} · Δ_struct ${pc(mandaDelta)}`);
  console.log(`    quién decide la VIABILIDAD: A_sys_env ${pc(mandaA)} · e_R ${pc(mandaE)}`);

  const anios = Object.keys(conteo).map(Number).sort((a, b) => a - b);
  const filas = anios.map((a) => {
    const c = conteo[a], tot = ZONAS.reduce((s, z) => s + c[z], 0);
    const dom = ZONAS.reduce((mx, z) => (c[z] > c[mx] ? z : mx), ZONAS[0]);
    return [a, ...ZONAS.map((z) => (c[z] / tot * 100).toFixed(2)), dom].join(',');
  });
  const ruta = path.join(__dirname, `r16_${v.id}_por_anio.csv`);
  fs.writeFileSync(ruta, ['anio,JARDIN_FERTIL_pct,CIERRE_pct,SELVA_HOSTIL_pct,COLAPSO_pct,dominante',
                          ...filas].join('\n'));
  const dom = {};
  for (const a of anios) { const d = filas[anios.indexOf(a)].split(',')[5]; dom[d] = (dom[d] || 0) + 1; }
  console.log(`    años por zona dominante: ${Object.entries(dom).map(([k, n]) => `${k} ${n}`).join(' · ')}`);
  salida.push(ruta);
}

console.log(`\n${'='.repeat(78)}`);
console.log('CSV escritos (pasarlos a evaluar_contra_ground_truth.js):');
for (const r of salida) console.log('  ' + path.basename(r));
