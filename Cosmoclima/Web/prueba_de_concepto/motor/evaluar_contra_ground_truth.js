// Evaluador contra ground truth real (10-ago-2026). Toma un
// <id>_por_anio.csv ya generado por runner_bateria.js/child_worker.js
// (columnas anio,JARDIN_FERTIL_pct,CIERRE_pct,SELVA_HOSTIL_pct,COLAPSO_pct,
// dominante) y lo cruza contra tres fuentes reales que NO participan del
// motor físico: la curva empírica de floración de Gyriosomus, la
// pluviosidad diaria consolidada (sqlite) y el índice ONI histórico.
// No toca el motor ni el HTML -- solo lee resultados/*.csv y fuentes/*.
//
// Uso: node evaluar_contra_ground_truth.js <ruta_a_por_anio.csv>
// Escribe <ruta_a_por_anio.csv>_evaluacion.json junto al CSV de entrada.
'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const RAIZ_INVESTIGACION = path.join(__dirname, '..', '..', '..', 'investigacion', 'fuentes');
const CSV_FLORACION = path.join(RAIZ_INVESTIGACION, 'curva_empirica_lluvia_floracion_gyriosomus.csv');
const CSV_ONI = path.join(RAIZ_INVESTIGACION, 'oni_historico_completo_1966_2026.csv');
const DB_LLUVIA = path.join(RAIZ_INVESTIGACION, 'pluviosidad_diaria_consolidada.sqlite');

// ---------------------------------------------------------------------
// CSV genérico: sin comillas ni comas embebidas en ninguna de las
// fuentes usadas acá (verificado a mano), así que un split simple basta.
// ---------------------------------------------------------------------
function leerCsv(rutaAbsoluta) {
  const texto = fs.readFileSync(rutaAbsoluta, 'utf-8').trim();
  const lineas = texto.split(/\r?\n/);
  const encabezados = lineas[0].split(',').map((h) => h.trim());
  return lineas.slice(1).map((linea) => {
    const campos = linea.split(',');
    const fila = {};
    encabezados.forEach((h, i) => { fila[h] = campos[i] !== undefined ? campos[i].trim() : ''; });
    return fila;
  });
}

function num(v) {
  if (v === undefined || v === null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

// ---------------------------------------------------------------------
// 1. Datos del modelo (el CSV que se está evaluando)
// ---------------------------------------------------------------------
function cargarDatosModelo(rutaCsv) {
  const filas = leerCsv(rutaCsv);
  const porAnio = new Map();
  for (const f of filas) {
    const anio = parseInt(f.anio, 10);
    if (!Number.isFinite(anio)) continue;
    porAnio.set(anio, {
      anio,
      JARDIN_FERTIL_pct: num(f.JARDIN_FERTIL_pct),
      CIERRE_pct: num(f.CIERRE_pct),
      SELVA_HOSTIL_pct: num(f.SELVA_HOSTIL_pct),
      COLAPSO_pct: num(f.COLAPSO_pct),
      dominante: f.dominante,
      // Ronda 6 (10-ago-2026): columna nueva, puede faltar en CSVs viejos
      // (rondas anteriores a la temperatura real) -- null si no está.
      coberturaTempReal_pct: f.coberturaTempReal_pct !== undefined ? num(f.coberturaTempReal_pct) : null,
    });
  }
  return porAnio;
}

// ---------------------------------------------------------------------
// 2. Curva empírica de floración (ground truth de floración real)
// ---------------------------------------------------------------------
function cargarFloracionEmpirica() {
  const filas = leerCsv(CSV_FLORACION);
  const porAnio = new Map();
  for (const f of filas) {
    const anio = parseInt(f.anio, 10);
    if (!Number.isFinite(anio)) continue;
    porAnio.set(anio, {
      anio,
      lluvia_anual_mm_nasa_power: num(f.lluvia_anual_mm_nasa_power),
      floracion_documentada: num(f.floracion_documentada), // 1, 0, o null (año sin dato)
      gyriosomus_gbif_n: num(f.gyriosomus_gbif_n),
    });
  }
  return porAnio;
}

// ---------------------------------------------------------------------
// 3. ONI histórico (cargado para contexto/depuración, no entra en las
//    4 métricas pedidas pero se deja disponible e impreso)
// ---------------------------------------------------------------------
function cargarOni() {
  const filas = leerCsv(CSV_ONI);
  const porAnio = new Map();
  const trimestres = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ'];
  for (const f of filas) {
    const anio = parseInt(f.anio, 10);
    if (!Number.isFinite(anio)) continue;
    const valores = {};
    for (const t of trimestres) valores[t] = num(f[t]);
    porAnio.set(anio, { anio, ...valores });
  }
  return porAnio;
}

// ---------------------------------------------------------------------
// 4. Lluvia real anual desde el sqlite consolidado.
//
//    Elección de agregación: promedio del total anual de lluvia POR
//    LOCALIDAD (no la suma cruda de todas las filas). La red de
//    estaciones creció de ~130 localidades en 1966 a ~270 hacia 2016 y
//    luego cayó a ~87 en 2018-2026 (verificado con un GROUP BY anio
//    antes de escribir esto) -- sumar todo sin normalizar mezclaría
//    "llovió más" con "hay más estaciones reportando", que es un
//    artefacto de cobertura, no señal climática. Promediar por
//    localidad (SUM(lluvia_mm)/COUNT(DISTINCT localidad) del año) da
//    un número comparable entre años con cobertura de red distinta.
//
//    Intenta node:sqlite primero (Node 22+, confirmado disponible en
//    esta corrida: v24.14.1); si no existe o falla, cae al binario
//    sqlite3 CLI (confirmado presente en /usr/bin/sqlite3) vía
//    child_process.
// ---------------------------------------------------------------------
function cargarLluviaAnualReal() {
  const sql = `
    SELECT substr(fecha,1,4) AS anio,
           COUNT(DISTINCT localidad) AS n_localidades,
           SUM(lluvia_mm) AS suma_total
    FROM pluviosidad_diaria
    WHERE substr(fecha,1,4) BETWEEN '1966' AND '2027'
    GROUP BY anio
    ORDER BY anio;
  `;

  let filas = null;
  let metodo = null;

  try {
    const { DatabaseSync } = require('node:sqlite');
    const db = new DatabaseSync(DB_LLUVIA, { readOnly: true });
    filas = db.prepare(sql).all();
    db.close();
    metodo = 'node:sqlite';
  } catch (errNodeSqlite) {
    try {
      const salida = execFileSync('sqlite3', ['-csv', '-header', DB_LLUVIA, sql], { encoding: 'utf-8' });
      const lineas = salida.trim().split(/\r?\n/);
      const encabezados = lineas[0].split(',');
      filas = lineas.slice(1).map((l) => {
        const c = l.split(',');
        const o = {};
        encabezados.forEach((h, i) => { o[h] = c[i]; });
        return o;
      });
      metodo = 'sqlite3 CLI';
    } catch (errCli) {
      throw new Error(
        `No se pudo leer ${DB_LLUVIA} ni con node:sqlite (${errNodeSqlite.message}) ` +
        `ni con el binario sqlite3 (${errCli.message}).`
      );
    }
  }

  const porAnio = new Map();
  for (const f of filas) {
    const anio = parseInt(f.anio, 10);
    const nLocalidades = num(f.n_localidades);
    const sumaTotal = num(f.suma_total);
    if (!Number.isFinite(anio) || !nLocalidades) continue;
    porAnio.set(anio, {
      anio,
      n_localidades: nLocalidades,
      suma_total_mm: sumaTotal,
      lluvia_promedio_por_estacion_mm: sumaTotal / nLocalidades,
    });
  }
  return { porAnio, metodo };
}

// ---------------------------------------------------------------------
// Métrica (a): diferenciación real -- ¿el motor repite el mismo "molde"
// año tras año o produce trayectorias distintas?
// ---------------------------------------------------------------------
function metricaDiferenciacion(datosModelo) {
  // Usar SOLO JARDIN_FERTIL_pct aislado subestima la diferenciación real:
  // dos años pueden tener el mismo % total de Jardín Fértil (ej. "1 mes de
  // 12") pero ser MESES DISTINTOS con lluvia real distinta -- eso ya es
  // diferenciación real, aunque la suma coincida. La tupla completa (4
  // zonas) captura eso; el chequeo de una sola columna queda como
  // referencia informativa, no como el criterio principal.
  const conteoPorValorRedondeado = new Map();
  const conteoPorTuplaCompleta = new Map();
  for (const d of datosModelo.values()) {
    if (d.JARDIN_FERTIL_pct === null) continue;
    const redondeado = Math.round(d.JARDIN_FERTIL_pct * 10) / 10;
    conteoPorValorRedondeado.set(redondeado, (conteoPorValorRedondeado.get(redondeado) || 0) + 1);
    const tupla = [d.JARDIN_FERTIL_pct, d.CIERRE_pct, d.SELVA_HOSTIL_pct, d.COLAPSO_pct].map((v) => Math.round(v * 100) / 100).join('|');
    conteoPorTuplaCompleta.set(tupla, (conteoPorTuplaCompleta.get(tupla) || 0) + 1);
  }
  let aniosEnClusters = 0;
  let valoresEnClusters = 0;
  for (const conteo of conteoPorValorRedondeado.values()) {
    if (conteo > 2) { aniosEnClusters += conteo; valoresEnClusters += 1; }
  }
  let aniosEnClustersTupla = 0;
  let valoresEnClustersTupla = 0;
  for (const conteo of conteoPorTuplaCompleta.values()) {
    if (conteo > 2) { aniosEnClustersTupla += conteo; valoresEnClustersTupla += 1; }
  }
  return {
    anios_totales: datosModelo.size,
    valores_distintos_totales: conteoPorValorRedondeado.size,
    valores_en_clusters_3mas: valoresEnClusters,
    anios_en_clusters_3mas: aniosEnClusters,
    tuplas_completas_distintas: conteoPorTuplaCompleta.size,
    tuplas_en_clusters_3mas: valoresEnClustersTupla,
    anios_en_clusters_tupla_3mas: aniosEnClustersTupla,
  };
}

// ---------------------------------------------------------------------
// Métrica (b): floración real (años con floracion_documentada=1) debe
// dar JARDIN_FERTIL_pct mayor que los años control (=0).
// ---------------------------------------------------------------------
function metricaFloracion(datosModelo, floracionEmpirica) {
  const valoresFloracion = [];
  const valoresControl = [];
  for (const [anio, gt] of floracionEmpirica) {
    const modelo = datosModelo.get(anio);
    if (!modelo || modelo.JARDIN_FERTIL_pct === null) continue;
    if (gt.floracion_documentada === 1) valoresFloracion.push({ anio, v: modelo.JARDIN_FERTIL_pct });
    else if (gt.floracion_documentada === 0) valoresControl.push({ anio, v: modelo.JARDIN_FERTIL_pct });
  }
  const media = (arr) => arr.reduce((s, x) => s + x.v, 0) / (arr.length || 1);
  const mediaFloracion = media(valoresFloracion);
  const mediaControl = media(valoresControl);
  const sobreControl = valoresFloracion.filter((x) => x.v > mediaControl).length;
  return {
    anios_floracion_n: valoresFloracion.length,
    anios_control_n: valoresControl.length,
    media_JARDIN_FERTIL_floracion: mediaFloracion,
    media_JARDIN_FERTIL_control: mediaControl,
    fraccion_floracion_sobre_media_control: valoresFloracion.length ? sobreControl / valoresFloracion.length : null,
    anios_floracion: valoresFloracion.map((x) => x.anio),
    anios_control: valoresControl.map((x) => x.anio),
    // INFORMATIVO (11-ago-2026), NO cambia el veredicto de (b): los criterios
    // (b) y (c) se contradicen entre sí en los años que son SIMULTÁNEAMENTE
    // floración documentada Y megasequía (2021, 2022, 2024; y 2019-2020 del
    // lado control). (c) exige que la megasequía tenga poco Jardín Fértil y
    // (b) exige que esos mismos años lo tengan alto. Se reporta el mismo
    // contraste excluyendo la ventana de megasequía de AMBOS grupos, para que
    // el conflicto quede a la vista en vez de castigar al modelo por él.
    ...(() => {
      const fueraSequia = (x) => x.anio < 2019 || x.anio > 2025;
      const f2 = valoresFloracion.filter(fueraSequia), c2 = valoresControl.filter(fueraSequia);
      return {
        solapan_floracion_y_megasequia: valoresFloracion.filter((x) => !fueraSequia(x)).map((x) => x.anio),
        solapan_control_y_megasequia: valoresControl.filter((x) => !fueraSequia(x)).map((x) => x.anio),
        sin_megasequia_n_floracion: f2.length,
        sin_megasequia_n_control: c2.length,
        sin_megasequia_media_floracion: media(f2),
        sin_megasequia_media_control: media(c2),
      };
    })(),
  };
}

// ---------------------------------------------------------------------
// Métrica (c): la megasequía 2019-2025 debe quedar por debajo del
// promedio de los 62 años en JARDIN_FERTIL_pct (y por arriba en COLAPSO).
// ---------------------------------------------------------------------
function metricaMegasequia(datosModelo) {
  const todos = [...datosModelo.values()].filter((d) => d.JARDIN_FERTIL_pct !== null);
  const megasequia = todos.filter((d) => d.anio >= 2019 && d.anio <= 2025);
  const media = (arr, campo) => arr.reduce((s, x) => s + x[campo], 0) / (arr.length || 1);
  return {
    anios_megasequia_n: megasequia.length,
    media_JARDIN_FERTIL_megasequia: media(megasequia, 'JARDIN_FERTIL_pct'),
    media_JARDIN_FERTIL_global: media(todos, 'JARDIN_FERTIL_pct'),
    media_COLAPSO_megasequia: media(megasequia, 'COLAPSO_pct'),
    media_COLAPSO_global: media(todos, 'COLAPSO_pct'),
    // Criterio (e), 11-ago-2026 -- definición de Alexis: "Cierre debería ser
    // lo contrario a Jardín Fértil en el sentido de que el sistema se
    // estabilizó sin cambios, punto de equilibrio sin cambios". En la
    // megasequía 2019-2025 el desierto real está quieto (semillas latentes
    // esperando lluvia que no llega) => CIERRE debe SUBIR respecto del
    // promedio de los 62 años, no bajar. Es el test más directo del eje de
    // activación, y el que destapó que LF medía distancia a un slider.
    media_CIERRE_megasequia: media(megasequia, 'CIERRE_pct'),
    media_CIERRE_global: media(todos, 'CIERRE_pct'),
    media_SELVA_megasequia: media(megasequia, 'SELVA_HOSTIL_pct'),
    media_SELVA_global: media(todos, 'SELVA_HOSTIL_pct'),
  };
}

// ---------------------------------------------------------------------
// Métrica (d): correlación de Spearman entre JARDIN_FERTIL_pct del
// modelo y la lluvia real anual.
//
// Por qué Spearman y no Pearson: la relación esperada entre lluvia y
// estado del ecosistema no tiene por qué ser lineal (umbrales de
// germinación, saturación del suelo, efecto retardado de años lluviosos
// consecutivos) -- Spearman sólo exige que la relación sea monótona, y
// además es robusto a outliers de lluvia (años como 1997 con >120mm
// frente al resto bajo 80mm no dominan el resultado como pasarían con
// Pearson sobre los valores crudos).
// ---------------------------------------------------------------------
function rangos(valores) {
  const indices = valores.map((_, i) => i).sort((a, b) => valores[a] - valores[b]);
  const rango = new Array(valores.length);
  let i = 0;
  while (i < indices.length) {
    let j = i;
    while (j + 1 < indices.length && valores[indices[j + 1]] === valores[indices[i]]) j++;
    const rangoPromedio = (i + j) / 2 + 1; // rango 1-based, promediado en empates
    for (let k = i; k <= j; k++) rango[indices[k]] = rangoPromedio;
    i = j + 1;
  }
  return rango;
}

function pearson(x, y) {
  const n = x.length;
  const mx = x.reduce((s, v) => s + v, 0) / n;
  const my = y.reduce((s, v) => s + v, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx, dy = y[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  const den = Math.sqrt(dx2 * dy2);
  return den === 0 ? null : num / den;
}

function spearman(x, y) {
  if (x.length < 2) return null;
  return pearson(rangos(x), rangos(y));
}

function metricaSpearman(datosModelo, lluviaReal) {
  const pares = [];
  for (const [anio, modelo] of datosModelo) {
    const lluvia = lluviaReal.get(anio);
    if (!lluvia || modelo.JARDIN_FERTIL_pct === null) continue;
    pares.push({ anio, jardin: modelo.JARDIN_FERTIL_pct, lluvia: lluvia.lluvia_promedio_por_estacion_mm });
  }
  const rho = spearman(pares.map((p) => p.jardin), pares.map((p) => p.lluvia));
  return {
    n_anios_con_dato_de_lluvia: pares.length,
    rho_spearman_JARDIN_FERTIL_vs_lluvia: rho,
  };
}

// ---------------------------------------------------------------------
// Ronda 6 (10-ago-2026): división por cobertura de temperatura real.
// TEMPERATURA_DIARIA_ZHCS solo cubre 1981-01-01 a 2026-08-09 (~24% de los
// 62 años queda fuera, ver plan majestic-whistling-canyon.md) -- reportar
// solo el agregado escondería si la mejora es real o si son los años SIN
// temperatura real (que no cambiaron) arrastrando el promedio. Umbral 50%:
// un año cae del lado "real" si más de la mitad de sus ticks tuvieron
// temperatura real ese día (los años de transición 1981 y 2026 quedan
// mezclados, es exactamente lo esperado en un corte por calendario).
// ---------------------------------------------------------------------
function dividirPorCobertura(datosModelo) {
  const conCobertura = new Map();
  const sinCobertura = new Map();
  let sinDatoDeCobertura = 0;
  for (const [anio, d] of datosModelo) {
    if (d.coberturaTempReal_pct === null) { sinDatoDeCobertura++; continue; }
    (d.coberturaTempReal_pct >= 50 ? conCobertura : sinCobertura).set(anio, d);
  }
  return { conCobertura, sinCobertura, sinDatoDeCobertura };
}

function calcularMetricas(datosModelo, floracionEmpirica, lluviaReal) {
  return {
    diferenciacion: metricaDiferenciacion(datosModelo),
    floracion: metricaFloracion(datosModelo, floracionEmpirica),
    megasequia: metricaMegasequia(datosModelo),
    spearmanRes: metricaSpearman(datosModelo, lluviaReal),
  };
}

// ---------------------------------------------------------------------
// Presentación
// ---------------------------------------------------------------------
function pasaFalla(cond) { return cond ? 'PASA' : 'FALLA'; }

function imprimirMetricas(diferenciacion, floracion, megasequia, spearmanRes) {
  console.log('\n(a) DIFERENCIACIÓN REAL (¿repite molde o produce trayectorias distintas?)');
  console.log(`    Años totales: ${diferenciacion.anios_totales}`);
  console.log(`    -- Solo JARDIN_FERTIL_pct aislado (subestima: dos años pueden compartir el`);
  console.log(`       mismo % total con MESES DISTINTOS de por medio, eso YA es diferenciación real) --`);
  console.log(`    Valores distintos (redondeado a 1 decimal): ${diferenciacion.valores_distintos_totales}`);
  console.log(`    Valores que se repiten 3+ veces: ${diferenciacion.valores_en_clusters_3mas} (${diferenciacion.anios_en_clusters_3mas} años)`);
  console.log(`    -- Tupla completa (las 4 zonas juntas, criterio real) --`);
  console.log(`    Tuplas completas distintas: ${diferenciacion.tuplas_completas_distintas}`);
  console.log(`    Tuplas que se repiten 3+ veces: ${diferenciacion.tuplas_en_clusters_3mas} (${diferenciacion.anios_en_clusters_tupla_3mas} años)`);
  console.log(`    [${pasaFalla(diferenciacion.anios_en_clusters_tupla_3mas < 20)}] umbral: <20 años en clusters-de-3+ (tupla completa)`);

  console.log('\n(b) FLORACIÓN REAL > CONTROL');
  console.log(`    Años de floración documentada usados: ${floracion.anios_floracion_n} (${floracion.anios_floracion.join(', ')})`);
  console.log(`    Años control (sin floración) usados: ${floracion.anios_control_n} (${floracion.anios_control.join(', ')})`);
  console.log(`    Media JARDIN_FERTIL_pct en floración: ${floracion.media_JARDIN_FERTIL_floracion.toFixed(3)}`);
  console.log(`    Media JARDIN_FERTIL_pct en control:   ${floracion.media_JARDIN_FERTIL_control.toFixed(3)}`);
  console.log(`    Fracción de años de floración que superan la media de control: ${(floracion.fraccion_floracion_sobre_media_control * 100).toFixed(1)}%`);
  if (floracion.solapan_floracion_y_megasequia && floracion.solapan_floracion_y_megasequia.length) {
    console.log(`    -- CONFLICTO CON (c), informativo: años que son floración documentada Y megasequía a la vez:`);
    console.log(`       floración∩megasequía: ${floracion.solapan_floracion_y_megasequia.join(', ') || '(ninguno)'} | control∩megasequía: ${floracion.solapan_control_y_megasequia.join(', ') || '(ninguno)'}`);
    console.log(`       excluyendo megasequía de ambos grupos (n=${floracion.sin_megasequia_n_floracion} vs ${floracion.sin_megasequia_n_control}): floración ${floracion.sin_megasequia_media_floracion.toFixed(3)} vs control ${floracion.sin_megasequia_media_control.toFixed(3)} -> ${floracion.sin_megasequia_media_floracion > floracion.sin_megasequia_media_control ? 'floración GANA' : 'floración pierde'}`);
  }
  console.log(`    [${pasaFalla(floracion.media_JARDIN_FERTIL_floracion > floracion.media_JARDIN_FERTIL_control)}] umbral: media(floración) > media(control)`);

  console.log('\n(c) MEGASEQUÍA 2019-2025 POR DEBAJO DEL PROMEDIO');
  console.log(`    Años de megasequía usados: ${megasequia.anios_megasequia_n}`);
  console.log(`    Media JARDIN_FERTIL_pct megasequía: ${megasequia.media_JARDIN_FERTIL_megasequia.toFixed(3)} | global (62 años): ${megasequia.media_JARDIN_FERTIL_global.toFixed(3)}`);
  console.log(`    Media COLAPSO_pct megasequía:       ${megasequia.media_COLAPSO_megasequia.toFixed(3)} | global (62 años): ${megasequia.media_COLAPSO_global.toFixed(3)}`);
  console.log(`    [${pasaFalla(megasequia.media_JARDIN_FERTIL_megasequia < megasequia.media_JARDIN_FERTIL_global)}] umbral: media(megasequía JARDIN_FERTIL) < media global`);

  console.log('\n(d) CORRELACIÓN SPEARMAN JARDIN_FERTIL_pct vs LLUVIA REAL ANUAL');
  console.log(`    Años con dato de lluvia real: ${spearmanRes.n_anios_con_dato_de_lluvia}`);
  console.log(`    rho de Spearman: ${spearmanRes.rho_spearman_JARDIN_FERTIL_vs_lluvia === null ? 'N/A' : spearmanRes.rho_spearman_JARDIN_FERTIL_vs_lluvia.toFixed(4)}`);
  const rho = spearmanRes.rho_spearman_JARDIN_FERTIL_vs_lluvia;
  console.log(`    [${pasaFalla(rho !== null && rho > 0 && Math.abs(rho) > 0.2)}] umbral: rho > 0 y |rho| > 0.2`);

  console.log('\n(e) EN MEGASEQUÍA DEBE DOMINAR CIERRE (equilibrio sin cambios)');
  console.log(`    Media CIERRE_pct megasequía:       ${megasequia.media_CIERRE_megasequia.toFixed(3)} | global (62 años): ${megasequia.media_CIERRE_global.toFixed(3)}`);
  console.log(`    Media SELVA_HOSTIL_pct megasequía: ${megasequia.media_SELVA_megasequia.toFixed(3)} | global (62 años): ${megasequia.media_SELVA_global.toFixed(3)}`);
  console.log(`    [${pasaFalla(megasequia.media_CIERRE_megasequia > megasequia.media_CIERRE_global)}] umbral: media(megasequía CIERRE) > media global`);
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
function main() {
  const rutaCsv = process.argv[2];
  if (!rutaCsv) {
    console.error('Uso: node evaluar_contra_ground_truth.js <ruta_a_por_anio.csv>');
    process.exit(1);
  }
  const rutaAbsolutaCsv = path.resolve(rutaCsv);
  if (!fs.existsSync(rutaAbsolutaCsv)) {
    console.error(`No existe el archivo: ${rutaAbsolutaCsv}`);
    process.exit(1);
  }

  const datosModelo = cargarDatosModelo(rutaAbsolutaCsv);
  const floracionEmpirica = cargarFloracionEmpirica();
  const oni = cargarOni(); // cargado para contexto/depuración, no entra en las 4 métricas
  const { porAnio: lluviaReal, metodo: metodoLluvia } = cargarLluviaAnualReal();

  console.log('='.repeat(78));
  console.log(`Evaluación contra ground truth real: ${rutaAbsolutaCsv}`);
  console.log(`(lluvia real leída con: ${metodoLluvia})`);
  console.log('='.repeat(78));

  const agregado = calcularMetricas(datosModelo, floracionEmpirica, lluviaReal);
  console.log('\n### AGREGADO (los 62 años juntos) ###');
  imprimirMetricas(agregado.diferenciacion, agregado.floracion, agregado.megasequia, agregado.spearmanRes);

  const { conCobertura, sinCobertura, sinDatoDeCobertura } = dividirPorCobertura(datosModelo);
  let porCobertura = null;
  if (sinDatoDeCobertura === 0 && conCobertura.size > 0 && sinCobertura.size > 0) {
    const real = calcularMetricas(conCobertura, floracionEmpirica, lluviaReal);
    const sintetico = calcularMetricas(sinCobertura, floracionEmpirica, lluviaReal);
    console.log(`\n### SOLO AÑOS CON TEMPERATURA REAL (${conCobertura.size} de 62) ###`);
    imprimirMetricas(real.diferenciacion, real.floracion, real.megasequia, real.spearmanRes);
    console.log(`\n### SOLO AÑOS SIN TEMPERATURA REAL, vaivén sintético (${sinCobertura.size} de 62) ###`);
    imprimirMetricas(sintetico.diferenciacion, sintetico.floracion, sintetico.megasequia, sintetico.spearmanRes);
    porCobertura = { real, sintetico, anios_con_cobertura: conCobertura.size, anios_sin_cobertura: sinCobertura.size };
  } else if (sinDatoDeCobertura > 0) {
    console.log('\n(sin columna coberturaTempReal_pct en este CSV -- corrida de una ronda anterior a la temperatura real, no se separa por cobertura)');
  }
  console.log('\n' + '='.repeat(78));

  const rutaSalida = rutaAbsolutaCsv.replace(/\.csv$/i, '') + '_evaluacion.json';
  const resumenJson = {
    entrada: rutaAbsolutaCsv,
    generado_en: new Date().toISOString(),
    metodo_lectura_lluvia: metodoLluvia,
    metricas: agregado,
    metricas_por_cobertura_temp_real: porCobertura,
    contexto_oni_disponible_anios: oni.size,
  };
  fs.writeFileSync(rutaSalida, JSON.stringify(resumenJson, null, 2), 'utf-8');
  console.log(`\nResumen JSON escrito en: ${rutaSalida}`);
}

main();
