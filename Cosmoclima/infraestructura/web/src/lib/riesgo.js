/**
 * DE MILÍMETROS A CONSECUENCIAS
 * ==============================
 *
 * La pregunta que originó todo esto: «si caen tantos mm en tanto tiempo, ¿qué se
 * rompe?». Aquí está la aritmética que la contesta, y las decisiones que la
 * hacen honesta.
 *
 * ★ POR QUÉ 72 HORAS Y NO UN DÍA
 * --------------------------------
 * Porque es la ventana con la que se midió la tasa de corte contra el temporal
 * de julio 2026, y un umbral sólo significa algo en la ventana en que fue
 * medido. Además el daño rara vez lo hace la lluvia de un día suelto: el 25 de
 * julio en Coquimbo llegó sobre tres días de agua acumulada.
 *
 * ★ POR QUÉ EL MÁXIMO Y NO EL PROMEDIO ENTRE CELDAS
 * ---------------------------------------------------
 * Una comuna grande abarca varias celdas. Promediarlas diluye justamente el
 * sector que se va a cortar: si en un extremo caen 120 mm y en el otro 10, el
 * promedio dice 65 y nadie se entera de que hay un sector en zona roja. Para
 * decidir si hay que preocuparse manda el peor punto del territorio.
 *
 * ⚠️ LO QUE ESTA CIFRA NO ES
 * ---------------------------
 * No es una predicción de que se va a cortar tal calle. Es la frecuencia con que
 * SE CORTÓ ALGUNA vía cuando llovió así, medida sobre 18 días de un temporal en
 * seis regiones. Un 22 % significa «en 1 de cada 5 días-celda con esta lluvia
 * hubo al menos un corte», no «esta comuna tiene 22 % de probabilidad».
 */

/** Los primeros días tienen ventana incompleta: se acumula lo disponible. */
export function acumulados72h(serie) {
  return serie.map((_, i) => {
    const desde = Math.max(0, i - 2);
    return serie.slice(desde, i + 1).reduce((a, b) => a + b, 0);
  });
}

/** La serie de la comuna: el máximo entre sus celdas, día a día. */
export function serieDeComuna(cut, celdasPorComuna, pronostico) {
  const info = celdasPorComuna.por_comuna?.[cut];
  if (!info || !info.celdas?.length) return null;
  const series = info.celdas.map((k) => pronostico.celdas[k]).filter(Boolean);
  if (!series.length) return null;
  const n = pronostico.dias;
  const serie = Array.from({ length: n }, (_, i) =>
    Math.max(...series.map((s) => s[i] ?? 0)),
  );
  return { serie, celdas: info.celdas, modo: info.modo };
}

/** La franja de la tabla con denominador en la que cae un acumulado. */
export function franjaDe(mm, umbrales) {
  const serie = umbrales.serie_para_pronostico ?? 'openmeteo';
  const t = umbrales.tasa_de_corte?.[serie];
  if (!t) return null;
  for (const f of t.franjas) {
    if (mm >= f.desde && (f.hasta == null || mm < f.hasta)) return f;
  }
  return t.franjas[t.franjas.length - 1] ?? null;
}

/**
 * El nivel que se pinta en el mapa. Los cortes son los que la medición mostró:
 * el salto real está en los 50 mm y se dispara sobre los 100. No son quintiles
 * repartidos por estética — moverlos desalinearía el color de la evidencia.
 */
export function nivelDe(mm) {
  if (mm == null) return { clave: 'sincobertura', etiqueta: 'sin cobertura', orden: -1 };
  if (mm >= 100) return { clave: 'muyalto', etiqueta: 'muy alto', orden: 4 };
  if (mm >= 50) return { clave: 'alto', etiqueta: 'alto', orden: 3 };
  if (mm >= 25) return { clave: 'medio', etiqueta: 'medio', orden: 2 };
  if (mm >= 10) return { clave: 'bajo', etiqueta: 'bajo', orden: 1 };
  return { clave: 'minimo', etiqueta: 'mínimo', orden: 0 };
}

export const COLORES = {
  sincobertura: '#3f3f46',
  minimo: '#1e3a5f',
  bajo: '#166e5c',
  medio: '#a16207',
  alto: '#c2410c',
  muyalto: '#9f1239',
};

/** El peor momento de los próximos días, que es lo que hay que mirar. */
export function peorVentana(serie) {
  const ac = acumulados72h(serie);
  let mejor = 0;
  for (let i = 1; i < ac.length; i++) if (ac[i] > ac[mejor]) mejor = i;
  return { indice: mejor, mm: ac[mejor], acumulados: ac };
}

/**
 * Qué elementos ceden con esa lluvia: los que en el temporal se cortaron con una
 * mediana IGUAL O MENOR. Se excluyen los que se midieron con menos de 10 tramos
 * — con 6 casos la mediana es casi anecdótica y aparecería como si fuera un
 * hallazgo firme.
 */
const NO_ES_ELEMENTO = new Set(['sin dato', 'Otro Elemento']);

export function elementosQueCeden(mm, umbrales) {
  return (umbrales.por_elemento ?? [])
    .filter(
      (e) =>
        // ⚠️ «sin dato» y «Otro Elemento» son huecos del catastro del MOP, no
        //    tipos de infraestructura. Mostrarlos como algo que «cede» sería
        //    presentar la ausencia de registro como si fuera un hallazgo.
        !NO_ES_ELEMENTO.has(e.elemento) &&
        (e.tramos ?? 0) >= 10 &&
        e.mm_72h_mediana <= mm,
    )
    .sort((a, b) => b.mm_72h_mediana - a.mm_72h_mediana);
}

/**
 * VIGENCIA DEL PRONÓSTICO
 * ========================
 *
 * ★★ Un pronóstico envejece y la aplicación no puede fingir que no.
 *
 * La ventana se baja una vez y empieza el día en que se generó, así que a la
 * mañana siguiente su primer día YA PASÓ. Con horas de antigüedad da igual; con
 * cinco días, alguien podría mirar el mapa en pleno temporal y estar viendo una
 * previsión vencida sin una sola señal de que lo está. Y el «día más lluvioso»
 * que abre la vista podría caer en un día que ya ocurrió.
 *
 * Devuelve el índice del primer día que todavía no ha terminado.
 */
export function primerDiaVigente(pronostico, hoy = new Date()) {
  const clave = [
    hoy.getFullYear(),
    String(hoy.getMonth() + 1).padStart(2, '0'),
    String(hoy.getDate()).padStart(2, '0'),
  ].join('-');
  const i = (pronostico.fechas ?? []).findIndex((f) => f >= clave);
  return i < 0 ? Math.max(0, (pronostico.fechas?.length ?? 1) - 1) : i;
}

/** Horas transcurridas desde que se generó el pronóstico. */
export function edadPronostico(pronostico, ahora = new Date()) {
  if (!pronostico?.generado) return null;
  const t = new Date(pronostico.generado);
  if (Number.isNaN(+t)) return null;
  return (ahora - t) / 36e5;
}

/**
 * El día que conviene mostrar al abrir: el de mayor lluvia acumulada del país.
 *
 * ★ Arrancar en el día 0 sería lo obvio y dejaría la aplicación pareciendo rota
 * la mayor parte del año: hoy, por ejemplo, las 328 comunas con cobertura caen
 * todas en «mínimo» y el mapa se ve de un solo color. Además el primer día tiene
 * la ventana de 72 h incompleta —sólo se ha acumulado un día—, así que subestima
 * por construcción. Lo que alguien viene a ver es cuándo se pone feo.
 */
export function diaMasLluvioso(comunas, celdasPorComuna, pronostico) {
  const total = new Array(pronostico.dias).fill(0);
  for (const c of comunas) {
    const s = serieDeComuna(c.cut, celdasPorComuna, pronostico);
    if (!s) continue;
    const ac = acumulados72h(s.serie);
    for (let i = 0; i < total.length; i++) total[i] += ac[i];
  }
  // ★ Sólo entre los días VIGENTES: abrir en un día ya ocurrido presentaría el
  //   pasado como previsión, que es peor que no abrir en ninguno.
  const desde = primerDiaVigente(pronostico);
  let mejor = desde;
  for (let i = desde + 1; i < total.length; i++) if (total[i] > total[mejor]) mejor = i;
  return mejor;
}

/** Cuánto se parece esta lluvia a lo normal del lugar (la normal anual). */
export function contraNormal(mmAnualPronosticado, cut, celdasPorComuna, climatologia) {
  const info = celdasPorComuna.por_comuna?.[cut];
  if (!info?.celdas?.length) return null;
  const normales = info.celdas
    .map((k) => climatologia.normal_anual?.[k])
    .filter((v) => typeof v === 'number');
  if (!normales.length) return null;
  const normal = Math.max(...normales);
  if (!normal) return null;
  return { normal, fraccion: mmAnualPronosticado / normal };
}
