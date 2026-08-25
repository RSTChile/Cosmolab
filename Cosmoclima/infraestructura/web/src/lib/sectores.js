/**
 * LA LÓGICA INVERSA · del sector hacia el territorio
 * ===================================================
 *
 * La primera pestaña recorre la cadena territorio → clima → infraestructura:
 * «¿qué hay en mi comuna y qué le viene?». Ésta la recorre al revés:
 * **sector → sus activos → sus celdas → pronóstico**, o sea «¿qué parte de este
 * sector queda expuesta?».
 *
 * ★★ AFECTADO Y EXPUESTO NO SON LO MISMO
 * ----------------------------------------
 * Sólo 6 de los 27 ítems con activos tienen umbral medido — los viales e
 * hídricos, porque el temporal de julio sólo dejó registro de vías del MOP. Para
 * ésos se puede decir **afectado**: la lluvia pronosticada supera el umbral con
 * que ese mismo tipo de elemento cedió en la realidad.
 *
 * Para los otros 21 —torres, escuelas, hospitales, subestaciones— sólo se puede
 * decir **expuesto**: va a llover mucho encima, y nadie ha medido nunca qué le
 * pasa a una torre con esa lluvia. Se cuentan por separado y se muestran por
 * separado. Sumarlos daría un número más grande y más impresionante que estaría
 * mintiendo.
 */

/**
 * El criterio compartido. Si llega `evaluar` desde la aplicación, manda ése —es
 * el que usa el umbral LOCAL, medido para cada celda—. La rama de abajo es el
 * respaldo de cuando el umbral local todavía no ha terminado de cargar.
 */
function clasifica(mm, n, cut, af, evaluar) {
  if (mm == null) return 'sin cobertura';
  if (evaluar) return evaluar(mm, n, cut).estado;
  const a = af[n];
  if (a?.tipo === 'medido') return mm >= a.umbral_mm_72h ? 'afectado' : 'bajo umbral';
  return mm >= 50 ? 'expuesto' : 'sin señal';
}

/** Los 20 sectores con sus ítems, activos y cuánto de eso está en riesgo hoy. */
export function resumenPorSector(datos, mmPorComuna, evaluar = null) {
  const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
  const af = datos.afectacion?.por_item ?? {};
  const sectores = new Map();

  for (const item of datos.matriz.items) {
    const s = item.sector ?? '(sin sector)';
    if (!sectores.has(s)) {
      sectores.set(s, {
        sector: s, items: 0, itemsConActivos: 0, activos: 0,
        afectados: 0, expuestos: 0, itemsMedidos: 0,
      });
    }
    sectores.get(s).items++;
  }

  for (const [cut, idx] of Object.entries(datos.activos.por_comuna ?? {})) {
    const mm = mmPorComuna.get(cut);
    for (const [n, cantidad] of Object.entries(idx)) {
      const item = porN.get(n);
      if (!item) continue;
      const r = sectores.get(item.sector ?? '(sin sector)');
      r.activos += cantidad;
      const est = clasifica(mm, n, cut, af, evaluar);
      if (est === 'afectado') r.afectados += cantidad;
      else if (est === 'expuesto') r.expuestos += cantidad;
    }
  }

  // cuántos ítems distintos tienen activos, y cuántos de ésos son medibles
  const vistos = new Map();
  for (const idx of Object.values(datos.activos.por_comuna ?? {})) {
    for (const n of Object.keys(idx)) {
      const item = porN.get(n);
      if (item) vistos.set(n, item.sector ?? '(sin sector)');
    }
  }
  for (const [n, s] of vistos) {
    const r = sectores.get(s);
    r.itemsConActivos++;
    if (af[n]?.tipo === 'medido') r.itemsMedidos++;
  }

  return [...sectores.values()].sort(
    (a, b) => b.afectados + b.expuestos - (a.afectados + a.expuestos) || b.activos - a.activos,
  );
}

/**
 * Los ítems de un sector, con lo que le pasa a cada uno.
 *
 * ★★ `soloCut` NO ES UN DETALLE: ES LA PREGUNTA.
 * Sin él, esta función siempre devolvía el total NACIONAL. Con Tarapacá
 * seleccionada, el panel mostraba «Puentes de Carreteras 1.473 de 6.733» —el
 * país entero— y nunca los 13 que hay ahí. Quien miraba concluía, con razón,
 * que la región no tenía nada de este sector.
 */
export function itemsDeSector(datos, sector, mmPorComuna, evaluar = null,
                              soloCut = null) {
  const af = datos.afectacion?.por_item ?? {};
  const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
  const acum = new Map();

  for (const [cut, idx] of Object.entries(datos.activos.por_comuna ?? {})) {
    if (soloCut && cut !== soloCut) continue;
    const mm = mmPorComuna.get(cut);
    for (const [n, cantidad] of Object.entries(idx)) {
      const item = porN.get(n);
      if (!item || item.sector !== sector) continue;
      if (!acum.has(n)) {
        acum.set(n, { n, item, activos: 0, enRiesgo: 0, comunas: 0, af: af[n] ?? null });
      }
      const r = acum.get(n);
      r.activos += cantidad;
      const est = clasifica(mm, n, cut, af, evaluar);
      if (est === 'afectado' || est === 'expuesto') {
        r.enRiesgo += cantidad;
        r.comunas++;
      }
    }
  }
  return [...acum.values()].sort((a, b) => b.enRiesgo - a.enRiesgo || b.activos - a.activos);
}

/**
 * Proporción de activos de un sector en riesgo, comuna por comuna. Es lo que
 * pinta el mapa en esta pestaña.
 *
 * ★ Proporción y no cantidad absoluta: si se pintara la cantidad, Santiago
 * saldría rojo siempre y el resto del país invisible, porque ahí está
 * concentrada la infraestructura. Lo que interesa es qué FRACCIÓN del sector
 * queda comprometida en cada comuna.
 */
export function fraccionPorComuna(datos, sector, mmPorComuna, evaluar = null) {
  const af = datos.afectacion?.por_item ?? {};
  const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
  const out = new Map();

  for (const [cut, idx] of Object.entries(datos.activos.por_comuna ?? {})) {
    const mm = mmPorComuna.get(cut);
    let total = 0;
    let riesgo = 0;
    for (const [n, cantidad] of Object.entries(idx)) {
      const item = porN.get(n);
      if (!item || item.sector !== sector) continue;
      total += cantidad;
      const est = clasifica(mm, n, cut, af, evaluar);
      if (est === 'afectado' || est === 'expuesto') riesgo += cantidad;
    }
    if (total) out.set(cut, { total, riesgo, fraccion: riesgo / total });
  }
  return out;
}

/** Los ítems del sector que están en riesgo en UNA comuna — para filtrar puntos. */
export function itemsEnRiesgo(datos, sector, cut, mm) {
  const af = datos.afectacion?.por_item ?? {};
  const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
  const idx = datos.activos.por_comuna?.[cut] ?? {};
  const out = new Set();
  if (mm == null) return out;
  for (const n of Object.keys(idx)) {
    const item = porN.get(n);
    if (!item || (sector && item.sector !== sector)) continue;
    const a = af[n];
    if (a?.tipo === 'medido' ? mm >= a.umbral_mm_72h : mm >= 50) out.add(n);
  }
  return out;
}

/**
 * Escala de color para la fracción comprometida.
 *
 * ★ LOS DOS PRIMEROS CASOS SON DISTINTOS Y TIENEN QUE VERSE DISTINTOS.
 *   `null` = la comuna no tiene NI UN activo de este sector — no hay nada que
 *            decir, y se pinta casi como el fondo.
 *   `0`    = SÍ tiene activos y ninguno cruza umbral con esta lluvia. Eso es
 *            información, no ausencia.
 *   Antes eran #3f3f46 y #1e3a5f: dos grises oscurísimos indistinguibles sobre
 *   el mapa negro, así que «no le pasa nada» y «aquí no hay nada» se veían
 *   igual. El norte, que casi nunca cruza umbral, parecía vacío de
 *   infraestructura.
 */
export function colorFraccion(f) {
  if (f == null) return '#27272a';
  if (f >= 0.75) return '#9f1239';
  if (f >= 0.5) return '#c2410c';
  if (f >= 0.25) return '#a16207';
  if (f > 0) return '#166e5c';
  return '#3b6ea5';
}

/**
 * UN COLOR POR SECTOR
 * ====================
 *
 * Pintar todos los activos del mismo color deja el mapa mudo cuando hay varias
 * categorías a la vista: se ve «hay muchas cosas» y no «hay hospitales y
 * escuelas». Los tonos están elegidos para distinguirse sobre el mapa oscuro y
 * para no chocar con la escala de riesgo de las comunas, que es la del rojo al
 * azul — por eso aquí dominan los tonos medios y saturados.
 */
export const COLOR_SECTOR = {
  'Transporte': '#f97316',
  'Hídrico': '#38bdf8',
  'Telecomunicaciones': '#a78bfa',
  'Educación': '#facc15',
  'Energía': '#fb7185',
  'Salud': '#4ade80',
  'Represas': '#22d3ee',
  'Protección Social': '#f472b6',
  'Servicios de Emergencia': '#ef4444',
  'Comercial': '#c084fc',
  'Seguridad': '#60a5fa',
  'Gobierno': '#94a3b8',
  'Nuclear': '#fde047',
  'Comunicaciones': '#818cf8',
  'Alimentario': '#86efac',
  'Financiero': '#67e8f9',
  'Químico': '#fdba74',
  'Industrial': '#cbd5e1',
  'Industria de Defensa': '#f87171',
  'Tecnologías Informáticas': '#d8b4fe',
};

export function colorSector(s) {
  return COLOR_SECTOR[s] ?? '#e5e7eb';
}
