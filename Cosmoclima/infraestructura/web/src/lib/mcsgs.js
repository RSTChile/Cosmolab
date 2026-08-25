/**
 * MCSGS · sincronización, que es el factor que la Matriz no mira
 * ===============================================================
 *
 * La Matriz evalúa cada activo por separado. El MCSGS dice que eso pierde lo
 * esencial: **la relación entre daño y pérdida funcional no es lineal**. Que
 * fallen diez nodos repartidos en el tiempo no es lo mismo que fallen los diez
 * el mismo día, aunque el daño total sea idéntico.
 *
 *     «Los sistemas no colapsan cuando son destruidos.
 *      Colapsan cuando dejan de poder moverse.»
 *
 * De los cinco factores del ICSGS, aquí se calculan los dos que tienen dato:
 *
 *     FCN  criticidad nodal   — viene precalculado desde PF e IRMD
 *     FSS  sincronización     — se calcula aquí, con el pronóstico
 *
 * ⚠️ NO se compone el ICSGS. Con FAS, FRC y FPI sin medir, el índice completo
 * sería un número con apariencia de medición. Se muestran los dos factores
 * medidos y se dice qué falta.
 */

let cache = null;
let enCurso = null;

export async function cargarMCSGS() {
  if (cache) return cache;
  if (!enCurso) {
    enCurso = fetch('datos/mcsgs.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => (cache = d));
  }
  return enCurso;
}

/**
 * FSS · qué fracción de los nodos de flujo cruza su umbral el mismo día,
 * ponderada por su criticidad nodal.
 *
 * ★ Ponderada y no simple: que se corte una carretera principal (FCN 0,744) y
 * que se corte un paso fronterizo menor (0,561) no pesan igual en la capacidad
 * del sistema para seguir moviéndose. Contar nodos a secas trataría ambos como
 * una unidad.
 */
export function sincronizacion(datos, mcsgs, mmPorComuna, evaluar, comunas) {
  if (!mcsgs) return null;
  const porItem = mcsgs.por_item ?? {};
  const flujo = Object.entries(porItem).filter(([, v]) => v.nodo === 'flujo' && v.con_activos);

  let pesoTotal = 0;
  let pesoAfectado = 0;
  const afectados = [];

  for (const [n, v] of flujo) {
    // ¿cuántos activos de este ítem hay, y cuántos cruzan hoy?
    let total = 0;
    let cruzan = 0;
    for (const c of comunas) {
      const k = datos.activos.por_comuna?.[c.cut]?.[n] ?? 0;
      if (!k) continue;
      total += k;
      const mm = mmPorComuna.get(c.cut);
      if (mm == null) continue;
      const e = evaluar ? evaluar(mm, n, c.cut) : null;
      if (e?.estado === 'afectado' || e?.estado === 'expuesto') cruzan += k;
    }
    if (!total) continue;
    const fraccion = cruzan / total;
    pesoTotal += v.fcn;
    pesoAfectado += v.fcn * fraccion;
    if (cruzan) {
      afectados.push({ n, ...v, total, cruzan, fraccion });
    }
  }

  if (!pesoTotal) return null;
  afectados.sort((a, b) => b.fcn * b.fraccion - a.fcn * a.fraccion);
  return {
    fss: pesoAfectado / pesoTotal,
    nodosConAfectacion: afectados.length,
    nodosTotales: flujo.length,
    afectados,
  };
}

/** Lectura en palabras. Deliberadamente sin número compuesto. */
export function leerFSS(fss) {
  if (fss == null) return null;
  if (fss >= 0.5) return { texto: 'sincronización alta', clave: 'alta' };
  if (fss >= 0.25) return { texto: 'sincronización media', clave: 'media' };
  if (fss > 0.02) return { texto: 'sincronización baja', clave: 'baja' };
  return { texto: 'sin sincronización', clave: 'nula' };
}
