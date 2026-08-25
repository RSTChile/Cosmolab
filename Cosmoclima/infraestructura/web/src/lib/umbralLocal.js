/**
 * EL UMBRAL LOCAL · lo que en Arica es desastre y en Valdivia es martes
 * ======================================================================
 *
 * ★★ POR QUÉ EL MILÍMETRO NACIONAL NO SIRVE SOLO
 * -------------------------------------------------
 * Medido sobre los 1.241 tramos cortados en julio de 2026, cada uno contra los
 * 36 años de su propia celda:
 *
 *     el milímetro que corta varía  4,1 veces entre zonas (43 mm a 174 mm)
 *     el percentil local se mueve   0,46 puntos (99,54 % a 100 %)
 *
 * Todos los cortes del país ocurrieron por encima del percentil 99,5 de su
 * propio lugar. **La rareza local viaja; el milímetro es una coincidencia
 * regional.** En Arica llueve 1 mm al año: diez milímetros en un día son un
 * desastre allí y un martes en Valdivia.
 *
 * Así que el umbral de cada elemento se traduce al percentil que ocupa donde
 * fue medido, y ese percentil se lee de vuelta en milímetros en cada celda.
 */

let cache = null;
let enCurso = null;

export async function cargarUmbralLocal() {
  if (cache) return cache;
  if (!enCurso) {
    enCurso = fetch('datos/umbral_celda.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => (cache = d));
  }
  return enCurso;
}

/**
 * El umbral de un ítem EN UNA CELDA concreta, en milímetros.
 *
 * Devuelve `null` cuando la celda no tiene episodios que permitan calcular
 * percentiles altos — declararlo es más honesto que devolver cero, que dejaría
 * todo «afectado» para siempre en el desierto absoluto.
 */
export function umbralEnCelda(item, celda, ul) {
  if (!ul) return null;
  const p = ul.percentil_por_item?.[String(item)];
  const fila = ul.por_celda?.[celda];
  if (p == null || !fila) return null;
  // el corte guardado más cercano por debajo del percentil pedido
  const cortes = Object.keys(fila).map(Number).sort((a, b) => a - b);
  let mejor = cortes[0];
  for (const c of cortes) if (c <= p + 1e-9) mejor = c;
  return fila[mejor.toFixed(3)] ?? null;
}

/**
 * Decide si un activo está afectado, prefiriendo el umbral local y cayendo al
 * nacional cuando la celda no tiene uno. Devuelve también CUÁL se usó, porque
 * la aplicación tiene que poder decirlo: no es lo mismo «supera lo que rompe
 * una carretera aquí» que «supera el promedio del país».
 */
export function evaluar(mm, item, celda, ul, umbralNacional) {
  if (mm == null) return { estado: 'sin cobertura', umbral: null, escala: null };
  const local = umbralEnCelda(item, celda, ul);
  const u = local ?? umbralNacional ?? null;
  if (u == null) {
    return {
      estado: mm >= 50 ? 'expuesto' : 'sin señal',
      umbral: null,
      escala: null,
    };
  }
  return {
    estado: mm >= u ? 'afectado' : 'bajo umbral',
    umbral: u,
    escala: local != null ? 'local' : 'nacional',
    nacional: umbralNacional ?? null,
  };
}
