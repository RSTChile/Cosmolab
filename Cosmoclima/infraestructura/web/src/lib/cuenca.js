/**
 * LA SEÑAL DE CUENCA · cuando el valle está seco y la cordillera no
 * ==================================================================
 *
 * Detectado por Alexis el 25-ago-2026: el tramo La Serena–Antofagasta de la
 * Ruta 5 no mostraba afectación pese a las lluvias anunciadas. Verificado contra
 * Open-Meteo en vivo:
 *
 *     Copiapó ciudad            0,0 mm en 16 días
 *     Tierra Amarilla           0,0 mm
 *     cordillera de Choapa     47,8 mm en 72 h
 *     cordillera de Elqui      34,9 mm
 *
 * **Sí llueve en el norte — en la cordillera.** El modelo evaluaba sólo la celda
 * del activo, así que miraba el lugar equivocado: el agua cae arriba y baja por
 * las quebradas hasta el camino del valle.
 *
 * ★★ POR QUÉ ESTO NO SE SUMA A «AFECTADO»
 * -----------------------------------------
 * La lluvia sobre la cuenca **no es lluvia sobre el activo**, y no existe
 * registro de «cuánta lluvia cordillerana corta este camino del valle». Se buscó
 * calibrarlo y no se pudo: en julio de 2026 llovió en todas partes a la vez (de
 * 691 cortes del norte, sólo 5 tuvieron poca lluvia local) y en el aluvión de
 * Copiapó de 2015 cayeron 94,5 mm en la ciudad además de 71,7 en la cordillera.
 * Ningún caso aísla el mecanismo.
 *
 * Por eso es una señal aparte, **declarada sin umbral**. Sumarla al conteo de
 * afectados daría un número mayor y sin respaldo.
 */

let cache = null;
let enCurso = null;

export async function cargarCuenca() {
  if (cache) return cache;
  if (!enCurso) {
    enCurso = fetch('datos/cuenca.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => (cache = d));
  }
  return enCurso;
}

/** Acumulado de 72 h de una celda en un día del pronóstico. */
function ac72(serie, dia) {
  if (!serie) return null;
  let t = 0;
  for (let k = Math.max(0, dia - 2); k <= dia; k++) t += serie[k] ?? 0;
  return t;
}

/**
 * La lluvia máxima sobre las celdas que drenan hacia ésta.
 *
 * Devuelve null cuando la celda no tiene cuenca calculada — sólo se calculó al
 * norte de la latitud −33, que es donde el valle puede estar seco mientras la
 * cordillera recibe el temporal. En el sur llueve en toda la cuenca a la vez y
 * la celda propia ya captura el evento.
 */
export function aguasArriba(celda, cuenca, pronostico, dia) {
  const arriba = cuenca?.por_celda?.[celda];
  if (!arriba?.length) return null;
  let peor = 0;
  let dondePeor = null;
  for (const c of arriba) {
    const v = ac72(pronostico.celdas?.[c], dia);
    if (v != null && v > peor) {
      peor = v;
      dondePeor = c;
    }
  }
  return { mm: peor, celdas: arriba.length, celdaPeor: dondePeor };
}

/**
 * ¿Vale la pena mostrarlo? Sólo cuando la cuenca aporta algo que la celda propia
 * no dice: bastante lluvia arriba y claramente más que abajo. Si ambas cifras se
 * parecen, la línea sobra y ensucia la pantalla.
 */
export function vale(mmLocal, arriba) {
  if (!arriba || arriba.mm < 20) return false;
  return arriba.mm >= Math.max(20, (mmLocal ?? 0) * 1.5);
}
