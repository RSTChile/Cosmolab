/**
 * TOPOJSON → GEOJSON, lo mínimo indispensable
 * ============================================
 *
 * `comunas.topo.json` viene cuantizado (1,4 MB en vez de los 103 MB de la capa
 * original). Descomprimirlo son ~60 líneas, y por eso no se agrega la
 * dependencia `topojson-client`: una librería entera para dos funciones que no
 * van a cambiar nunca.
 *
 * El formato es sencillo de leer una vez que se conocen las dos ideas:
 *
 *   · las coordenadas están en enteros y hay que aplicarles `transform`
 *     (escala + traslación) para volver a grados;
 *   · los enteros son DELTAS respecto del punto anterior, no posiciones
 *     absolutas, así que hay que ir sumando;
 *   · un arco negativo significa «este mismo arco, recorrido al revés», y la
 *     fórmula del índice es `~i` (o sea `-i - 1`). Ese detalle es el que más se
 *     equivoca al implementarlo a mano: sin él, las comunas vecinas comparten
 *     frontera pero dibujada en sentido contrario y el polígono sale roto.
 */

function descuantizar(arco, transform) {
  const [sx, sy] = transform.scale;
  const [tx, ty] = transform.translate;
  let x = 0;
  let y = 0;
  return arco.map(([dx, dy]) => {
    x += dx;
    y += dy;
    return [x * sx + tx, y * sy + ty];
  });
}

function unirArcos(indices, arcos) {
  const puntos = [];
  for (const i of indices) {
    // ★ Índice negativo = arco invertido. `~i` es `-i - 1`.
    const arco = i < 0 ? arcos[~i].slice().reverse() : arcos[i];
    // El último punto de un arco coincide con el primero del siguiente: se
    // descarta para no duplicar vértices en la unión.
    puntos.push(...(puntos.length ? arco.slice(1) : arco));
  }
  return puntos;
}

/** Convierte un objeto de la topología en una FeatureCollection de GeoJSON. */
export function aGeoJSON(topo, nombreObjeto) {
  const arcos = topo.transform
    ? topo.arcs.map((a) => descuantizar(a, topo.transform))
    : topo.arcs;

  const convertir = (g) => {
    if (g.type === 'Polygon') {
      return { type: 'Polygon', coordinates: g.arcs.map((anillo) => unirArcos(anillo, arcos)) };
    }
    if (g.type === 'MultiPolygon') {
      return {
        type: 'MultiPolygon',
        coordinates: g.arcs.map((poli) => poli.map((anillo) => unirArcos(anillo, arcos))),
      };
    }
    return null;
  };

  const features = [];
  for (const g of topo.objects[nombreObjeto].geometries) {
    const geometry = convertir(g);
    if (geometry) features.push({ type: 'Feature', properties: g.properties ?? {}, geometry });
  }
  return { type: 'FeatureCollection', features };
}
