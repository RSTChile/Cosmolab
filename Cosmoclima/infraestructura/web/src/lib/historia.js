/**
 * LA HISTORIA CLIMÁTICA · carga diferida y selección de territorio
 * =================================================================
 *
 * `historia_mensual.json` pesa 2,82 MB. No se carga al entrar: sólo cuando
 * alguien abre el gráfico. Quien viene a mirar el mapa no debería pagar por una
 * serie de 36 años que quizá no va a abrir.
 *
 * ★ CUATRO NIVELES, UNA MISMA CURVA
 * -----------------------------------
 * Las claves llevan prefijo: `CL` el país, `R05` una región, `P051` una
 * provincia, `C05101` una comuna. El gráfico no distingue: recibe una serie y la
 * dibuja. Lo que cambia con el nivel es cuánta dispersión interna hay, y por eso
 * vienen tres estadísticos en vez de uno.
 */

let cache = null;
let enCurso = null;

export async function cargarHistoria() {
  if (cache) return cache;
  if (!enCurso) {
    enCurso = Promise.all([
      fetch('datos/historia_mensual.json').then((r) => (r.ok ? r.json() : null)),
      fetch('datos/enso.json').then((r) => (r.ok ? r.json() : null)),
    ]).then(([hist, enso]) => {
      cache = hist ? { ...hist, enso: enso?.bandas ?? [] } : null;
      return cache;
    });
  }
  return enCurso;
}

/** La serie diaria de una comuna, sólo cuando se pide el detalle fino. */
export async function cargarDiaria(cut) {
  const r = await fetch(`datos/historia_diaria/${cut}.json`);
  return r.ok ? r.json() : null;
}

/**
 * El árbol de territorios que alimenta el selector.
 *
 * ⚠️ Sólo entran los que EXISTEN en la historia. Tres comunas insulares tienen
 * pronóstico pero no historia —ERA5-Land no cubre las islas oceánicas pequeñas—
 * y ofrecerlas en el selector para después mostrar un gráfico vacío sería peor
 * que no ofrecerlas.
 */
export function arbolTerritorios(datos, hist) {
  if (!hist) return { pais: null, regiones: [], provincias: [], comunas: [] };
  const hay = (k) => Boolean(hist.territorios[k]);
  const t = datos.territorios;
  return {
    pais: hay('CL') ? { clave: 'CL', nombre: 'Chile' } : null,
    regiones: t.regiones
      .map((r) => ({ clave: `R${r.cut}`, nombre: r.nombre ?? r.region ?? r.cut, cut: r.cut }))
      .filter((r) => hay(r.clave)),
    provincias: t.provincias
      .map((p) => ({ clave: `P${p.cut}`, nombre: p.nombre, reg: p.cut_reg }))
      .filter((p) => hay(p.clave)),
    comunas: t.comunas
      .map((c) => ({ clave: `C${c.cut}`, nombre: c.comuna, prov: c.cut_prov, reg: c.cut_reg, cut: c.cut }))
      .filter((c) => hay(c.clave)),
  };
}

/** Suma por año, para la etiqueta de resumen. */
export function totalesAnuales(meses, valores) {
  const por = new Map();
  for (let i = 0; i < meses.length; i++) {
    const a = meses[i].slice(0, 4);
    por.set(a, (por.get(a) ?? 0) + valores[i]);
  }
  return por;
}

/**
 * Los umbrales que tiene sentido dibujar sobre el gráfico.
 *
 * ★ Son mm en 72 h y el gráfico mensual muestra mm en un mes: no son
 * comparables sin más, y por eso la línea se dibuja SÓLO en la vista diaria.
 * Ponerla sobre la mensual daría la impresión de que un mes bajo la línea está a
 * salvo, cuando el daño lo hacen tres días seguidos dentro de ese mes.
 */
export function umbralesDibujables(datos) {
  const af = datos.afectacion?.por_item ?? {};
  const vistos = new Map();
  for (const [n, v] of Object.entries(af)) {
    if (v.tipo === 'medido' && v.umbral_mm_72h) {
      vistos.set(v.elemento ?? n, v.umbral_mm_72h);
    }
  }
  return [...vistos.entries()]
    .map(([elemento, mm]) => ({ elemento, mm }))
    .sort((a, b) => a.mm - b.mm);
}
