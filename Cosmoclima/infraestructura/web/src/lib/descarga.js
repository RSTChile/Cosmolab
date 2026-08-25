/**
 * DESCARGA POR COMUNA
 * ====================
 *
 * ★ Un alcalde mira la página; su equipo necesita el archivo. Sin esto, lo que
 *   se ve en pantalla no puede entrar a un informe, a una minuta del COGRID ni a
 *   una hoja de cálculo, y la herramienta se queda en la pantalla.
 *
 * Se arma en el navegador con lo que ya está cargado: no hace falta que el
 * servidor prepare nada.
 */

/** Escapa un campo para CSV: comillas dobladas y entrecomillado si hace falta. */
function campo(v) {
  const t = v == null ? '' : String(v);
  return /[",;\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
}

export function aCSV(filas, columnas) {
  const l = [columnas.join(';')];
  for (const f of filas) l.push(columnas.map((c) => campo(f[c])).join(';'));
  // ★ BOM al inicio: sin él, Excel en Windows abre los acentos rotos y el
  //   primer reflejo de quien recibe el archivo es que el dato está mal.
  return '\ufeff' + l.join('\r\n');
}

export function bajar(nombre, texto) {
  const b = new Blob([texto], { type: 'text/csv;charset=utf-8' });
  const u = URL.createObjectURL(b);
  const a = document.createElement('a');
  a.href = u;
  a.download = nombre;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}

/** Los activos de una comuna, con su ítem, umbral y antecedentes. */
export function filasDeComuna(detalle, datos, comuna, mm) {
  const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
  const af = datos.afectacion?.por_item ?? {};
  return (detalle ?? []).map((a) => {
    const item = porN.get(String(a.n));
    const x = af[String(a.n)];
    const umbral = x?.tipo === 'medido' ? x.umbral_mm_72h : null;
    return {
      comuna: comuna?.comuna ?? '',
      region: comuna?.region ?? '',
      item: a.n,
      elemento: item?.elemento ?? '',
      sector: item?.sector ?? '',
      irmd: item?.IRMD ?? '',
      activo: a.a || '(sin nombre en el catastro)',
      latitud: a.y,
      longitud: a.x,
      mm_72h_pronosticados: mm == null ? '' : mm.toFixed(1),
      umbral_mm_72h: umbral ?? '',
      situacion:
        mm == null ? 'sin cobertura'
        : umbral != null ? (mm >= umbral ? 'afectado' : 'bajo umbral')
        : mm >= 50 ? 'expuesto' : 'sin señal',
      antecedentes: (a.h ?? [])
        .map((h) =>
          h.t === 'pc' ? `punto crítico SENAPRED: ${h.c} (${h.r}) a ${h.d} m`
          : h.t === 'via' ? `vía cortada ${h.f} ${h.g} a ${h.d} m`
          : `${h.p} ${h.m} ${h.a} a ${h.d} m`,
        )
        .join(' | '),
    };
  });
}

export const COLUMNAS = [
  'comuna', 'region', 'item', 'elemento', 'sector', 'irmd', 'activo',
  'latitud', 'longitud', 'mm_72h_pronosticados', 'umbral_mm_72h', 'situacion',
  'antecedentes',
];
