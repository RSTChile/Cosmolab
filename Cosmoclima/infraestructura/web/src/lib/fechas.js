/**
 * FECHAS EN FORMATO CHILENO
 * ==========================
 *
 * Todo lo que se muestra en pantalla va en **DD-MM-YY**, que es como se lee una
 * fecha en Chile. Los datos se guardan en ISO (`2026-08-30`) porque así ordenan
 * bien y no dependen de la configuración de nadie; la conversión ocurre sólo al
 * momento de mostrar.
 *
 * ⚠️ Se parte la cadena a mano en vez de usar `new Date(iso)`: esa forma
 * interpreta la fecha como UTC y, en un huso al oeste de Greenwich como el
 * nuestro, devuelve **el día anterior**. Es un error clásico y silencioso —
 * todas las fechas aparecerían corridas un día sin que nada falle.
 */

/** `2026-08-30` → `30-08-26` */
export function fechaCorta(iso) {
  if (!iso) return '—';
  const [a, m, d] = String(iso).slice(0, 10).split('-');
  return `${d}-${m}-${a.slice(2)}`;
}

/** `2026-08-30` → `30-08` (para ejes y etiquetas apretadas) */
export function diaMes(iso) {
  if (!iso) return '—';
  const [, m, d] = String(iso).slice(0, 10).split('-');
  return `${d}-${m}`;
}

/** Rangos que vienen como texto: `1990-01-01 a 2026-08-21`. */
export function rangoCorto(texto) {
  if (!texto) return '—';
  return String(texto).replace(/(\d{4})-(\d{2})-(\d{2})/g, (_, a, m, d) => `${d}-${m}-${a.slice(2)}`);
}
