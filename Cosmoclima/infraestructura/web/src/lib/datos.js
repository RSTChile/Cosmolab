/**
 * CARGA DE DATOS · y el portero que revisa que nada se haya perdido
 * ==================================================================
 *
 * ★ POR QUÉ LA APLICACIÓN SE NIEGA A ARRANCAR SI EL MANIFIESTO NO CUADRA
 * ------------------------------------------------------------------------
 * Este proyecto lleva semanas encontrándose el mismo tipo de error: **nada
 * revienta, todo devuelve un número plausible.** Un `$batch` que perdió 208
 * filas devolvió HTTP 200. Una columna calculada que daba 1 en todas las filas
 * se aceptó sin protestar. Un CSV con 0,6667 donde iba 2/3 movió 237 ítems de
 * banda sin una sola advertencia.
 *
 * Un mapa al que le faltan comunas se ve exactamente igual que uno completo.
 * Por eso `construir.py` escribe los conteos de todo lo que generó y aquí se
 * comparan contra lo cargado: si no calzan, la aplicación muestra el error en
 * pantalla en vez de dibujar un mapa incompleto que nadie va a cuestionar.
 */

const BASE = 'datos';

async function traer(nombre) {
  const r = await fetch(`${BASE}/${nombre}`);
  if (!r.ok) throw new Error(`No se pudo cargar ${nombre} (HTTP ${r.status})`);
  return r.json();
}

export async function cargarTodo() {
  const [manifiesto, territorios, matriz, activos, climatologia, pronostico, celdas, umbrales, topo] =
    await Promise.all([
      traer('manifiesto.json'),
      traer('territorios.json'),
      traer('matriz.json'),
      traer('activos_por_comuna.json'),
      traer('climatologia.json'),
      traer('pronostico.json'),
      traer('celdas_por_comuna.json'),
      traer('umbrales.json'),
      traer('comunas.topo.json'),
    ]);

  // ── el portero ──────────────────────────────────────────────────────────
  const esperado = manifiesto.esperado ?? {};
  const real = {
    comunas: territorios.comunas.length,
    provincias: territorios.provincias.length,
    regiones: territorios.regiones.length,
    comunas_geometria: topo.objects.comunas.geometries.length,
    items: matriz.items.length,
  };
  const fallas = [];
  for (const [k, v] of Object.entries(real)) {
    if (esperado[k] != null && esperado[k] !== v) {
      fallas.push(`${k}: se esperaban ${esperado[k]} y llegaron ${v}`);
    }
  }
  // El pronóstico es lo único que puede venir corto sin ser un error: se baja de
  // un servicio externo y el propio artefacto declara qué celdas fallaron.
  const fallidas = pronostico.celdas_fallidas?.length ?? 0;

  return {
    manifiesto, territorios, matriz, activos, climatologia, pronostico,
    celdas, umbrales, topo, fallas, fallidas,
  };
}
