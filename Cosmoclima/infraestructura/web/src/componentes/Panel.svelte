<script>
  import {
    serieDeComuna, peorVentana, nivelDe, franjaDe, elementosQueCeden, COLORES,
  } from '../lib/riesgo.js';
  import { fechaCorta } from '../lib/fechas.js';
  import { aCSV, bajar, filasDeComuna, COLUMNAS } from '../lib/descarga.js';
  import { cargarCuenca, aguasArriba, vale } from '../lib/cuenca.js';

  let { datos, cut, dia = 0, evaluar = null } = $props();

  const comuna = $derived(datos.territorios.comunas.find((c) => c.cut === cut) ?? null);
  const s = $derived(cut ? serieDeComuna(cut, datos.celdas, datos.pronostico) : null);
  const ventana = $derived(s ? peorVentana(s.serie) : null);
  const mmHoy = $derived(ventana ? ventana.acumulados[dia] ?? 0 : null);
  const nivel = $derived(nivelDe(s ? mmHoy : null));
  const franja = $derived(mmHoy != null ? franjaDe(mmHoy, datos.umbrales) : null);
  const ceden = $derived(mmHoy != null ? elementosQueCeden(mmHoy, datos.umbrales) : []);

  // ★ El umbral de la carretera EN ESTA COMUNA, que es lo que hace comparable
  //   la cifra de arriba con algo real del lugar.
  const umbralAqui = $derived(
    evaluar && mmHoy != null ? evaluar(mmHoy, '616', cut) : null,
  );

  // Los ítems de la Matriz presentes en la comuna, ordenados por cuántos activos.
  const items = $derived.by(() => {
    const idx = datos.activos.por_comuna?.[cut] ?? {};
    const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
    return Object.entries(idx)
      .map(([n, cantidad]) => ({ cantidad, item: porN.get(n) }))
      .filter((x) => x.item)
      .sort((a, b) => b.cantidad - a.cantidad);
  });
  const totalActivos = $derived(items.reduce((a, b) => a + b.cantidad, 0));
  const fecha = $derived(fechaCorta(datos.pronostico.fechas?.[dia]));

  // ★ La cuenca: sólo importa donde el valle puede estar seco y la cordillera no.
  let cuenca = $state(null);
  $effect(() => { if (!cuenca) cargarCuenca().then((d) => (cuenca = d)); });

  const celdaAqui = $derived(datos.celdas.por_comuna?.[cut]?.celdas?.[0] ?? null);
  const arriba = $derived(
    celdaAqui && cuenca ? aguasArriba(celdaAqui, cuenca, datos.pronostico, dia) : null,
  );
  const mostrarCuenca = $derived(vale(mmHoy, arriba));

  let bajando = $state(false);
  async function descargar() {
    if (!cut) return;
    bajando = true;
    try {
      const r = await fetch(`datos/activos/${cut}.json`);
      const detalle = r.ok ? await r.json() : [];
      const filas = filasDeComuna(detalle, datos, comuna, mmHoy);
      bajar(`MICR_${comuna.comuna.replace(/\s+/g, '_')}_${fecha}.csv`,
            aCSV(filas, COLUMNAS));
    } finally {
      bajando = false;
    }
  }
</script>

{#if !comuna}
  <div class="vacio">
    <h2>Elige una comuna</h2>
    <p>
      El mapa colorea cada comuna según la lluvia acumulada en 72 horas que el
      pronóstico le asigna para el día seleccionado.
    </p>
    <p class="nota">
      El color no es una alerta. Es la lluvia esperada puesta en la escala con que
      se midió, en el temporal de julio de 2026, cada cuántos días-celda se cortó
      alguna vía.
    </p>
  </div>
{:else}
  <header>
    <h2>{comuna.comuna}</h2>
    <p class="lugar">{comuna.provincia} · {comuna.region}</p>
  </header>

  {#if !s}
    <div class="aviso">
      <strong>Sin cobertura climática.</strong>
      Esta comuna no tiene ninguna celda de pronóstico asignada. No significa que
      no vaya a llover: significa que no lo sabemos.
    </div>
  {:else}
    <section class="titular" style="border-color: {COLORES[nivel.clave]}">
      <div class="mm">
        <span class="cifra">{mmHoy.toFixed(1)}</span><span class="unidad">mm / 72 h</span>
      </div>
      <div class="etiqueta" style="color: {COLORES[nivel.clave]}">{nivel.etiqueta}</div>
      <p class="fecha">al {fecha}</p>
      {#if mostrarCuenca}
        <!-- ★★ El caso que el modelo no veía: aquí no llueve, pero sobre la
             cordillera que drena hasta aquí sí. Va SEPARADO de los milímetros
             locales y sin umbral, porque no hay registro de cuánta lluvia
             cordillerana corta un camino del valle. -->
        <p class="cuenca">
          ⚠ aquí {mmHoy.toFixed(0)} mm, pero
          <b>{arriba.mm.toFixed(0)} mm sobre la cordillera</b> que drena hasta acá
          <span class="cf">({arriba.celdas} celda{arriba.celdas === 1 ? '' : 's'} aguas arriba · sin umbral medido)</span>
        </p>
      {/if}
      {#if umbralAqui?.umbral != null}
        <p class="umbral">
          aquí una carretera cede con <b>{umbralAqui.umbral.toFixed(0)} mm</b>
          {#if umbralAqui.escala === 'local' && umbralAqui.nacional}
            <span class="cf">· {umbralAqui.nacional} mm es el promedio nacional</span>
          {/if}
        </p>
      {/if}
    </section>

    {#if franja}
      <section>
        <h3>Con esta lluvia, ¿cuántas veces se cortó una vía?</h3>
        <p class="tasa">
          <strong>{(franja.tasa * 100).toFixed(1)} %</strong> de los días-celda
          con lluvia en esta franja terminaron con al menos un corte.
        </p>
        <p class="detalle">
          Medido sobre {franja.dias_celda.toLocaleString('es-CL')} días-celda del
          temporal de julio 2026, de los cuales {franja.con_corte} tuvieron corte.
        </p>
        <p class="nota">
          Es la frecuencia observada cuando llovió así, no la probabilidad de que
          se corte una calle concreta de esta comuna.
        </p>
      </section>
    {/if}

    {#if ceden.length}
      <section>
        <h3>Elementos que cedieron con esta lluvia o menos</h3>
        <ul class="ceden">
          {#each ceden as e}
            <li>
              <span class="nombre">{e.elemento}</span>
              <span class="dato">{e.mm_72h_mediana.toFixed(0)} mm · {e.tramos} tramos</span>
            </li>
          {/each}
        </ul>
      </section>
    {/if}

    <section>
      <h3>El peor momento de los próximos {datos.pronostico.dias} días</h3>
      <p class="detalle">
        {ventana.mm.toFixed(1)} mm en 72 h, hacia el
        {fechaCorta(datos.pronostico.fechas?.[ventana.indice])}.
      </p>
    </section>
  {/if}

  <section>
    <h3>
      Infraestructura registrada
      {#if items.length}
        <button class="bajar" onclick={descargar} disabled={bajando}>
          {bajando ? 'preparando…' : 'descargar CSV'}
        </button>
      {/if}
    </h3>
    {#if !items.length}
      <p class="detalle">Sin activos indexados en esta comuna.</p>
    {:else}
      <p class="detalle">
        {totalActivos.toLocaleString('es-CL')} activos en {items.length} ítems de la Matriz.
      </p>
      <ul class="items">
        {#each items.slice(0, 12) as { item, cantidad }}
          <li>
            <span class="nombre">{item.elemento}</span>
            <span class="dato">
              {cantidad.toLocaleString('es-CL')}
              <em class="irmd" class:alto={item.IRMD === 'Alto'}>{item.IRMD}</em>
            </span>
          </li>
        {/each}
      </ul>
      {#if items.length > 12}
        <p class="detalle">y {items.length - 12} ítems más.</p>
      {/if}
    {/if}
  </section>
{/if}

<style>
  header { margin-bottom: 1rem; }
  h2 { margin: 0; font-size: 1.5rem; letter-spacing: -0.01em; }
  .lugar { margin: 0.15rem 0 0; color: #9ca3af; font-size: 0.85rem; }
  h3 {
    margin: 0 0 0.4rem;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9ca3af;
    font-weight: 600;
  }
  section { margin-bottom: 1.4rem; }
  .titular {
    border-left: 3px solid;
    padding: 0.6rem 0 0.6rem 0.8rem;
  }
  .cifra { font-size: 2.3rem; font-weight: 650; letter-spacing: -0.03em; }
  .unidad { margin-left: 0.4rem; color: #9ca3af; font-size: 0.9rem; }
  .etiqueta { text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.08em; font-weight: 600; }
  .fecha { margin: 0.2rem 0 0; color: #6b7280; font-size: 0.8rem; }
  .cuenca {
    margin: 0.4rem 0 0; padding: 0.35rem 0.5rem; font-size: 0.76rem;
    color: #d6d3d1; background: #1c1917; border-left: 2px solid #a16207;
    line-height: 1.45;
  }
  .cuenca b { color: #fbbf24; }
  .cuenca .cf { color: #78716c; display: block; font-size: 0.7rem; }
  .umbral { margin: 0.25rem 0 0; font-size: 0.76rem; color: #9ca3af; }
  .umbral b { color: #fca5a5; }
  .umbral .cf { color: #6b7280; }
  .tasa { margin: 0 0 0.3rem; font-size: 0.95rem; }
  .tasa strong { font-size: 1.3rem; }
  .detalle { margin: 0.2rem 0; color: #9ca3af; font-size: 0.82rem; line-height: 1.5; }
  .nota { margin: 0.4rem 0 0; color: #6b7280; font-size: 0.76rem; line-height: 1.5; font-style: italic; }
  ul { list-style: none; margin: 0.4rem 0 0; padding: 0; }
  li {
    display: flex; justify-content: space-between; gap: 1rem;
    padding: 0.32rem 0; border-bottom: 1px solid #1f2430; font-size: 0.85rem;
  }
  .nombre { color: #e5e7eb; }
  .dato { color: #9ca3af; white-space: nowrap; }
  .irmd { font-style: normal; font-size: 0.72rem; color: #6b7280; margin-left: 0.4rem; }
  .irmd.alto { color: #f87171; }
  .bajar {
    float: right; background: #12151d; color: #9ca3af;
    border: 1px solid #2a3040; border-radius: 5px; padding: 0.1rem 0.45rem;
    font: inherit; font-size: 0.66rem; text-transform: none; letter-spacing: 0;
    cursor: pointer;
  }
  .bajar:hover:not(:disabled) { color: #e5e7eb; border-color: #4b5563; }
  .bajar:disabled { opacity: 0.5; cursor: default; }
  .aviso {
    background: #1c1917; border-left: 3px solid #a16207;
    padding: 0.7rem 0.9rem; font-size: 0.85rem; line-height: 1.5; color: #d6d3d1;
  }
  .vacio { color: #9ca3af; }
  .vacio h2 { color: #e5e7eb; margin-bottom: 0.6rem; }
  .vacio p { font-size: 0.88rem; line-height: 1.6; }
</style>
