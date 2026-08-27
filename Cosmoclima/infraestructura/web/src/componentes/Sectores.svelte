<script>
  import { resumenPorSector, itemsDeSector, colorSector } from '../lib/sectores.js';
  import { fechaCorta } from '../lib/fechas.js';

  let {
    datos, mmPorComuna, evaluar = null,
    sector = $bindable(null), cut = $bindable(null),
    sectoresApagados = $bindable(new Set()), dia = 0,
  } = $props();

  /** ★ Enciende o apaga los puntos de un sector en el mapa, sin entrar en su
   *  detalle. Con 782.531 activos indexados, una comuna grande queda cubierta
   *  de puntos y deja de leerse: hace falta poder apagarlos. */
  function alternar(s) {
    const n = new Set(sectoresApagados);
    n.has(s) ? n.delete(s) : n.add(s);
    sectoresApagados = n;
  }

  const resumen = $derived(resumenPorSector(datos, mmPorComuna, evaluar));
  const conActivos = $derived(resumen.filter((r) => r.activos));
  const todosEncendidos = $derived(sectoresApagados.size === 0);
  // ★ Cuando hay comuna elegida, los ítems son LOS DE ESA COMUNA. Antes esta
  //   lista era siempre nacional y contradecía al territorio que el usuario
  //   tenía seleccionado.
  const items = $derived(
    sector ? itemsDeSector(datos, sector, mmPorComuna, evaluar, cut) : [],
  );
  const itemsPais = $derived(
    sector ? itemsDeSector(datos, sector, mmPorComuna, evaluar) : [],
  );
  const activosAqui = $derived(items.reduce((a, b) => a + b.activos, 0));
  const comuna = $derived(cut ? datos.territorios.comunas.find((c) => c.cut === cut) : null);
  const mmComuna = $derived(cut ? mmPorComuna.get(cut) : null);
  const fecha = $derived(fechaCorta(datos.pronostico.fechas?.[dia]));

  // Los activos individuales de la comuna elegida, cargados bajo demanda.
  let detalle = $state(null);
  let cargando = $state(false);
  $effect(() => {
    const c = cut;
    if (!c) { detalle = null; return; }
    cargando = true;
    fetch(`datos/activos/${c}.json`)
      .then((r) => (r.ok ? r.json() : []))
      .then((d) => { if (cut === c) { detalle = d; cargando = false; } })
      .catch(() => { detalle = []; cargando = false; });
  });

  // ★ Calculado, no escrito a mano. La versión anterior nombraba ocho sectores
  //   fijos —entre ellos Nuclear, Químico y Alimentario— y decía «27 ítems»;
  //   al poblar esos sectores el texto quedó afirmando algo falso y nadie se
  //   enteró, porque era una frase suelta en el HTML.
  const sinActivos = $derived(
    resumen.filter((r) => !r.activos).map((r) => r.sector),
  );
  const itemsUbicados = $derived(
    new Set(
      Object.values(datos.activos.por_comuna ?? {}).flatMap((o) => Object.keys(o)),
    ).size,
  );

  const af = $derived(datos.afectacion?.por_item ?? {});
  const porN = $derived(new Map(datos.matriz.items.map((i) => [String(i.n), i])));

  // ★ Sólo los AFECTADOS: es el filtro que pidió el caso de uso. Un listado de
  //   todo lo que hay en la comuna no le sirve a nadie; el que sirve es «estos
  //   son los que se inundan si cae esta lluvia».
  const afectadosAqui = $derived.by(() => {
    if (!detalle || mmComuna == null) return [];
    return detalle.filter((a) => {
      const item = porN.get(String(a.n));
      if (!item || (sector && item.sector !== sector)) return false;
      const e = evaluar ? evaluar(mmComuna, a.n, cut) : null;
      if (e) return e.estado === 'afectado' || e.estado === 'expuesto';
      const x = af[String(a.n)];
      return x?.tipo === 'medido' ? mmComuna >= x.umbral_mm_72h : mmComuna >= 50;
    });
  });
  const medidosAqui = $derived(
    afectadosAqui.filter((a) => af[String(a.n)]?.tipo === 'medido'),
  );
</script>

{#if !sector}
  <div class="intro">
    <h2>Por sector</h2>
    <p>
      Aquí la pregunta va al revés que en el mapa por comuna: en vez de «qué le
      viene a este territorio», <strong>qué parte de este sector queda
      comprometida</strong> con la lluvia del {fecha}.
    </p>
  </div>

  {#if cut}
    <p class="filtro">
      <button class="todos" onclick={() => (sectoresApagados = new Set())}
              disabled={todosEncendidos}>encender todos</button>
      <button class="todos" onclick={() =>
                (sectoresApagados = new Set(conActivos.map((r) => r.sector)))}
              disabled={sectoresApagados.size === conActivos.length}>apagar todos</button>
      <span class="ayuda">
        La casilla enciende o apaga los puntos de ese sector en el mapa. El
        nombre entra a su detalle.
      </span>
    </p>
  {/if}

  <ul class="sectores">
    {#each resumen as r}
      <li class="fila">
        {#if r.activos}
          <input
            type="checkbox"
            class="ojo"
            checked={!sectoresApagados.has(r.sector)}
            onchange={() => alternar(r.sector)}
            aria-label="Mostrar {r.sector} en el mapa"
            style="accent-color: {colorSector(r.sector)}"
          />
        {:else}
          <span class="ojo hueco"></span>
        {/if}
        <button onclick={() => (sector = r.sector)} class:vacio={!r.activos}>
          <span class="nom">
            <!-- ★ El mismo color con que sus elementos se dibujan en el mapa.
                 Relleno si hay algo que pintar; hueco si el sector no tiene
                 activos ubicados — el color le corresponde, pero no va a
                 aparecer nunca en el mapa. -->
            <i
              class="punto"
              class:sindatos={!r.activos}
              style={r.activos
                ? `background: ${colorSector(r.sector)}`
                : `border-color: ${colorSector(r.sector)}`}
            ></i>{r.sector}
          </span>
          {#if r.activos}
            <span class="cifras">
              {#if r.afectados}<em class="af">{r.afectados.toLocaleString('es-CL')}</em>{/if}
              {#if r.expuestos}<em class="ex">{r.expuestos.toLocaleString('es-CL')}</em>{/if}
              <span class="tot">
                {r.afectados || r.expuestos ? 'de ' : ''}{r.activos.toLocaleString('es-CL')}
              </span>
            </span>
          {:else}
            <span class="cifras"><span class="nada">sin activos ubicados</span></span>
          {/if}
        </button>
      </li>
    {/each}
  </ul>

  <p class="leyenda-mini">
    <em class="af">rojo</em> = afectado, con umbral medido ·
    <em class="ex">ámbar</em> = expuesto, sin umbral conocido
  </p>
  <p class="hueco">
    {#if sinActivos.length}
      {sinActivos.length === 1 ? 'Un sector no tiene' : `${sinActivos.length} sectores no tienen`}
      ni un activo georreferenciado —{sinActivos.join(', ')}—. No es que no les
      pase nada: es que el catastro no llega ahí.
    {/if}
    {itemsUbicados} de los {datos.matriz.items.length} ítems de la Matriz tienen
    ubicación.
  </p>
{:else}
  <header>
    <button class="volver" onclick={() => { sector = null; cut = null; }}>← sectores</button>
    <h2>
      <i class="punto grande" style="background: {colorSector(sector)}"></i>{sector}
    </h2>
    <p class="lugar">
      {#if comuna}{comuna.comuna} · {comuna.region}{:else}todo el país · al {fecha}{/if}
    </p>
  </header>

  {#if comuna}
    <section>
      <h3>Los que se afectan aquí con esta lluvia</h3>
      {#if mmComuna == null}
        <p class="detalle">Esta comuna no tiene cobertura climática.</p>
      {:else if cargando}
        <p class="detalle">Cargando activos…</p>
      {:else if !afectadosAqui.length}
        <p class="detalle">
          Ninguno: con {mmComuna.toFixed(1)} mm en 72 h ningún activo de este
          sector cruza su umbral en esta comuna.
          {#if activosAqui}
            <strong>Sí hay {activosAqui.toLocaleString('es-CL')}
            {activosAqui === 1 ? 'activo' : 'activos'}</strong> de este sector
            aquí, listados abajo: que no se afecten con esta lluvia no es lo
            mismo que no existir.
          {/if}
        </p>
      {:else}
        <p class="detalle">
          {afectadosAqui.length.toLocaleString('es-CL')} activos con
          {mmComuna.toFixed(1)} mm en 72 h
          {#if medidosAqui.length}
            · <strong>{medidosAqui.length}</strong> con umbral medido
          {/if}
        </p>
        <ul class="activos">
          {#each afectadosAqui.slice(0, 60) as a}
            {@const x = af[String(a.n)]}
            <li class:medido={x?.tipo === 'medido'}>
              <span class="nombre">{a.a}</span>
              <span class="dato">
                {porN.get(String(a.n))?.elemento?.slice(0, 26) ?? ''}
                {#if x?.tipo === 'medido'}<em class="u">{x.umbral_mm_72h} mm</em>{/if}
              </span>
            </li>
          {/each}
        </ul>
        {#if afectadosAqui.length > 60}
          <p class="detalle">y {afectadosAqui.length - 60} más.</p>
        {/if}
        <p class="nota">
          Los marcados con milímetros tienen umbral medido sobre fallas reales.
          Los demás sólo indican que va a llover mucho encima.
        </p>
      {/if}
    </section>
  {/if}

  <section>
    <h3>
      {#if comuna}Ítems del sector en {comuna.comuna}{:else}Ítems del sector en todo el país{/if}
    </h3>
    {#if !items.length}
      <p class="detalle">
        {#if comuna}
          Esta comuna no tiene ningún activo georreferenciado de este sector.
          {#if itemsPais.length}
            En el resto del país sí hay: {itemsPais.length}
            {itemsPais.length === 1 ? 'ítem' : 'ítems'} con activos ubicados.
          {/if}
        {:else}
          Este sector no tiene activos georreferenciados.
        {/if}
      </p>
    {:else}
      <ul class="items">
        {#each items as it}
          <li>
            <span class="nombre">{it.item.elemento}</span>
            <span class="dato">
              {#if it.enRiesgo}
                <em class:af={it.af?.tipo === 'medido'} class:ex={it.af?.tipo !== 'medido'}>
                  {it.enRiesgo.toLocaleString('es-CL')}
                </em>
                <span class="tot">de {it.activos.toLocaleString('es-CL')}</span>
              {:else}
                <!-- ★ Sin afectados, «de 157» queda huérfano y se lee como si
                     faltara un número. El total solo es la información. -->
                <span class="tot">{it.activos.toLocaleString('es-CL')}</span>
              {/if}
            </span>
          </li>
          {#if it.af?.tipo === 'medido'}
            <li class="explica">cede con {it.af.umbral_mm_72h} mm/72 h · {it.af.origen}</li>
          {:else if it.af?.porque}
            <li class="explica sin">{it.af.porque}</li>
          {/if}
        {/each}
      </ul>
    {/if}
  </section>

  {#if !comuna}
    <p class="nota">
      Elige una comuna en el mapa para ver los activos uno por uno.
    </p>
  {/if}
{/if}

<style>
  li.fila { display: flex; align-items: center; gap: 0.5rem; }
  li.fila > button { flex: 1; min-width: 0; }
  .ojo { width: 13px; height: 13px; flex: none; cursor: pointer; margin: 0; }
  .ojo.hueco { visibility: hidden; }

  .filtro {
    display: flex; align-items: center; gap: 0.45rem; flex-wrap: wrap;
    margin: 0 0 0.75rem; font-size: 0.74rem;
  }
  .filtro .todos {
    background: none; border: 1px solid #2a3140; color: #9ca3af;
    font: inherit; padding: 0.15rem 0.5rem; border-radius: 3px; cursor: pointer;
  }
  .filtro .todos:hover:not(:disabled) { color: #e5e7eb; border-color: #3d4657; }
  .filtro .todos:disabled { opacity: 0.35; cursor: default; }
  .filtro .ayuda { color: #6b7280; flex-basis: 100%; line-height: 1.45; }

  h2 { margin: 0; font-size: 1.35rem; letter-spacing: -0.01em; }
  h3 {
    margin: 0 0 0.4rem; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: #9ca3af; font-weight: 600;
  }
  header { margin-bottom: 1rem; }
  .lugar { margin: 0.15rem 0 0; color: #9ca3af; font-size: 0.85rem; }
  .volver {
    background: none; border: none; color: #6b7280; font: inherit;
    font-size: 0.76rem; padding: 0 0 0.35rem; cursor: pointer;
  }
  .volver:hover { color: #e5e7eb; }
  section { margin-bottom: 1.4rem; }
  .intro h2 { margin-bottom: 0.5rem; }
  .intro p { font-size: 0.86rem; line-height: 1.6; color: #9ca3af; margin: 0 0 1rem; }

  ul { list-style: none; margin: 0; padding: 0; }
  .sectores li { margin: 0; }
  .sectores button {
    display: flex; justify-content: space-between; align-items: baseline; gap: 0.8rem;
    width: 100%; padding: 0.42rem 0.3rem; background: none; border: none;
    border-bottom: 1px solid #1f2430; color: #e5e7eb; font: inherit;
    font-size: 0.85rem; text-align: left; cursor: pointer;
  }
  .sectores button:hover { background: #161b26; }
  .sectores button.vacio .nom { color: #6b7280; }
  .punto {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 0.45rem;
    vertical-align: baseline;
    border: 1px solid transparent;
  }
  /* Renombrado desde `.hueco`: esa clase es la del párrafo de aviso, con
     fondo y borde izquierdo, y el círculo la heredaba entera. */
  .punto.sindatos { background: none; }
  .punto.grande { width: 11px; height: 11px; margin-right: 0.5rem; }
  .cifras { white-space: nowrap; font-size: 0.8rem; }
  .tot { color: #6b7280; margin-left: 0.35rem; }
  .nada { color: #52525b; font-size: 0.74rem; font-style: italic; }
  em { font-style: normal; font-weight: 600; margin-left: 0.35rem; }
  em.af { color: #f87171; }
  em.ex { color: #fbbf24; }
  em.u { color: #f87171; font-size: 0.72rem; margin-left: 0.4rem; }

  .items li, .activos li {
    display: flex; justify-content: space-between; gap: 1rem;
    padding: 0.3rem 0; border-bottom: 1px solid #1f2430; font-size: 0.84rem;
  }
  .activos li.medido .nombre { color: #fca5a5; }
  .explica {
    display: block; border: none; color: #6b7280; font-size: 0.72rem;
    padding: 0 0 0.5rem; line-height: 1.4;
  }
  .explica.sin { color: #52525b; font-style: italic; }
  .nombre { color: #e5e7eb; }
  .dato { color: #9ca3af; white-space: nowrap; }
  .detalle { margin: 0.2rem 0; color: #9ca3af; font-size: 0.82rem; line-height: 1.5; }
  .nota { margin: 0.5rem 0 0; color: #6b7280; font-size: 0.75rem; line-height: 1.5; font-style: italic; }
  .leyenda-mini { font-size: 0.74rem; color: #6b7280; margin: 0.7rem 0 0; }
  .hueco {
    margin: 0.9rem 0 0; padding: 0.6rem 0.75rem; background: #16131a;
    border-left: 3px solid #a16207; font-size: 0.76rem; line-height: 1.55; color: #a8a29e;
  }
</style>
