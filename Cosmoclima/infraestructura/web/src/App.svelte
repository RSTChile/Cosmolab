<script>
  import { cargarTodo } from './lib/datos.js';
  import {
    serieDeComuna, diaMasLluvioso, primerDiaVigente, edadPronostico, COLORES,
  } from './lib/riesgo.js';
  import { fechaCorta, rangoCorto } from './lib/fechas.js';
  import { colorSector } from './lib/sectores.js';
  import { cargarUmbralLocal, evaluar as evaluarLocal } from './lib/umbralLocal.js';
  import { acumulados72h } from './lib/riesgo.js';
  import Mapa from './componentes/Mapa.svelte';
  import Panel from './componentes/Panel.svelte';
  import Dias from './componentes/Dias.svelte';
  import Sectores from './componentes/Sectores.svelte';
  import Historia from './componentes/Historia.svelte';
  import Acerca from './componentes/Acerca.svelte';
  import Consulta from './componentes/Consulta.svelte';

  let datos = $state(null);
  let error = $state(null);
  let seleccion = $state(null);
  let dia = $state(0);
  let busqueda = $state('');

  // «2026-08-25 03:41» → «25-08 03:41», el formato que usa la barra de Captura.
  const fechaBarra = (() => {
    const [f, h] = String(__COMPILADO__).split(' ');
    const [, m, d] = f.split('-');
    return `${d}-${m}${h ? ` ${h}` : ''}`;
  })();
  let pestana = $state('comuna');
  let sector = $state(null);
  // ★ Qué sectores se dibujan en el mapa. Con 782.531 activos, una comuna
  //   grande queda cubierta de puntos y no se lee nada: hacen falta apagables.
  //   `null` = todos encendidos, que es el estado inicial.
  let sectoresApagados = $state(new Set());
  let detalleComuna = $state(null);
  // Desplegado de entrada: el gráfico ya no compite con el mapa por la
  // pantalla, así que no hay razón para esconderlo. (El valor del padre manda
  // sobre el `$bindable(true)` del componente — por eso se fija aquí.)
  let historiaAbierta = $state(true);
  let umbralLocal = $state(null);
  let ruta5Traza = $state(null);
  let ruta5Riesgo = $state(null);

  // La lluvia acumulada de cada comuna para el día elegido, calculada una vez y
  // compartida por el mapa y los dos paneles.
  const mmPorComuna = $derived.by(() => {
    const m = new Map();
    if (!datos) return m;
    for (const c of datos.territorios.comunas) {
      const s = serieDeComuna(c.cut, datos.celdas, datos.pronostico);
      m.set(c.cut, s ? acumulados72h(s.serie)[dia] ?? 0 : null);
    }
    return m;
  });

  // ★ Los puntos que el mapa dibuja en la pestaña por sector: los activos que
  //   YA pasaron el filtro de afectación. El mapa no filtra nada, sólo pinta.
  const puntosAfectados = $derived.by(() => {
    if (pestana !== 'sector' || !seleccion || !detalleComuna || !datos) return null;
    sectoresApagados;   // dependencia explícita: apagar un sector repinta
    const mm = mmPorComuna.get(seleccion);
    if (mm == null) return null;
    const af = datos.afectacion?.por_item ?? {};
    const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
    return detalleComuna
      .filter((a) => {
        const item = porN.get(String(a.n));
        if (!item) return false;
        // Dentro de un sector concreto manda ese sector; en la vista general
        // se dibujan todos los que no estén apagados.
        if (sector ? item.sector !== sector
                   : sectoresApagados.has(item.sector)) return false;
        const e = evaluar(mm, a.n, seleccion);
        return e.estado === 'afectado' || e.estado === 'expuesto';
      })
      // Se le cuelga a cada punto lo que el globo necesita mostrar. Va aquí y no
      // en el archivo por comuna porque repetir el nombre del sector en 92.481
      // filas pesaría más que todo lo demás junto.
      .map((a) => {
        const item = porN.get(String(a.n));
        const x = af[String(a.n)];
        const e = evaluar(mm, a.n, seleccion);
        return {
          ...a,
          _sector: item?.sector ?? '',
          _elemento: item?.elemento ?? '',
          // ★ El umbral que se muestra es el que DECIDIÓ el estado: el LOCAL de
          //   esta celda. Tomarlo de `afectacion.json` daba el nacional, y en
          //   los ítems donde ese campo viene vacío el globo llegaba a decir
          //   «supera los  mm», sin número.
          _umbral: e.umbral ?? '',
          _escala: e.escala ?? '',
          // ★★ ROJO Y ÁMBAR NO SON DOS GRADOS DEL MISMO RIESGO.
          //    'afectado'  la lluvia supera el umbral con que ESE tipo de
          //                elemento cedió en la realidad → riesgo medido
          //    'expuesto'  va a llover mucho encima y nadie ha medido nunca qué
          //                le pasa a este tipo de elemento → riesgo DESCONOCIDO,
          //                que no es lo mismo que riesgo menor
          _estado: e.estado,
          _mm: mm,
          _origen: x?.tipo === 'medido' ? (x.origen ?? '') : '',
        };
      });
  });

  // Carga bajo demanda de los activos de la comuna elegida.
  $effect(() => {
    const c = pestana === 'sector' ? seleccion : null;
    if (!c) { detalleComuna = null; return; }
    let vigente = true;
    fetch(`datos/activos/${c}.json`)
      .then((r) => (r.ok ? r.json() : []))
      .then((d) => { if (vigente) detalleComuna = d; })
      .catch(() => { if (vigente) detalleComuna = []; });
    return () => { vigente = false; };
  });

  cargarTodo()
    .then((d) => {
      datos = d;
      // Abrir en el día que importa, no en el primero. Ver `diaMasLluvioso`.
      dia = diaMasLluvioso(d.territorios.comunas, d.celdas, d.pronostico);
    })
    .catch((e) => (error = e.message));

  // ★★ El umbral LOCAL. Se carga aparte porque son 310 KB que sólo hacen falta
  //   para evaluar, no para dibujar el mapa.
  $effect(() => {
    if (datos && !umbralLocal) cargarUmbralLocal().then((d) => (umbralLocal = d));
  });

  /** La celda representativa de cada comuna: la misma que usa el mapa. */
  const celdaDeComuna = $derived.by(() => {
    const m = new Map();
    if (!datos) return m;
    for (const [cut, info] of Object.entries(datos.celdas.por_comuna ?? {})) {
      if (info.celdas?.length) m.set(cut, info.celdas[0]);
    }
    return m;
  });

  /**
   * ★★ LA ÚNICA FUNCIÓN QUE DECIDE SI ALGO ESTÁ AFECTADO.
   *
   * Va aquí y se pasa hacia abajo para que no haya tres componentes decidiendo
   * lo mismo por su cuenta y divergiendo con el tiempo. Prefiere el umbral
   * LOCAL —el percentil que ese elemento ocupa, leído en la distribución de esa
   * celda— y cae al nacional cuando la celda no tiene episodios suficientes.
   *
   * Medido sobre los 1.241 cortes de julio: el nacional detectaba el 73 % y el
   * local detecta el 88 %, incluyendo el 100 % de los del norte, donde el
   * nacional no detectaba NINGUNO.
   */
  const evaluar = $derived((mm, item, cut) => {
    const nac = datos?.afectacion?.por_item?.[String(item)];
    const uNac = nac?.tipo === 'medido' ? nac.umbral_mm_72h : null;
    return evaluarLocal(mm, item, celdaDeComuna.get(cut), umbralLocal, uNac);
  });

  const desdeVigente = $derived(datos ? primerDiaVigente(datos.pronostico) : 0);
  const edadHoras = $derived(datos ? edadPronostico(datos.pronostico) : null);
  // Más de 24 h es un pronóstico de ayer; más de 72 h ya no debería usarse para
  // decidir nada sin volver a generarlo.
  const frescura = $derived(
    edadHoras == null ? null : edadHoras > 72 ? 'viejo' : edadHoras > 24 ? 'tibio' : 'fresco',
  );

  const serieElegida = $derived(
    datos && seleccion ? serieDeComuna(seleccion, datos.celdas, datos.pronostico)?.serie ?? null : null,
  );

  const coincidencias = $derived.by(() => {
    if (!datos || busqueda.trim().length < 2) return [];
    const q = busqueda
      .trim()
      .toLowerCase()
      .normalize('NFD')
      .replace(/[̀-ͯ]/g, '');
    return datos.territorios.comunas
      .filter((c) =>
        c.comuna.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').includes(q),
      )
      .slice(0, 8);
  });

  // ★ «hay, sin afectación» y «sin activos» son estados distintos y la leyenda
  //   tiene que nombrarlos por separado: antes el primero decía sólo «nada»,
  //   que se leía como «aquí no hay infraestructura de este sector».
  const FRACCIONES = [
    ['#27272a', 'sin activos'],
    ['#3b6ea5', 'hay, sin afectación'],
    ['#166e5c', '< 25 %'],
    ['#a16207', '25–50'],
    ['#c2410c', '50–75'],
    ['#9f1239', '> 75 %'],
  ];

  const LEYENDA = [
    ['minimo', '< 10 mm'],
    ['bajo', '10–25'],
    ['medio', '25–50'],
    ['alto', '50–100'],
    ['muyalto', '> 100'],
  ];

  function elegir(c) {
    seleccion = c.cut;
    busqueda = '';
  }
</script>

{#if error}
  <div class="pantalla">
    <div class="fallo">
      <h1>No se pudo cargar</h1>
      <p>{error}</p>
      <p class="pista">Genera los datos con <code>npm run datos</code> y recarga.</p>
    </div>
  </div>
{:else if !datos}
  <div class="pantalla"><p class="cargando">Cargando…</p></div>
{:else if datos.fallas.length}
  <!-- ★ El portero. Un mapa incompleto se ve igual que uno completo, así que
       cuando los conteos no calzan la aplicación se niega a dibujar. -->
  <div class="pantalla">
    <div class="fallo">
      <h1>Los datos no cuadran</h1>
      <p>Lo cargado no coincide con lo que declara el manifiesto:</p>
      <ul>{#each datos.fallas as f}<li>{f}</li>{/each}</ul>
      <p class="pista">
        Se prefiere no mostrar nada antes que un mapa al que le falten comunas sin
        que se note. Vuelve a correr <code>npm run datos</code>.
      </p>
    </div>
  </div>
{:else}
  <!-- ★ Misma barra que App Captura, medida sobre la propia página: fondo
       rgba(6,10,22,.82), borde inferior azulado, el nombre en blanco con la
       versión en el dorado del sello, y debajo la línea de compilación. -->
  <header class="barra">
    <button
      class="banner"
      onclick={() => { seleccion = null; sector = null; busqueda = ''; }}
      title="Volver a la vista general"
    >
      <img src="micr-logo.png" alt="" width="96" height="96" />
      <span class="txt">
        <span class="nombre">RMD <strong>2.0</strong> MICR</span>
        <span class="version" title="Versión compilada (UTC): {__VERSION__} · {__COMPILADO__}">
          Matriz de Infraestructura Crítica · app v{__VERSION__} · {fechaBarra}
        </span>
      </span>
    </button>
    <nav class="enlaces">
      {#if datos && frescura}
        <!-- ★ La edad del pronóstico va ARRIBA y a la vista, no en gris de 9 px
             al pie: es la diferencia entre mirar una previsión y mirar el
             pasado creyendo que es previsión. -->
        <span class="frescura {frescura}" title="Generado el {fechaCorta(datos.pronostico.generado)}">
          {#if edadHoras < 1}
            recién actualizado
          {:else if edadHoras < 24}
            pronóstico de hace {Math.round(edadHoras)} h
          {:else}
            ⚠ pronóstico de hace {Math.round(edadHoras / 24)} día{Math.round(edadHoras / 24) === 1 ? '' : 's'}
          {/if}
        </span>
      {/if}
      <a href="https://captura.cosmosemiotica.cl/" target="_blank" rel="noopener noreferrer">
        App Captura
      </a>
    </nav>
  </header>

  <div class="disposicion">
    <div class="lienzo">
      <Mapa
        {datos}
        bind:seleccion
        {dia}
        modo={pestana === 'sector' ? 'sector' : 'clima'}
        {sector}
        {mmPorComuna}
        puntos={puntosAfectados}
        ruta5={ruta5Traza}
        tramosRuta5={ruta5Riesgo}
        {evaluar}
      />

      <div class="buscador">
        <input
          type="search"
          placeholder="Buscar comuna…"
          bind:value={busqueda}
          aria-label="Buscar comuna"
        />
        {#if coincidencias.length}
          <ul class="sugerencias">
            {#each coincidencias as c}
              <li>
                <button onclick={() => elegir(c)}>
                  {c.comuna}<span>{c.region}</span>
                </button>
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <div class="leyenda">
        {#if pestana === 'sector' && sector}
          <span class="titulo">fracción del sector comprometida</span>
          {#each FRACCIONES as [color, texto]}
            <span class="par"><i style="background: {color}"></i>{texto}</span>
          {/each}
          {#if puntosAfectados?.length}
            <span class="par sep">
              <i style="background: {colorSector(sector)}"></i>{sector}
            </span>
            <!-- ★ La marca hueca necesita leyenda propia: sin ella el ámbar se
                 lee como «riesgo intermedio» y significa lo contrario. -->
            <span class="par"><i class="marca-medido"></i>riesgo medido</span>
            <span class="par"><i class="marca-desconocido"></i>riesgo desconocido</span>
          {/if}
        {:else}
          <span class="titulo">lluvia acumulada 72 h</span>
          {#each LEYENDA as [clave, texto]}
            <span class="par"><i style="background: {COLORES[clave]}"></i>{texto}</span>
          {/each}
        {/if}
      </div>
    </div>

    <aside>
      <div class="marca">
        <h1>MICR · Infraestructura y clima</h1>
        <p>
          {datos.matriz.items.length} ítems · {datos.territorios.comunas.length} comunas ·
          pronóstico de {datos.pronostico.dias} días
        </p>
      </div>

      <nav class="pestanas">
        <button class:activa={pestana === 'comuna'} onclick={() => (pestana = 'comuna')}>
          Por comuna
        </button>
        <button class:activa={pestana === 'sector'} onclick={() => (pestana = 'sector')}>
          Por sector
        </button>
        <button class:activa={pestana === 'consulta'} onclick={() => (pestana = 'consulta')}>
          Consultar
        </button>
        <button class:activa={pestana === 'acerca'} onclick={() => (pestana = 'acerca')}>
          Acerca de
        </button>
      </nav>

      <div class="franja-dias">
        <Dias pronostico={datos.pronostico} serie={serieElegida} bind:dia {desdeVigente} />
      </div>

      <div class="contenido">
        {#if pestana === 'comuna'}
          <Panel {datos} cut={seleccion} {dia} {evaluar} />
        {:else if pestana === 'sector'}
          <Sectores {datos} {mmPorComuna} {evaluar} bind:sector bind:cut={seleccion}
                    bind:sectoresApagados {dia} />
        {:else if pestana === 'consulta'}
          <Consulta {datos} {mmPorComuna} {evaluar} {dia} bind:ruta5Traza bind:ruta5Riesgo />
        {:else}
          <Acerca {datos} />
        {/if}
      </div>

      <footer>
        {#if datos.fallidas}
          <p class="alerta">
            {datos.fallidas} celdas del pronóstico no se pudieron bajar.
          </p>
        {/if}
        <p>
          Pronóstico: {datos.pronostico.fuente}, generado el
          {fechaCorta(datos.pronostico.generado)}.
          Climatología congelada {rangoCorto(datos.climatologia.congelada)}.
        </p>
        <p class="limite">
          <strong>Esto es insumo, no una alerta.</strong>
          Las alertas las emite SENAPRED. Aquí se cruza infraestructura registrada
          con lluvia pronosticada; nada de lo que se muestra reemplaza esa función.
        </p>
      </footer>
    </aside>
  </div>

  <!-- ★ Fuera del grid y a ancho completo: el gráfico no compite con el mapa
       por la pantalla, se llega a él bajando. Así el mapa se ve entero y la
       serie tiene sitio para respirar. -->
  <Historia {datos} cut={seleccion} bind:abierto={historiaAbierta} />
{/if}

<style>
  .barra {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    height: 58px;
    padding: 12.8px 20px;
    background: rgba(6, 10, 22, 0.82);
    border-bottom: 1px solid rgba(125, 165, 255, 0.16);
  }
  .banner {
    display: flex;
    align-items: center;
    gap: 8.8px;
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    text-align: left;
  }
  .banner img {
    width: 29.6px;
    height: 29.6px;
    border-radius: 50%;
    object-fit: contain;
    box-shadow: 0 0 0 1px rgba(227, 182, 103, 0.34);
  }
  .txt { display: flex; flex-direction: column; }
  .nombre {
    color: #eaf1ff;
    font-size: 15.2px;
    font-weight: 500;
    letter-spacing: 0.304px;
    line-height: 1.15;
  }
  .nombre strong { color: #e3b667; font-weight: 800; }
  .version {
    color: #8695ba;
    font-size: 9.92px;
    letter-spacing: 0.397px;
    line-height: 11.9px;
  }
  .enlaces { display: flex; align-items: center; gap: 1.1rem; }
  .frescura {
    font-size: 0.72rem;
    padding: 0.12rem 0.5rem;
    border-radius: 99px;
    border: 1px solid transparent;
  }
  .frescura.fresco { color: #6ee7b7; border-color: #06543c; background: #052e22; }
  .frescura.tibio { color: #fbbf24; border-color: #6b4708; background: #2b1f05; }
  .frescura.viejo { color: #fca5a5; border-color: #7f1d1d; background: #2b0a0a; }
  .enlaces a {
    color: #8695ba;
    text-decoration: none;
    font-size: 0.8rem;
    padding: 0.2rem 0;
    border-bottom: 1px solid transparent;
  }
  .enlaces a:hover { color: #eaf1ff; border-bottom-color: #e3b667; }

  /* ★★ La página tiene SCROLL VERTICAL, no es una pantalla fija.
     El mapa ocupa la altura completa de la ventana menos la barra, así que se
     ve entero y sin recortes; el gráfico queda debajo y se llega a él bajando.
     Antes ambos competían por la misma pantalla: o el mapa quedaba mutilado o
     el gráfico salía aplastado. */
  .disposicion {
    display: grid;
    grid-template-columns: 1fr 400px;
    height: calc(100vh - 58px);
  }
  .lienzo { position: relative; min-height: 0; }

  aside {
    display: flex;
    flex-direction: column;
    background: #0e121a;
    border-left: 1px solid #1f2430;
    overflow: hidden;
  }
  .marca { padding: 1rem 1.2rem 0.7rem; border-bottom: 1px solid #1f2430; }
  .marca h1 { margin: 0; font-size: 0.95rem; letter-spacing: 0.01em; }
  .marca p { margin: 0.2rem 0 0; font-size: 0.74rem; color: #6b7280; }
  .pestanas {
    display: flex;
    gap: 2px;
    padding: 0 0.8rem;
    border-bottom: 1px solid #1f2430;
  }
  .pestanas button {
    flex: 1;
    padding: 0.5rem 0.15rem;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    color: #6b7280;
    font: inherit;
    font-size: 0.72rem;
    cursor: pointer;
    white-space: nowrap;
  }
  .pestanas button:hover { color: #9ca3af; }
  .pestanas button.activa { color: #e5e7eb; border-bottom-color: #c2410c; }

  .franja-dias {
    height: 66px;
    padding: 0.5rem 0.8rem 0.35rem;
    border-bottom: 1px solid #1f2430;
    flex: none;
  }
  .contenido { flex: 1; overflow-y: auto; padding: 1.2rem; }
  footer {
    border-top: 1px solid #1f2430;
    padding: 0.7rem 1.2rem 0.9rem;
    font-size: 0.7rem;
    color: #6b7280;
    line-height: 1.5;
  }
  footer p { margin: 0 0 0.35rem; }
  .limite { color: #9ca3af; }
  .alerta { color: #fbbf24; }

  .buscador { position: absolute; top: 12px; left: 12px; width: 250px; z-index: 5; }
  input {
    width: 100%;
    padding: 0.5rem 0.7rem;
    background: #0e121aee;
    border: 1px solid #2a3040;
    border-radius: 6px;
    color: #e5e7eb;
    font: inherit;
    font-size: 0.85rem;
  }
  input:focus { outline: none; border-color: #4b5563; }
  .sugerencias {
    list-style: none;
    margin: 4px 0 0;
    padding: 4px;
    background: #0e121af8;
    border: 1px solid #2a3040;
    border-radius: 6px;
  }
  .sugerencias button {
    display: flex;
    justify-content: space-between;
    gap: 0.6rem;
    width: 100%;
    padding: 0.35rem 0.5rem;
    background: none;
    border: none;
    color: #e5e7eb;
    font: inherit;
    font-size: 0.82rem;
    text-align: left;
    cursor: pointer;
    border-radius: 4px;
  }
  .sugerencias button:hover { background: #1f2937; }
  .sugerencias span { color: #6b7280; font-size: 0.74rem; }

  .leyenda {
    position: absolute;
    left: 12px;
    bottom: 12px;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.45rem 0.7rem;
    background: #0e121aee;
    border: 1px solid #2a3040;
    border-radius: 6px;
    font-size: 0.72rem;
    color: #9ca3af;
    z-index: 5;
  }
  .leyenda .titulo { color: #6b7280; }
  .par { display: flex; align-items: center; gap: 0.3rem; }
  .par.sep { border-left: 1px solid #2a3040; padding-left: 0.7rem; }
  .leyenda i { width: 11px; height: 11px; border-radius: 2px; display: inline-block; }
  .leyenda i.marca-medido {
    border-radius: 50%; background: rgba(239, 68, 68, 0.22);
    border: 1.6px solid #ef4444;
  }
  .leyenda i.marca-desconocido {
    border-radius: 50%; background: none; border: 1.4px solid #f59e0b;
  }

  .pantalla {
    display: grid;
    place-items: center;
    height: 100vh;
    padding: 2rem;
  }
  .cargando { color: #6b7280; }
  .fallo { max-width: 520px; }
  .fallo h1 { font-size: 1.2rem; margin: 0 0 0.6rem; color: #f87171; }
  .fallo p { font-size: 0.9rem; line-height: 1.6; color: #9ca3af; margin: 0 0 0.5rem; }
  .fallo ul { color: #e5e7eb; font-size: 0.85rem; line-height: 1.7; }
  .pista { color: #6b7280 !important; }
  code { background: #1f2430; padding: 0.1rem 0.35rem; border-radius: 3px; }

  @media (max-width: 900px) {
    .disposicion {
      grid-template-columns: 1fr;
      grid-template-rows: 48vh 1fr;
      height: auto;
    }
    aside { border-left: none; border-top: 1px solid #1f2430; }
  }
</style>
