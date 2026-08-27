<script>
  import { onMount, onDestroy } from 'svelte';
  import maplibregl from 'maplibre-gl';
  import 'maplibre-gl/dist/maplibre-gl.css';
  import { aGeoJSON } from '../lib/topo.js';
  import { COLORES, nivelDe, serieDeComuna, peorVentana } from '../lib/riesgo.js';
  import { fraccionPorComuna, colorFraccion, colorSector } from '../lib/sectores.js';

  let {
    datos,
    seleccion = $bindable(null),
    dia = 0,
    // ★ `modo` cambia qué significa el color: 'clima' pinta lluvia por comuna
    //   (pestaña 1) y 'sector' pinta la fracción del sector comprometida.
    modo = 'clima',
    sector = null,
    mmPorComuna = null,
    puntos = null,
    ruta5 = null,
    tramosRuta5 = null,
    evaluar = null,
  } = $props();

  let contenedor;
  let mapa = null;
  let listo = $state(false);

  // ★★ CON MAPA BASE, Y POR QUÉ SE CAMBIÓ DE OPINIÓN
  //   La primera versión dibujaba las comunas sobre un fondo liso, para no
  //   depender de ningún servidor ajeno. Estaba mal pensado: el objetivo de
  //   esta herramienta es poder decir «el paso bajo nivel de tal calle se
  //   inunda», y un punto rojo sobre negro no dice DÓNDE está. Sin calles el
  //   mapa es mudo justo en el momento en que más tiene que hablar.
  //
  //   Se usa el estilo vectorial oscuro de CARTO: sin clave de API, calles
  //   nítidas a cualquier acercamiento y con sus nombres. Vectorial y no
  //   imágenes porque a nivel de manzana las imágenes se ven borrosas, que es
  //   exactamente el nivel donde hace falta leer el nombre de la calle.
  //   La atribución a OpenStreetMap y CARTO es obligatoria y va activada.
  const ESTILO = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

  /**
   * Dónde insertar nuestras capas: **debajo de las calles**, no sólo debajo de
   * los nombres.
   *
   * ★ Medido en el estilo de CARTO: 19 de sus 23 capas de calles están ANTES de
   * la primera etiqueta. Insertar «antes del primer símbolo» —que era el reflejo
   * obvio— dejaba el relleno de comunas ENCIMA de casi todas las calles y las
   * borraba, que es justo lo que este cambio venía a arreglar.
   */
  function antesDeLasCalles() {
    const capas = mapa.getStyle()?.layers ?? [];
    for (const c of capas) {
      if (/road|highway|transportation|tunnel|bridge/.test(c.id)) return c.id;
    }
    for (const c of capas) if (c.type === 'symbol') return c.id;
    return undefined;
  }

  function colorPorComuna() {
    // Una expresión `match` de MapLibre: comuna → color, resuelta en la GPU.
    const expr = ['match', ['get', 'CUT_COM']];
    const frac =
      modo === 'sector' && sector && mmPorComuna
        ? fraccionPorComuna(datos, sector, mmPorComuna, evaluar)
        : null;
    for (const c of datos.territorios.comunas) {
      let color = COLORES.sincobertura;
      if (frac) {
        const f = frac.get(c.cut);
        color = f ? colorFraccion(f.fraccion) : '#20242e';
      } else {
        const s = serieDeComuna(c.cut, datos.celdas, datos.pronostico);
        if (s) {
          const ac = peorVentana(s.serie).acumulados;
          color = COLORES[nivelDe(ac[dia] ?? 0).clave];
        }
      }
      expr.push(c.cut, color);
    }
    expr.push(COLORES.sincobertura);
    return expr;
  }

  onMount(() => {
    const geo = aGeoJSON(datos.topo, 'comunas');
    mapa = new maplibregl.Map({
      container: contenedor,
      style: ESTILO,
      center: [-71, -38],
      zoom: 3.4,
    });
    mapa.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    // ★ Sin este manejador, un error de estilo de MapLibre no aparece por ningún
    //   lado: la capa simplemente no se dibuja y el mapa se ve vacío, que es
    //   exactamente el modo de fallo silencioso que este proyecto persigue.
    mapa.on('error', (e) => console.error('[mapa]', e?.error?.message ?? e));
    if (import.meta.env.DEV) window.__mapa = mapa;

    mapa.on('load', () => {
      const antes = antesDeLasCalles();
      mapa.addSource('comunas', { type: 'geojson', data: geo });
      mapa.addLayer({
        id: 'relleno', type: 'fill', source: 'comunas',
        paint: {
          'fill-color': colorPorComuna(),
          // ★ El relleno CEDE al acercarse. A escala de país el color es lo que
          //   importa y va casi opaco; a escala de manzana lo que importa son
          //   las calles, así que baja a un tinte. Sin esto habría que elegir
          //   entre ver el riesgo o ver dónde está, y hacen falta las dos.
          'fill-opacity': ['interpolate', ['linear'], ['zoom'], 5, 0.82, 9, 0.45, 12, 0.18],
        },
      }, antes);
      mapa.addLayer({
        id: 'borde', type: 'line', source: 'comunas',
        paint: { 'line-color': '#0b0e14', 'line-width': 0.3, 'line-opacity': 0.7 },
      }, antes);
      mapa.addLayer({
        id: 'elegida', type: 'line', source: 'comunas',
        paint: { 'line-color': '#e5e7eb', 'line-width': 2 },
        filter: ['==', ['get', 'CUT_COM'], ''],
      }, antes);

      // ★ Los activos uno por uno. Sólo aparecen con una comuna elegida: un
      //   sector como Telecomunicaciones tiene 16.660 activos y pintarlos todos
      //   a escala nacional satura el mapa y lo vuelve lento sin decir nada.
      mapa.addSource('puntos', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      mapa.addLayer({
        id: 'puntos', type: 'circle', source: 'puntos',
        paint: {
          // Crecen con el zoom: legibles de lejos, precisos de cerca.
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 7, 3, 12, 7, 16, 11],
          // ★ Un color por categoría: con varias a la vista, un solo color
          //   diría «hay muchas cosas» en vez de «hay hospitales y escuelas».
          'circle-color': ['get', 'color'],
          'circle-stroke-color': '#0b0e14',
          'circle-stroke-width': 1,
          'circle-opacity': 0.95,
        },
      });  // los puntos SÍ van arriba de todo: son el dato, no el contexto

      // ── LA MARCA DE RIESGO ────────────────────────────────────────────────
      // Dos marcas, y deliberadamente NO son dos tonos del mismo semáforo:
      //
      //   disco rojo    el elemento cruza el umbral con que ese mismo tipo de
      //                 elemento cedió en la realidad. Riesgo MEDIDO.
      //   anillo ámbar  va a llover mucho encima y nadie ha medido nunca qué le
      //                 pasa a este tipo de elemento. Riesgo DESCONOCIDO.
      //
      // ★ El anillo es hueco a propósito. Con dos discos de distinto color el
      //   ámbar se lee como «riesgo moderado», y no lo es: es riesgo sin medir,
      //   que perfectamente puede ser mayor. La forma dice «aquí falta un
      //   dato», el color solo no lo diría.
      // Anillo fino alrededor de los puntos donde hay más de un activo: sin
      // esto, un punto con cuatro activos se ve idéntico a uno con uno solo.
      mapa.addLayer({
        id: 'apilados', type: 'circle', source: 'puntos',
        filter: ['>', ['length', ['get', 'juntos']], 40],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 7, 8, 12, 15, 16, 22],
          'circle-color': 'rgba(0,0,0,0)',
          'circle-stroke-color': '#e5e7eb',
          'circle-stroke-width': 1,
          'circle-stroke-opacity': 0.45,
        },
      });
      mapa.addLayer({
        id: 'riesgo-medido', type: 'circle', source: 'puntos',
        filter: ['==', ['get', 'estado'], 'afectado'],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 7, 6, 12, 12, 16, 18],
          'circle-color': '#ef4444',
          'circle-opacity': 0.22,
          'circle-stroke-color': '#ef4444',
          'circle-stroke-width': 1.6,
          'circle-stroke-opacity': 0.95,
        },
      });
      mapa.addLayer({
        id: 'riesgo-desconocido', type: 'circle', source: 'puntos',
        filter: ['==', ['get', 'estado'], 'expuesto'],
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 7, 6, 12, 12, 16, 18],
          'circle-color': 'rgba(0,0,0,0)',   // hueco: es la mitad del mensaje
          'circle-stroke-color': '#f59e0b',
          'circle-stroke-width': 1.4,
          'circle-stroke-opacity': 0.9,
        },
      });

      // ── la Ruta 5, como LÍNEA y no como puntos ──────────────────────────
      //   Es la única infraestructura de la página que es lineal por
      //   naturaleza: pintarla como una nube de puntos habría perdido lo único
      //   que la hace lo que es, que es continuar.
      mapa.addSource('ruta5', {
        type: 'geojson',
        data: { type: 'FeatureCollection', features: [] },
      });
      mapa.addLayer({
        id: 'ruta5-halo', type: 'line', source: 'ruta5',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#0b0e14', 'line-opacity': 0.8,
          'line-width': ['interpolate', ['linear'], ['zoom'], 4, 4, 10, 8],
        },
      });
      mapa.addLayer({
        id: 'ruta5', type: 'line', source: 'ruta5',
        layout: { 'line-cap': 'round', 'line-join': 'round' },
        paint: {
          'line-color': '#f59e0b',
          'line-width': ['interpolate', ['linear'], ['zoom'], 4, 1.6, 10, 4],
        },
      });
      // Los tramos que hoy superan su umbral, encima y en rojo.
      mapa.addSource('ruta5riesgo', {
        type: 'geojson', data: { type: 'FeatureCollection', features: [] },
      });
      mapa.addLayer({
        id: 'ruta5riesgo', type: 'circle', source: 'ruta5riesgo',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 4, 2.5, 10, 6],
          'circle-color': '#ef4444',
          'circle-stroke-color': '#0b0e14',
          'circle-stroke-width': 1,
        },
      });
      listo = true;
    });

    mapa.on('click', 'relleno', (e) => {
      seleccion = e.features?.[0]?.properties?.CUT_COM ?? null;
    });
    const globo = new maplibregl.Popup({
      closeButton: false, closeOnClick: false, offset: 10,
    });
    const esc = (t) =>
      String(t ?? '').replace(/[&<>"]/g, (c) =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

    /** El globo: qué es, con cuánta lluvia cede y qué le ha pasado antes. */
    function ficha(p) {
      // ★ Sin nombre se muestra el TIPO. Escribir «None» era peor que nada:
      //   parece un error del programa cuando es un hueco del catastro.
      const titulo = p.nombre?.trim() || p.elemento || 'Sin nombre registrado';
      const l = [`<strong>${esc(titulo)}</strong>`];
      if (p.nombre?.trim() && p.elemento) l.push(`<i>${esc(p.elemento)}</i>`);
      if (p.sector) l.push(`<span class="sec">${esc(p.sector)}</span>`);
      // ★ Por qué está marcado. Sin esto la marca es un color sin motivo.
      if (p.estado === 'afectado') {
        // ★ Los milímetros llegan como suma de flotantes: sin redondear, el
      //   globo llega a decir «57.599999999999994 mm».
      const mm = p.mm === '' || p.mm == null ? '' : Number(p.mm).toFixed(1);
      const u = p.umbral === '' || p.umbral == null
          ? null : Number(p.umbral).toFixed(1);
        const donde = p.escala === 'local'
          ? 'el umbral medido para este lugar' : 'el umbral medido';
        l.push(`<span class="riesgo medido">Riesgo medido — le caen ` +
               `${esc(mm)} mm en 72 h` +
               (u ? ` y ${donde} es ${u} mm` : '') +
               `${p.origen ? ` (${esc(p.origen)})` : ''}.</span>`);
      } else if (p.estado === 'expuesto') {
        l.push(`<span class="riesgo desconocido">Riesgo desconocido — le caen ` +
               `${esc(mm)} mm en 72 h y nadie ha medido nunca con cuánta ` +
               `lluvia cede este tipo de elemento.</span>`);
      } else if (p.umbral) {
        l.push(`<span class="um">cede con ${esc(p.umbral)} mm/72 h</span>`);
      }

      // Si varios activos comparten este punto exacto, se nombran todos: de
      // otro modo el globo describe uno y los demás son invisibles.
      let j = [];
      try { j = JSON.parse(p.juntos || '[]'); } catch { j = []; }
      if (j.length > 1) {
        l.push(`<hr><span class="tit">${j.length} activos en este mismo punto` +
               `</span>`);
        for (const x of j.slice(0, 6)) l.push(`<span class="ant">· ${esc(x)}</span>`);
        if (j.length > 6) l.push(`<span class="ant">y ${j.length - 6} más</span>`);
      }

      let h = [];
      try { h = JSON.parse(p.hist || '[]'); } catch { h = []; }
      if (h.length) {
        l.push('<hr><span class="tit">Antecedentes en el sector</span>');
        for (const a of h.slice(0, 4)) {
          if (a.t === 'pc') {
            l.push(`<span class="ant">· Punto crítico SENAPRED — ${esc(a.c)}` +
                   `${a.r ? ` (riesgo ${esc(a.r)})` : ''} · ${a.d} m</span>`);
          } else if (a.t === 'via') {
            const f = esc(a.f).slice(0, 10).split('-').reverse().join('-');
            l.push(`<span class="ant">· Vía cortada el ${f}` +
                   `${a.g ? ` — ${esc(a.g)}` : ''} · ${a.d} m</span>`);
          } else {
            l.push(`<span class="ant">· ${esc(a.p)} en ${esc(a.m)} ${esc(a.a)}` +
                   `${a.pp ? ` con ${esc(a.pp)}` : ''} · ${a.d} m</span>`);
          }
        }
        if (h.length > 4) l.push(`<span class="ant">y ${h.length - 4} más</span>`);
      }
      return `<div class="ficha">${l.join('')}</div>`;
    }

    for (const capa of ['riesgo-medido', 'riesgo-desconocido']) {
      mapa.on('mouseenter', capa, (e) => {
        mapa.getCanvas().style.cursor = 'pointer';
        const f = e.features?.[0];
        if (f) globo.setLngLat(f.geometry.coordinates).setHTML(ficha(f.properties)).addTo(mapa);
      });
      mapa.on('mouseleave', capa, () => {
        mapa.getCanvas().style.cursor = '';
        globo.remove();
      });
    }
    mapa.on('mouseenter', 'puntos', (e) => {
      mapa.getCanvas().style.cursor = 'pointer';
      const f = e.features?.[0];
      if (f) globo.setLngLat(f.geometry.coordinates).setHTML(ficha(f.properties)).addTo(mapa);
    });
    mapa.on('mouseleave', 'puntos', () => {
      mapa.getCanvas().style.cursor = '';
      globo.remove();
    });
    mapa.on('mouseenter', 'relleno', () => (mapa.getCanvas().style.cursor = 'pointer'));
    mapa.on('mouseleave', 'relleno', () => (mapa.getCanvas().style.cursor = ''));
  });

  onDestroy(() => mapa?.remove());

  // Repintar cuando cambia el día del pronóstico.
  $effect(() => {
    if (listo && mapa?.getLayer('relleno')) {
      mapa.setPaintProperty('relleno', 'fill-color', colorPorComuna());
    }
  });

  // Resaltar la comuna elegida.
  $effect(() => {
    if (listo && mapa?.getLayer('elegida')) {
      mapa.setFilter('elegida', ['==', ['get', 'CUT_COM'], seleccion ?? '']);
    }
  });

  // La traza de la Ruta 5 y sus tramos en riesgo.
  $effect(() => {
    if (!listo || !mapa?.getSource('ruta5')) return;
    mapa.getSource('ruta5').setData({
      type: 'FeatureCollection',
      features: (ruta5?.trazas ?? []).map((l) => ({
        type: 'Feature',
        properties: {},
        geometry: { type: 'LineString', coordinates: l.map(([la, lo]) => [lo, la]) },
      })),
    });
    mapa.getSource('ruta5riesgo').setData({
      type: 'FeatureCollection',
      features: (tramosRuta5 ?? []).map((t) => ({
        type: 'Feature',
        properties: {},
        geometry: { type: 'Point', coordinates: [t.x, t.y] },
      })),
    });
  });

  // Los puntos que entrega el panel (ya filtrados a los afectados).
  $effect(() => {
    if (!listo || !mapa?.getSource('puntos')) return;
    const lista = puntos ?? [];

    // ★★ ACTIVOS QUE COMPARTEN COORDENADA EXACTA.
    //    En Pirque, los 2 aeródromos y los 2 radares están en el MISMO punto:
    //    el catastro del MOP le da a la radioayuda la posición de la pista. El
    //    mapa dibujaba cuatro puntos y dos quedaban debajo, así que el panel
    //    decía «2 y 2» y sólo se veían 2. No se mueven las coordenadas —serían
    //    falsas—: se cuenta cuántos hay en cada punto y el globo los lista.
    const enElPunto = new Map();
    for (const a of lista) {
      const k = `${a.y.toFixed(5)},${a.x.toFixed(5)}`;
      if (!enElPunto.has(k)) enElPunto.set(k, []);
      enElPunto.get(k).push(`${a._elemento || ''}${a.a ? ` · ${a.a}` : ''}`);
    }

    mapa.getSource('puntos').setData({
      type: 'FeatureCollection',
      features: lista.map((a) => ({
        type: 'Feature',
        properties: {
          nombre: a.a || '',
          elemento: a._elemento ?? '',
          sector: a._sector ?? '',
          color: colorSector(a._sector),
          estado: a._estado ?? '',
          mm: a._mm ?? '',
          origen: a._origen ?? '',
          umbral: a._umbral ?? '',
          escala: a._escala ?? '',
          juntos: JSON.stringify(
            enElPunto.get(`${a.y.toFixed(5)},${a.x.toFixed(5)}`) ?? []),
          hist: JSON.stringify(a.h ?? []),
        },
        geometry: { type: 'Point', coordinates: [a.x, a.y] },
      })),
    });
    // Encuadrar sobre ellos: de nada sirve pintar 40 puntos si el mapa sigue
    // mostrando el país entero y no se ven.
    if (lista.length) {
      let x0 = 180, y0 = 90, x1 = -180, y1 = -90;
      for (const a of lista) {
        x0 = Math.min(x0, a.x); x1 = Math.max(x1, a.x);
        y0 = Math.min(y0, a.y); y1 = Math.max(y1, a.y);
      }
      mapa.fitBounds([[x0, y0], [x1, y1]], { padding: 70, maxZoom: 14, duration: 700 });
    }
  });
</script>

<div class="mapa" bind:this={contenedor}></div>

<style>
  :global(.maplibregl-popup-content) {
    background: #0e121a;
    border: 1px solid #2a3040;
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
    max-width: 300px;
    box-shadow: 0 6px 20px #0009;
  }
  :global(.maplibregl-popup-tip) { border-top-color: #0e121a !important; }
  :global(.ficha .riesgo) {
    display: block; margin-top: 0.35rem; padding: 0.3rem 0.4rem;
    border-left: 2px solid; font-size: 0.72rem; line-height: 1.45;
  }
  :global(.ficha .riesgo.medido) { border-color: #ef4444; color: #fca5a5; }
  :global(.ficha .riesgo.desconocido) { border-color: #f59e0b; color: #fcd34d; }
  :global(.ficha) { font-size: 0.76rem; line-height: 1.45; color: #e5e7eb; }
  :global(.ficha strong) { display: block; font-size: 0.85rem; margin-bottom: 0.15rem; }
  :global(.ficha i) { display: block; color: #9ca3af; font-style: normal; }
  :global(.ficha .sec) { display: block; color: #6b7280; font-size: 0.7rem; }
  :global(.ficha .um) { display: block; color: #f87171; margin-top: 0.2rem; }
  :global(.ficha hr) { border: none; border-top: 1px solid #2a3040; margin: 0.4rem 0 0.3rem; }
  :global(.ficha .tit) {
    display: block; color: #9ca3af; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.15rem;
  }
  :global(.ficha .ant) { display: block; color: #a8a29e; font-size: 0.72rem; }

  .mapa {
    position: absolute;
    inset: 0;
    background: #0b0e14;
  }
</style>
