<script>
  import { cargarHistoria, cargarDiaria, arbolTerritorios, umbralesDibujables } from '../lib/historia.js';
  import { fechaCorta } from '../lib/fechas.js';

  let { datos, cut = null, abierto = $bindable(true) } = $props();

  let hist = $state(null);
  let cargando = $state(false);
  let clave = $state('CL');
  let estad = $state('mediana');
  let diaria = $state(null);      // serie diaria de la comuna, si se pidió
  let modo = $state('mensual');
  // ★★ `$state` y no `let` a secas: con `bind:this`, Svelte 5 sólo vuelve a
  //   ejecutar el efecto de dibujo si la variable es reactiva. Sin esto el
  //   canvas se creaba pero `pintar()` no corría nunca — quedaba en su tamaño
  //   por defecto de 300×150 y en blanco, con TODOS los datos ya cargados. Un
  //   fallo silencioso perfecto: ni un error en consola.
  let lienzo = $state(null);
  let ancho = $state(900);
  // Alto generoso: comprimido en 190 px las curvas se aplastan y no se
  // distingue un temporal de un invierno normal.
  const alto = 420;

  // ventana visible, en índices de la serie
  let desde = $state(0);
  let hasta = $state(1);
  let arrastrando = false;
  let xArrastre = 0;
  let cursor = $state(null);

  // ★ La carga se dispara al ABRIR, no al montar: son 2,82 MB que quien sólo
  //   quiere ver el mapa no tiene por qué pagar.
  $effect(() => {
    if (abierto && !hist && !cargando) {
      cargando = true;
      cargarHistoria().then((h) => {
        hist = h;
        cargando = false;
        if (h) { desde = 0; hasta = h.meses.length - 1; }
      });
    }
  });

  // Seguir la comuna elegida en el mapa: si estás mirando Ovalle, el gráfico
  // debería hablar de Ovalle sin que haya que buscarla otra vez en el selector.
  $effect(() => {
    if (cut && hist?.territorios[`C${cut}`]) clave = `C${cut}`;
  });

  // Al cambiar de territorio o de modo, la ventana vuelve a la serie completa.
  $effect(() => {
    const n = serie?.valores?.length ?? 0;
    if (n) { desde = 0; hasta = n - 1; }
  });

  const arbol = $derived(hist ? arbolTerritorios(datos, hist) : null);
  const territorio = $derived(hist?.territorios?.[clave] ?? null);

  const serie = $derived.by(() => {
    if (!hist || !territorio) return null;
    if (modo === 'diaria' && diaria) {
      const d0 = new Date(diaria.desde + 'T12:00:00');
      return {
        valores: diaria.mm,
        etiqueta: (i) => {
          const d = new Date(d0);
          d.setDate(d.getDate() + i);
          return fechaCorta(d.toISOString().slice(0, 10));
        },
        unidad: 'mm/día',
      };
    }
    return {
      valores: territorio[estad],
      etiqueta: (i) => {
        const m = hist.meses[i];
        return m ? `${m.slice(5)}-${m.slice(2, 4)}` : '';
      },
      unidad: 'mm/mes',
    };
  });

  const umbrales = $derived(modo === 'diaria' ? umbralesDibujables(datos) : []);

  function pedirDiaria() {
    if (!clave.startsWith('C') || clave === 'CL') return;
    cargarDiaria(clave.slice(1)).then((d) => {
      diaria = d;
      modo = d ? 'diaria' : 'mensual';
    });
  }

  // ── dibujo ────────────────────────────────────────────────────────────────
  function pintar() {
    if (!lienzo || !serie) return;
    const dpr = window.devicePixelRatio || 1;
    const W = ancho, H = alto;
    lienzo.width = W * dpr;
    lienzo.height = H * dpr;
    lienzo.style.width = W + 'px';
    lienzo.style.height = H + 'px';
    const c = lienzo.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);

    const ML = 48, MR = 12, MT = 14, MB = 24;
    const w = W - ML - MR, h = H - MT - MB;
    const v = serie.valores;
    const n = Math.max(1, hasta - desde + 1);
    const px = (i) => ML + ((i - desde) / n) * w;

    let tope = 0;
    for (let i = desde; i <= hasta; i++) tope = Math.max(tope, v[i] ?? 0);
    tope = Math.max(tope, 10) * 1.1;
    const py = (mm) => MT + h - (mm / tope) * h;

    // ── bandas de El Niño y La Niña, al fondo ──────────────────────────────
    if (modo === 'mensual' && hist.enso?.length) {
      for (const b of hist.enso) {
        const i0 = hist.meses.findIndex((m) => m >= b.desde.slice(0, 7));
        const i1 = hist.meses.findIndex((m) => m > b.hasta.slice(0, 7));
        if (i0 < 0) continue;
        const a = Math.max(i0, desde), z = Math.min(i1 < 0 ? v.length - 1 : i1, hasta);
        if (z <= a) continue;
        // Rojo el Niño, azul la Niña. Muy tenues: son contexto, no el dato.
        c.fillStyle = b.tipo === 'nino' ? 'rgba(220,70,50,0.13)' : 'rgba(70,130,220,0.13)';
        c.fillRect(px(a), MT, px(z) - px(a), h);
      }
    }

    // ── rejilla ────────────────────────────────────────────────────────────
    c.strokeStyle = '#1f2430';
    c.fillStyle = '#6b7280';
    c.font = '10px ui-sans-serif, system-ui, sans-serif';
    c.lineWidth = 1;
    for (let k = 0; k <= 5; k++) {
      const mm = (tope / 5) * k, y = Math.round(py(mm)) + 0.5;
      c.beginPath(); c.moveTo(ML, y); c.lineTo(W - MR, y); c.stroke();
      c.textAlign = 'right'; c.textBaseline = 'middle';
      c.fillText(Math.round(mm), ML - 5, y);
    }

    // ── las barras ─────────────────────────────────────────────────────────
    const ancho1 = Math.max(1, w / n);
    for (let i = desde; i <= hasta; i++) {
      const mm = v[i] ?? 0;
      if (mm <= 0) continue;
      const y = py(mm);
      c.fillStyle = mm >= tope * 0.6 ? '#60a5fa' : '#3b82f6';
      c.fillRect(px(i), y, Math.max(1, ancho1 * 0.85), MT + h - y);
    }

    // ── umbrales medidos, sólo en diario ───────────────────────────────────
    for (const u of umbrales) {
      if (u.mm > tope) continue;
      const y = Math.round(py(u.mm)) + 0.5;
      c.strokeStyle = 'rgba(248,113,113,0.55)';
      c.setLineDash([4, 3]);
      c.beginPath(); c.moveTo(ML, y); c.lineTo(W - MR, y); c.stroke();
      c.setLineDash([]);
      c.fillStyle = 'rgba(248,113,113,0.8)';
      c.textAlign = 'left'; c.textBaseline = 'bottom';
      c.fillText(`${u.elemento.slice(0, 22)} ${u.mm}`, ML + 4, y - 1);
    }

    // ── eje inferior ───────────────────────────────────────────────────────
    c.fillStyle = '#6b7280';
    c.textAlign = 'center'; c.textBaseline = 'top';
    const salto = Math.max(1, Math.round(n / 12));
    for (let i = desde; i <= hasta; i += salto) {
      c.fillText(serie.etiqueta(i), px(i), MT + h + 4);
    }

    // ── cursor ─────────────────────────────────────────────────────────────
    if (cursor != null && cursor >= desde && cursor <= hasta) {
      const x = px(cursor) + ancho1 / 2;
      c.strokeStyle = '#e5e7eb55';
      c.beginPath(); c.moveTo(x, MT); c.lineTo(x, MT + h); c.stroke();
    }
  }

  $effect(() => { pintar(); });

  // ── interacción ───────────────────────────────────────────────────────────
  function alRodar(e) {
    if (!serie) return;
    e.preventDefault();
    const n = hasta - desde + 1;
    const r = lienzo.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (e.clientX - r.left - 42) / (r.width - 50)));
    const centro = desde + f * n;
    const factor = e.deltaY > 0 ? 1.25 : 0.8;
    let nn = Math.round(n * factor);
    nn = Math.min(serie.valores.length, Math.max(6, nn));
    let d = Math.round(centro - f * nn);
    d = Math.max(0, Math.min(serie.valores.length - nn, d));
    desde = d;
    hasta = d + nn - 1;
  }

  function alMover(e) {
    if (!serie) return;
    const r = lienzo.getBoundingClientRect();
    const n = hasta - desde + 1;
    if (arrastrando) {
      const dx = e.clientX - xArrastre;
      const salto = Math.round((dx / (r.width - 50)) * n);
      if (salto) {
        let d = desde - salto;
        d = Math.max(0, Math.min(serie.valores.length - n, d));
        desde = d; hasta = d + n - 1;
        xArrastre = e.clientX;
      }
      return;
    }
    const f = (e.clientX - r.left - 42) / (r.width - 50);
    const i = Math.round(desde + f * n);
    cursor = i >= desde && i <= hasta ? i : null;
  }
</script>

<section class="historia" class:abierto>
  <button class="cabecera" onclick={() => (abierto = !abierto)}>
    <span class="flecha">{abierto ? '▾' : '▸'}</span>
    Historia climática
    {#if territorio}
      <em>{territorio.n}</em>
      <span class="celdas">{territorio.celdas} celda{territorio.celdas === 1 ? '' : 's'}</span>
    {/if}
  </button>

  {#if abierto}
    {#if cargando}
      <p class="aviso">Cargando 36 años de serie…</p>
    {:else if !hist}
      <p class="aviso">No se pudo cargar la historia.</p>
    {:else}
      <div class="mandos">
        <select bind:value={clave} onchange={() => { modo = 'mensual'; diaria = null; }}>
          {#if arbol?.pais}<option value={arbol.pais.clave}>Chile</option>{/if}
          <optgroup label="Regiones">
            {#each arbol.regiones as r}<option value={r.clave}>{r.nombre}</option>{/each}
          </optgroup>
          <optgroup label="Provincias">
            {#each arbol.provincias as p}<option value={p.clave}>{p.nombre}</option>{/each}
          </optgroup>
          <optgroup label="Comunas">
            {#each arbol.comunas as c}<option value={c.clave}>{c.nombre}</option>{/each}
          </optgroup>
        </select>

        {#if modo === 'mensual'}
          <!-- ★ El selector de estadístico sólo aparece cuando hay más de una
               celda: con una sola los tres coinciden y ofrecerlo sería sugerir
               una diferencia que no existe. -->
          {#if (territorio?.celdas ?? 1) > 1}
            <div class="segmentos">
              {#each [['mediana', 'típico'], ['p75', 'alto'], ['maximo', 'peor punto']] as [k, t]}
                <button class:activo={estad === k} onclick={() => (estad = k)}>{t}</button>
              {/each}
            </div>
          {/if}
        {/if}

        {#if clave.startsWith('C') && clave !== 'CL'}
          <button class="modo" onclick={() => (modo === 'diaria' ? (modo = 'mensual') : pedirDiaria())}>
            {modo === 'diaria' ? 'ver mensual' : 'ver día a día'}
          </button>
        {/if}

        {#if serie && (hasta - desde + 1) < serie.valores.length}
          <button class="modo" onclick={() => { desde = 0; hasta = serie.valores.length - 1; }}>
            todo
          </button>
        {/if}

        <span class="lectura">
          {#if cursor != null && serie?.valores[cursor] != null}
            {serie.etiqueta(cursor)} · <strong>{serie.valores[cursor].toFixed(1)}</strong> {serie.unidad}
          {:else}
            rueda para acercar · arrastra para desplazar
          {/if}
        </span>
      </div>

      <div class="lienzo" bind:clientWidth={ancho}>
        <canvas
          bind:this={lienzo}
          onwheel={alRodar}
          onmousemove={alMover}
          onmousedown={(e) => { arrastrando = true; xArrastre = e.clientX; }}
          onmouseup={() => (arrastrando = false)}
          onmouseleave={() => { arrastrando = false; cursor = null; }}
        ></canvas>
      </div>

      <p class="pie">
        {#if modo === 'mensual'}
          Lluvia mensual {hist.desde?.slice(0, 4)}–{hist.hasta?.slice(0, 4)} · ERA5-Land 0,1°
          · fondo <i class="nino"></i> El Niño <i class="nina"></i> La Niña (ONI, NOAA)
        {:else}
          Lluvia diaria · las líneas rojas son los umbrales medidos con que cada
          elemento cedió en el temporal de julio 2026
        {/if}
      </p>
    {/if}
  {/if}
</section>

<style>
  .historia {
    border-top: 1px solid #1f2430;
    background: #0b0e14;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .cabecera {
    display: flex; align-items: baseline; gap: 0.5rem;
    background: none; border: none; color: #9ca3af; font: inherit;
    font-size: 0.78rem; padding: 0.45rem 0.9rem; cursor: pointer; text-align: left;
  }
  .cabecera:hover { color: #e5e7eb; }
  .flecha { color: #6b7280; font-size: 0.7rem; }
  .cabecera em { color: #e5e7eb; font-style: normal; font-weight: 600; }
  .celdas { color: #6b7280; font-size: 0.7rem; }
  .mandos {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
    padding: 0 0.9rem 0.4rem;
  }
  select {
    background: #12151d; color: #e5e7eb; border: 1px solid #2a3040;
    border-radius: 5px; padding: 0.2rem 0.4rem; font: inherit; font-size: 0.76rem;
    max-width: 210px;
  }
  .segmentos { display: flex; border: 1px solid #2a3040; border-radius: 5px; overflow: hidden; }
  .segmentos button {
    background: none; border: none; color: #6b7280; font: inherit;
    font-size: 0.72rem; padding: 0.2rem 0.45rem; cursor: pointer;
  }
  .segmentos button.activo { background: #1f2937; color: #e5e7eb; }
  .modo {
    background: #12151d; color: #9ca3af; border: 1px solid #2a3040;
    border-radius: 5px; padding: 0.2rem 0.5rem; font: inherit; font-size: 0.72rem;
    cursor: pointer;
  }
  .modo:hover { color: #e5e7eb; }
  .lectura { color: #6b7280; font-size: 0.72rem; margin-left: auto; }
  .lectura strong { color: #e5e7eb; }
  .lienzo { padding: 0 0.9rem; }
  canvas { display: block; cursor: crosshair; }
  .pie { margin: 0.25rem 0 0.5rem; padding: 0 0.9rem; color: #6b7280; font-size: 0.68rem; }
  .pie i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 0.15rem 0 0.3rem; }
  .pie i.nino { background: rgba(220, 70, 50, 0.5); }
  .pie i.nina { background: rgba(70, 130, 220, 0.5); }
  .aviso { color: #6b7280; font-size: 0.8rem; padding: 0 0.9rem 0.7rem; }
</style>
