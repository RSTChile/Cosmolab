<script>
  /**
   * LA CONSULTA · «¿qué puentes pueden verse afectados en esta región?»
   *
   * Es la pregunta que le da sentido a todo lo demás, y la respuesta se da en
   * TRES NIVELES DE EVIDENCIA, no en un número único:
   *
   *   1. los que YA fallaron, con nombre, fecha y la lluvia que los tumbó
   *   2. los que están catalogados como punto crítico, sin lluvia asociada
   *   3. los que sólo comparten tipo con los que fallaron
   *
   * Mezclarlos daría una cifra más grande y más impresionante que estaría
   * mintiendo: no es lo mismo un puente que ya se cortó que uno que sólo se
   * parece a otros que se cortaron.
   */
  import { fechaCorta } from '../lib/fechas.js';
  import { colorSector } from '../lib/sectores.js';
  import { cargarMCSGS, sincronizacion, leerFSS } from '../lib/mcsgs.js';

  let {
    datos, mmPorComuna, evaluar = null, dia = 0,
    ruta5Traza = $bindable(null), ruta5Riesgo = $bindable(null),
  } = $props();

  let evidencia = $state(null);
  let cargando = $state(false);
  let item = $state('618');          // puentes: el ejemplo que originó esto
  let ambito = $state('CL');
  let ruta5 = $state(null);
  let mcsgs = $state(null);

  // ★★ El MCSGS mira lo que la Matriz no: no cuántos activos fallan, sino si
  //   fallan A LA VEZ. Se carga aparte porque sólo se usa en esta vista.
  $effect(() => {
    if (!mcsgs) cargarMCSGS().then((d) => (mcsgs = d));
  });

  const sinc = $derived(
    mcsgs && comunas.length ? sincronizacion(datos, mcsgs, mmPorComuna, evaluar, comunas) : null,
  );
  const lectura = $derived(sinc ? leerFSS(sinc.fss) : null);

  // ★ La Ruta 5 es un caso aparte porque NO ESTÁ en el catastro de la Matriz:
  //   de sus 14.036 activos viales, 1.522 la nombran («Cruce Ruta 5 …») y sólo
  //   7 empiezan por ella. Se trae de OpenStreetMap y se consulta como si fuera
  //   un elemento más, declarando que viene de otro lado.
  $effect(() => {
    if (item === 'R5' && !ruta5) {
      fetch('datos/ruta5.json').then((r) => (r.ok ? r.json() : null)).then((d) => (ruta5 = d));
    }
  });

  const tramosRuta5 = $derived.by(() => {
    if (item !== 'R5' || !ruta5) return null;
    const cuts = new Set(comunas.map((c) => c.cut));
    const nombre = new Map(comunas.map((c) => [c.cut, c.comuna]));
    return ruta5.segmentos
      .filter((t) => cuts.has(t.cut))
      .map((t) => {
        const mm = mmPorComuna.get(t.cut);
        const e = evaluar ? evaluar(mm, '616', t.cut) : null;
        return { ...t, comuna: nombre.get(t.cut), mm, ev: e };
      })
      .filter((t) => t.ev?.estado === 'afectado');
  });

  $effect(() => {
    if (!evidencia && !cargando) {
      cargando = true;
      fetch('datos/evidencia.json')
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => { evidencia = d; cargando = false; })
        .catch(() => { cargando = false; });
    }
  });

  // ── qué elementos se pueden consultar ────────────────────────────────────
  const conActivos = $derived.by(() => {
    const cuenta = new Map();
    for (const idx of Object.values(datos.activos.por_comuna ?? {})) {
      for (const [n, c] of Object.entries(idx)) cuenta.set(n, (cuenta.get(n) ?? 0) + c);
    }
    const porN = new Map(datos.matriz.items.map((i) => [String(i.n), i]));
    return [...cuenta.entries()]
      .map(([n, total]) => ({ n, total, item: porN.get(n) }))
      .filter((x) => x.item)
      .sort((a, b) => b.total - a.total);
  });

  // ── las comunas del ámbito elegido ───────────────────────────────────────
  const comunas = $derived.by(() => {
    const t = datos.territorios.comunas;
    if (ambito === 'CL') return t;
    const c = ambito.slice(1);
    if (ambito[0] === 'R') return t.filter((x) => x.cut_reg === c);
    if (ambito[0] === 'P') return t.filter((x) => x.cut_prov === c);
    return t.filter((x) => x.cut === c);
  });

  const af = $derived(datos.afectacion?.por_item ?? {});
  const umbral = $derived(af[item]?.tipo === 'medido' ? af[item].umbral_mm_72h : null);
  const elemento = $derived(conActivos.find((x) => x.n === item)?.item ?? null);

  /** El resultado, en sus tres niveles. */
  const resultado = $derived.by(() => {
    const cuts = new Set(comunas.map((c) => c.cut));
    const nombreCom = new Map(comunas.map((c) => [c.cut, c.comuna]));
    let cruzan = 0, total = 0, sinCobertura = 0;

    const locales = [];
    for (const c of comunas) {
      const n = datos.activos.por_comuna?.[c.cut]?.[item] ?? 0;
      if (!n) continue;
      total += n;
      const mm = mmPorComuna.get(c.cut);
      if (mm == null) { sinCobertura += n; continue; }
      const e = evaluar ? evaluar(mm, item, c.cut) : null;
      if (e?.umbral != null && e.escala === 'local') locales.push(e.umbral);
      if (e ? (e.estado === 'afectado' || e.estado === 'expuesto')
            : (umbral != null ? mm >= umbral : mm >= 50)) cruzan += n;
    }
    locales.sort((a, b) => a - b);
    const umbralLocalMediano = locales.length ? locales[Math.floor(locales.length / 2)] : null;

    // los que tienen evidencia propia, dentro del ámbito y que además cruzan hoy
    const nombrados = [];
    for (const [cut, lista] of Object.entries(evidencia?.por_comuna ?? {})) {
      if (!cuts.has(cut)) continue;
      const mm = mmPorComuna.get(cut);
      if (mm == null) continue;
      for (const e of lista) {
        if (e.n !== item) continue;
        const ev = evaluar ? evaluar(mm, item, cut) : null;
        if (ev ? !(ev.estado === 'afectado' || ev.estado === 'expuesto')
               : (umbral != null ? mm < umbral : mm < 50)) continue;
        nombrados.push({ ...e, comuna: nombreCom.get(cut), mmHoy: mm });
      }
    }
    // primero los que cayeron con MENOS lluvia que la pronosticada: si aguantó
    // menos que lo que viene, es el que hay que mirar primero.
    nombrados.sort((a, b) => a.mm - b.mm);
    return { total, cruzan, sinCobertura, nombrados, umbralLocalMediano };
  });

  const fecha = $derived(fechaCorta(datos.pronostico.fechas?.[dia]));

  // El mapa dibuja; aquí se decide qué. Se sube por binding en vez de repetir
  // el cálculo en el componente del mapa.
  $effect(() => { ruta5Traza = item === 'R5' ? ruta5 : null; });
  $effect(() => { ruta5Riesgo = tramosRuta5; });
</script>

<div class="consulta">
  <div class="pregunta">
    <span>¿Qué</span>
    <select bind:value={item}>
      <option value="R5">Ruta 5 · Panamericana</option>
      {#each conActivos as x}
        <option value={x.n}>{x.item.elemento}</option>
      {/each}
    </select>
    <span>puede verse afectado en</span>
    <select bind:value={ambito}>
      <option value="CL">todo Chile</option>
      <optgroup label="Regiones">
        {#each datos.territorios.regiones as r}
          <option value={`R${r.cut}`}>{r.nombre}</option>
        {/each}
      </optgroup>
      <optgroup label="Provincias">
        {#each datos.territorios.provincias as p}
          <option value={`P${p.cut}`}>{p.nombre}</option>
        {/each}
      </optgroup>
      <optgroup label="Comunas">
        {#each datos.territorios.comunas as c}
          <option value={`C${c.cut}`}>{c.comuna}</option>
        {/each}
      </optgroup>
    </select>
    <span>al {fecha}?</span>
  </div>

  {#if item === 'R5'}
    {#if !ruta5}
      <p class="nota">Cargando la traza de la Ruta 5…</p>
    {:else}
      {@const total = ruta5.segmentos.filter((t) => comunas.some((c) => c.cut === t.cut)).length}
      <p class="titular">
        <strong style="color: #f59e0b">{(tramosRuta5?.length ?? 0)}</strong>
        de {total} tramos de ~5 km superan el umbral de carpeta de rodadura
        <b>en su propia zona</b>.
      </p>
      {#if tramosRuta5?.length}
        <h4>Dónde</h4>
        <ul>
          {#each Object.entries(
            tramosRuta5.reduce((a, t) => ((a[t.comuna] = a[t.comuna] ?? []).push(t), a), {}),
          ).sort((a, b) => b[1].length - a[1].length).slice(0, 20) as [com, ts]}
            <li>
              <div class="nom">{com}</div>
              <div class="det">
                {ts.length} tramo{ts.length === 1 ? '' : 's'} ·
                hasta <b>{Math.max(...ts.map((t) => t.mm)).toFixed(0)} mm</b> en 72 h
              </div>
            </li>
          {/each}
        </ul>
      {/if}
      <p class="nota">
        ⚠️ La Ruta 5 <b>no está en el catastro de la Matriz</b>: de sus 14.036
        activos viales, 1.522 la nombran como referencia y sólo 7 empiezan por
        ella. Esta traza viene de OpenStreetMap, y el umbral es el de carpeta de
        rodadura, medido sobre 570 tramos del temporal de julio.
      </p>
    {/if}
  {:else if !elemento}
    <p class="nota">Este elemento no tiene activos georreferenciados.</p>
  {:else}
    <p class="titular">
      {#if resultado.cruzan}
        <strong style="color: {colorSector(elemento.sector)}">
          {resultado.cruzan.toLocaleString('es-CL')}
        </strong>
        de {resultado.total.toLocaleString('es-CL')}
        {#if resultado.umbralLocalMediano != null}
          superan el umbral <b>de su propia zona</b>
          (mediana <b>{resultado.umbralLocalMediano.toFixed(0)} mm/72 h</b>{#if umbral != null},
          contra {umbral} mm del promedio nacional{/if})
        {:else if umbral != null}
          superan los <b>{umbral} mm/72 h</b> con que este tipo cedió en la realidad
        {:else}
          quedan bajo lluvia intensa (sobre 50 mm/72 h)
        {/if}
      {:else}
        Ninguno de los {resultado.total.toLocaleString('es-CL')} cruza su umbral con la lluvia de este día.
      {/if}
    </p>

    {#if resultado.umbralLocalMediano != null}
      <p class="nota">
        El umbral se calcula <b>para cada lugar</b>: es el percentil que este
        elemento ocupa donde fue medido, leído en los 36 años de esa celda. En el
        norte árido bastan ~23 mm para cortar una carretera y en el sur hacen
        falta ~123. Medido contra los 1.241 cortes de julio, el umbral local
        detecta el 88 % y el nacional único sólo el 73 %.
      </p>
    {/if}

    {#if umbral == null && resultado.total}
      <p class="nota">
        ⚠️ Este elemento <b>no tiene umbral medido</b>: nadie ha registrado con
        cuánta lluvia falla. Lo de arriba dice que va a llover fuerte encima, no
        que vaya a fallar.
      </p>
    {/if}

    {#if cargando}
      <p class="nota">Buscando antecedentes…</p>
    {:else if resultado.nombrados.length}
      <h4>Estos ya fallaron antes, y con cuánta lluvia</h4>
      <ul>
        {#each resultado.nombrados.slice(0, 25) as e}
          <li class:menos={e.mm <= e.mmHoy}>
            <div class="nom">{e.a || '(sin nombre en el catastro)'}</div>
            <div class="det">
              {e.comuna} ·
              {#if e.t === 'via'}
                se cortó el {fechaCorta(e.f)}{e.g ? ` (${e.g})` : ''}
              {:else}
                {e.g || 'evento'} en {fechaCorta(e.f).slice(3)}
              {/if}
              tras un episodio de <b>{e.mm.toFixed(0)} mm</b>
              {#if e.mm <= e.mmHoy}
                — <em>hoy se pronostican {e.mmHoy.toFixed(0)}</em>
              {/if}
              <span class="dist">a {e.d} m</span>
            </div>
          </li>
        {/each}
      </ul>
      {#if resultado.nombrados.length > 25}
        <p class="nota">y {resultado.nombrados.length - 25} más con antecedente propio.</p>
      {/if}
      <p class="nota">
        Los destacados cedieron con <b>igual o menos lluvia</b> que la
        pronosticada para este día. El antecedente está a la distancia indicada:
        prueba que el sector tiene historia, no que sea exactamente ese activo.
      </p>
    {:else if resultado.cruzan}
      <p class="nota">
        Ninguno de ellos tiene antecedente propio registrado: lo que se sabe
        viene del tipo de elemento, no de estos activos en particular.
      </p>
    {/if}

    {#if sinc && sinc.nodosConAfectacion}
      <section class="sistemico">
        <h4>Los nodos de flujo, mirados juntos</h4>
        <p class="titular">
          <strong class={lectura.clave}>{(100 * sinc.fss).toFixed(0)} %</strong>
          de la capacidad de flujo del territorio queda comprometida ·
          <em>{lectura.texto}</em>
        </p>
        <p class="nota">
          <b>{sinc.nodosConAfectacion} de {sinc.nodosTotales}</b> tipos de nodo de
          flujo tienen activos cruzando su umbral el mismo día. Un nodo de flujo
          es un paso obligado —un puente, una subestación, un tramo sin
          alternativa—: se reconstruye mucho más lento que un edificio, y su
          fallo interrumpe todo lo que pasaba por ahí.
        </p>
        <ul class="nodos">
          {#each sinc.afectados.slice(0, 8) as a}
            <li>
              <span class="nombre">{a.elemento}</span>
              <span class="dato">
                {a.cruzan.toLocaleString('es-CL')} de {a.total.toLocaleString('es-CL')}
                <em class="fcn">criticidad {a.fcn.toFixed(2)}</em>
              </span>
            </li>
          {/each}
        </ul>
        <p class="nota">
          ⚠️ Esto <b>no</b> es el índice de colapso del RMD. Ese índice pide cinco
          factores y aquí hay dos con dato medido —criticidad nodal y
          sincronización—; faltan acoplamiento, resiliencia y propagación.
          Componerlo igual daría un número con apariencia de medición.
        </p>
      </section>
    {/if}

    {#if resultado.sinCobertura}
      <p class="nota">
        {resultado.sinCobertura.toLocaleString('es-CL')} sin cobertura climática.
      </p>
    {/if}
  {/if}
</div>

<style>
  .consulta {
    border-bottom: 1px solid #1f2430;
    padding: 0.7rem 0 0.9rem;
    margin-bottom: 1rem;
  }
  .pregunta {
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem;
    font-size: 0.82rem; color: #9ca3af; line-height: 1.9;
  }
  select {
    background: #12151d; color: #e5e7eb; border: 1px solid #2a3040;
    border-radius: 5px; padding: 0.15rem 0.3rem; font: inherit;
    font-size: 0.78rem; max-width: 195px;
  }
  .titular { margin: 0.6rem 0 0; font-size: 0.88rem; color: #9ca3af; line-height: 1.5; }
  .titular strong { font-size: 1.5rem; font-weight: 700; }
  .titular b { color: #e5e7eb; }
  h4 {
    margin: 0.9rem 0 0.3rem; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #9ca3af; font-weight: 600;
  }
  ul { list-style: none; margin: 0; padding: 0; }
  li { padding: 0.35rem 0; border-top: 1px solid #1f2430; }
  li.menos { border-left: 2px solid #f87171; padding-left: 0.5rem; }
  .nom { color: #e5e7eb; font-size: 0.83rem; }
  .det { color: #6b7280; font-size: 0.74rem; line-height: 1.45; }
  .det b { color: #fca5a5; }
  .det em { color: #fbbf24; font-style: normal; }
  .dist { margin-left: 0.3rem; opacity: 0.7; }
  .sistemico {
    margin-top: 1.2rem; padding-top: 0.9rem; border-top: 1px solid #2a3040;
  }
  .sistemico .titular { margin: 0.2rem 0 0; }
  .sistemico strong { font-size: 1.6rem; }
  .sistemico strong.alta { color: #f87171; }
  .sistemico strong.media { color: #fbbf24; }
  .sistemico strong.baja { color: #60a5fa; }
  .sistemico strong.nula { color: #6b7280; }
  .sistemico em { color: #9ca3af; font-style: normal; font-size: 0.82rem; }
  .nodos { list-style: none; margin: 0.5rem 0 0; padding: 0; }
  .nodos li {
    display: flex; justify-content: space-between; gap: 1rem;
    padding: 0.28rem 0; border-top: 1px solid #1f2430; font-size: 0.8rem;
  }
  .fcn { color: #6b7280; font-size: 0.7rem; margin-left: 0.4rem; font-style: normal; }
  .nota { margin: 0.45rem 0 0; color: #6b7280; font-size: 0.75rem; line-height: 1.5; }
  .nota b { color: #9ca3af; }
</style>
