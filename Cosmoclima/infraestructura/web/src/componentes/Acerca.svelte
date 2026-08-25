<script>
  /**
   * ACERCA DE · el modelo, sus fuentes, sus límites y su vocabulario.
   *
   * ★ Va en una pestaña y no encima del mapa a propósito. Esta página es un
   *   instrumento; un mapa que hay que atravesar leyendo tres pantallas de texto
   *   deja de usarse. Aquí está todo lo que hace falta para juzgar si creerle a
   *   un número, y en el mapa queda sólo lo que hay que ver sí o sí.
   */
  let { datos } = $props();

  let seccion = $state('modelo');

  const medidos = $derived(
    Object.values(datos.afectacion?.por_item ?? {}).filter((v) => v.tipo === 'medido'),
  );
  const expuestos = $derived(
    Object.values(datos.afectacion?.por_item ?? {}).filter((v) => v.tipo === 'expuesto'),
  );
  const itemsTotales = $derived(datos.matriz.items.length);
  const itemsUbicados = $derived(
    new Set(Object.values(datos.activos.por_comuna ?? {}).flatMap((o) => Object.keys(o))).size,
  );
  const activos = $derived(datos.activos.total_indexado ?? 0);

  // ★ Cobertura y acoplamiento se cargan sólo si se abre su pestaña: son 42 KB
  //   que no tiene por qué pagar quien viene a mirar el mapa.
  let cobertura = $state(null);
  let acople = $state(null);
  $effect(() => {
    if (seccion !== 'cobertura' || cobertura) return;
    fetch('datos/cobertura.json').then((r) => r.json()).then((d) => (cobertura = d));
    fetch('datos/acoplamiento.json').then((r) => r.json()).then((d) => (acople = d));
  });
  const sectores = $derived(
    Object.entries(cobertura?.por_sector ?? {})
      .map(([s, v]) => ({ s, ...v }))
      .sort((a, b) => b.faltan - a.faltan),
  );

  const GLOSARIO = [
    ['MICR', 'Matriz de Infraestructura Crítica y Riesgo. El inventario que ordena la infraestructura del país en 846 ítems y 20 sectores, y le asigna a cada uno su nivel de riesgo.'],
    ['IRMD', 'Índice de Riesgo de la Matriz de Decisión. La calificación de riesgo que la Matriz le da a cada ítem: Alto, Medio o Bajo.'],
    ['SENAPRED', 'Servicio Nacional de Prevención y Respuesta ante Desastres. El organismo que emite las alertas en Chile. Esta página no las emite: le entrega insumo.'],
    ['MOP', 'Ministerio de Obras Públicas. De su registro de vías cortadas salen los umbrales medidos.'],
    ['CIGIDEN', 'Centro de Investigación para la Gestión Integrada del Riesgo de Desastres. Aporta el catastro histórico de eventos con fecha, lugar y la lluvia que los provocó.'],
    ['ERA5-Land', 'El reanálisis climático europeo de Copernicus, con una malla de unos 9 km. Es la fuente de los 36 años de historia de lluvia.'],
    ['ONI', 'Oceanic Niño Index. El indicador de la NOAA que define cuándo hay El Niño o La Niña. Son las bandas de color del gráfico.'],
    ['Acumulado 72 h', 'La lluvia caída en los tres días previos, sumada. Es la ventana con que se midieron los umbrales, y por eso todo se compara en ella.'],
    ['p75', 'El cuartil alto. En un territorio con muchas celdas, el valor que sólo supera una de cada cuatro: sirve para ver cuánta diferencia interna hay.'],
    ['Umbral local', 'La lluvia con que un elemento cede EN ESE LUGAR. Se obtiene tomando el percentil que su umbral medido ocupa donde fue medido, y leyéndolo en los 36 años de esa celda concreta.'],
    ['Celda', 'Un cuadrado de 0,1° de lado —unos 9 km— de la malla climática. Todo lo que cae dentro comparte la misma lluvia estimada.'],
  ];
</script>

<nav class="sub">
  {#each [['modelo', 'El modelo'], ['fuentes', 'Fuentes'], ['limites', 'Límites'], ['cobertura', 'Cobertura'], ['glosario', 'Glosario']] as [k, t]}
    <button class:activa={seccion === k} onclick={() => (seccion = k)}>{t}</button>
  {/each}
</nav>

{#if seccion === 'modelo'}
  <h2>Qué hace esta página</h2>
  <p class="entrada">
    Chile sabe dónde está la amenaza y sabe qué infraestructura importa. Lo que
    faltaba era cruzar las dos cosas. Eso es todo lo que hace esto, en cuatro
    pasos:
  </p>

  <ol class="pasos">
    <li>
      <b>Dónde está la infraestructura.</b>
      {activos.toLocaleString('es-CL')} activos georreferenciados —hospitales,
      escuelas, puentes, torres, plantas de agua— repartidos en las 345 comunas.
    </li>
    <li>
      <b>Cuánta lluvia viene.</b>
      El pronóstico de {datos.pronostico.dias} días para las
      {Object.keys(datos.pronostico.celdas).length.toLocaleString('es-CL')} celdas
      que cubren el país, y 36 años de historia para saber si eso es mucho
      <i>ahí</i>.
    </li>
    <li>
      <b>Con cuánta lluvia cedió antes esto mismo.</b>
      En el temporal de julio de 2026 se cortaron 1.289 tramos de vía con fecha y
      kilómetro. De ahí sale que una carretera cede con unos 109 mm en 72 horas y
      un puente aguanta hasta 135.
    </li>
    <li>
      <b>Cuántas veces se cortó algo cuando llovió así.</b>
      Como se conocen todos los días del período —los que cortaron y los que
      no—, se puede decir una frecuencia y no una impresión: sobre 100 mm en 72
      horas <b>se cortó una vía en 1 de cada 4 días-celda</b>; bajo 50 mm, en 1
      de cada 20.
    </li>
  </ol>

  <div class="caja">
    <b>Lo que la distingue de un mapa del tiempo</b> es el cuarto paso. Un
    pronóstico dice cuánta agua cae. Esto dice qué se ha roto antes cuando cayó
    esa agua, y con qué frecuencia.
  </div>
{:else if seccion === 'fuentes'}
  <h2>De dónde sale cada número</h2>
  <p class="entrada">
    Ninguna cifra de esta página es una estimación propia sin origen. Cada una
    viene de un registro público y se puede rastrear hasta él.
  </p>

  <table>
    <thead><tr><th>Dato</th><th>Fuente</th></tr></thead>
    <tbody>
      <tr><td>Lluvia pronosticada, {datos.pronostico.dias} días</td><td>{datos.pronostico.fuente}</td></tr>
      <tr><td>Lluvia histórica 1990-2026</td><td>ERA5-Land 0,1° · Copernicus (Unión Europea)</td></tr>
      <tr><td>El Niño / La Niña</td><td>Índice ONI de la NOAA (Estados Unidos)</td></tr>
      <tr><td>Umbrales de falla</td><td>Vías afectadas del MOP, temporal jul-2026 · catastro de CIGIDEN</td></tr>
      <tr><td>Puntos críticos</td><td>SENAPRED, 15.799 puntos a resolución de calle</td></tr>
      <tr><td>Infraestructura</td><td>Matriz de Infraestructura Crítica y Riesgo (MICR)</td></tr>
      <tr><td>Comunas y límites</td><td>División político-administrativa oficial</td></tr>
      <tr><td>Calles del mapa</td><td>OpenStreetMap · CARTO</td></tr>
    </tbody>
  </table>

  <h3>Los umbrales medidos, uno por uno</h3>
  <p class="entrada">
    Sólo {medidos.length} de los {itemsUbicados} ítems con ubicación tienen umbral
    medido. Cada uno declara sobre cuántos casos reales se calculó — y con nueve
    casos no se puede afirmar lo mismo que con quinientos setenta.
  </p>
  <ul class="umbrales">
    {#each medidos.slice().sort((a, b) => b.umbral_mm_72h - a.umbral_mm_72h) as m}
      <li>
        <span class="el">{m.elemento}</span>
        <span class="mm">{m.umbral_mm_72h} mm/72 h</span>
        <span class="orig">{m.origen} · confianza {m.confianza}</span>
      </li>
    {/each}
  </ul>
{:else if seccion === 'limites'}
  <h2>Lo que esta página no puede decir</h2>
  <p class="entrada">
    Están todos juntos a propósito. Repartidos entre notas al pie, cada uno se
    lee como un detalle que se puede saltar; juntos son la única forma de usar
    esto sin equivocarse.
  </p>

  <ol class="limites">
    <li>
      <b>No es una alerta.</b> Las alertas las emite SENAPRED. Aquí se cruza
      infraestructura registrada con lluvia pronosticada, y nada de lo que se
      muestra reemplaza esa función.
    </li>
    <li>
      <b>El catastro cubre el {(100 * itemsUbicados / itemsTotales).toFixed(1)} % de la Matriz.</b>
      Sólo {itemsUbicados} de {itemsTotales} ítems tienen ubicación. Ocho sectores
      completos —Nuclear, Químico, Financiero, Industrial, Defensa,
      Comunicaciones, Alimentario y Tecnologías Informáticas— no tienen ni un
      activo georreferenciado. Que no aparezcan no significa que estén a salvo.
    </li>
    <li>
      <b>«Afectado» y «expuesto» no son lo mismo.</b>
      {medidos.length} ítems tienen umbral medido y de ellos se puede decir que
      ceden. Los otros {expuestos.length} sólo admiten decir que va a llover mucho
      encima: nadie ha medido nunca con cuánta lluvia falla una torre de
      telecomunicaciones o una escuela.
    </li>
    <li>
      <b>Una frecuencia no es una probabilidad.</b> «1 de cada 4» significa que
      en 1 de cada 4 días-celda con esa lluvia se cortó <i>alguna</i> vía, medido
      sobre 18 días de un temporal en seis regiones. No es la probabilidad de que
      se corte una calle concreta.
    </li>
    <li>
      <b>La fecha del registro no es la del daño.</b> Medido: los tramos que
      figuran cortados sin lluvia tienen su temporal entre 3 y 9 días antes, con
      mediana de 4. El Ministerio publica la fecha del informe, no la del corte.
    </li>
    <li>
      <b>Cercanía no es identidad.</b> Que un punto crítico esté a 200 m de una
      escuela no prueba que la escuela se inunde: prueba que el sector tiene
      antecedentes. Por eso cada uno muestra su distancia.
    </li>
    <li>
      <b>El umbral es local, no nacional.</b> Un mismo milímetro no significa lo
      mismo en Arica —donde llueve 1 mm al año— que en Valdivia. Medido sobre
      los 1.241 tramos cortados en julio de 2026: los milímetros que cortan
      varían <b>4,1 veces</b> entre zonas, mientras el percentil local se mueve
      dentro de una banda de <b>0,46 puntos</b> — todos los cortes del país
      ocurrieron sobre el percentil 99,5 de su propia celda. Por eso el umbral
      de cada elemento se traduce al percentil que ocupa donde fue medido y se
      lee de vuelta en cada lugar: una carretera cede con ~23 mm en el norte
      árido y con ~123 en el sur. Con el umbral único se detectaba el 73 % de
      los cortes reales y <b>ninguno</b> de los del norte; con el local, el 88 %.
    </li>
    <li>
      <b>Las islas no tienen historia.</b> ERA5-Land no cubre las islas oceánicas
      pequeñas —su malla las descarta—, así que Juan Fernández, Pascua y
      Desventuradas tienen pronóstico pero no serie histórica.
    </li>
    <li>
      <b>Sólo lluvia.</b> La Matriz reconoce además sismo, viento, nieve y
      marejada, pero este trabajo únicamente ha calibrado precipitación. Las
      demás amenazas están declaradas y sin medir.
    </li>
  </ol>
{:else if seccion === 'cobertura'}
  <h2>Qué parte de la Matriz está cubierta</h2>
  <p class="entrada">
    La Matriz tiene <b>{cobertura?.total_items ?? 846} ítems</b> y este proyecto
    ha ubicado activos en <b>{cobertura?.con_activos ?? 0}</b>. Leído así parece un
    fracaso. Pero ese número mezcla dos cosas muy distintas, y separarlas cambia
    qué trabajo queda por hacer.
  </p>

  {#if cobertura}
    <table>
      <thead>
        <tr><th>Qué clase de cosa nombra el ítem</th><th>Ítems</th><th>Con activos</th></tr>
      </thead>
      <tbody>
        {#each Object.entries(cobertura.clases).sort((a, b) => b[1].items - a[1].items) as [c, v]}
          <tr>
            <td>{c}</td>
            <td class="num">{v.items}</td>
            <td class="num">{v.con_activos}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    <div class="caja">
      <b>Sólo {cobertura.fisicos} de los {cobertura.total_items} ítems nombran algo
      que pueda tener coordenada.</b> Los demás son roles («Personal de TI»),
      programas («Sistemas de Gestión de Contenido»), categorías de riesgo
      («Infraestructura Vulnerable a Ransomware») o contenidos («Artículos de
      Prensa»). No es que falten sus datos: no hay datos que buscar. La cobertura
      real sobre lo que sí es catastrable es
      <b>{cobertura.cobertura_sobre_fisicos} %</b>, no
      {(100 * cobertura.con_activos / cobertura.total_items).toFixed(1)} %.
    </div>

    <h3>Dónde queda trabajo por hacer</h3>
    <p class="entrada">
      Ítems que nombran algo físico y todavía no tienen catastro. Aquí sí se cierra
      el hueco buscando la fuente.
    </p>
    <table>
      <thead>
        <tr><th>Sector</th><th>Físicos</th><th>Con activos</th><th>Faltan</th></tr>
      </thead>
      <tbody>
        {#each sectores as f}
          <tr>
            <td>{f.s}</td>
            <td class="num">{f.fisicos}</td>
            <td class="num">{f.con_activos}</td>
            <td class="num falta">{f.faltan}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    <div class="caja">
      {cobertura.advertencia}
    </div>
  {/if}

  <h3>Qué falla junto con qué</h3>
  {#if acople}
    <p class="entrada">
      Un activo dañado rara vez falla solo. Sobre los
      <b>50.457 eventos de emergencia de SENAPRED (2015-2024)</b> se midió qué
      tipos ocurren el mismo día y en la misma comuna más de lo que el azar
      explicaría. La medida es el <i>lift</i>: 1 es indistinguible del azar, y
      sólo se listan los pares que lo superan.
    </p>
    <table>
      <thead>
        <tr><th>Ocurren juntos</th><th>Casos</th><th>Lift</th></tr>
      </thead>
      <tbody>
        {#each acople.pares.filter((p) => p.lift > 1).slice(0, 8) as p}
          <tr>
            <td>{p.a} <span class="orig">con {p.b}</span></td>
            <td class="num">{p.juntos}</td>
            <td class="num">{p.lift.toFixed(1)}×</td>
          </tr>
        {/each}
      </tbody>
    </table>
    <div class="caja">
      <b>El acoplamiento medido es débil.</b> El par más acoplado llega a
      {acople.pares[0]?.lift.toFixed(1)}×, no a 5 ni a 10. Eso es un resultado, no
      una carencia: sostiene que los eventos se agrupan algo más que por azar, y
      no sostiene una cadena de colapso automática. {acople.advertencia}
    </div>
  {/if}

{:else}
  <h2>Glosario</h2>
  <p class="entrada">
    Las siglas que aparecen en la página, sin dar por sabido nada.
  </p>
  <dl>
    {#each GLOSARIO as [sigla, def]}
      <dt>{sigla}</dt>
      <dd>{def}</dd>
    {/each}
  </dl>
{/if}

<style>
  .sub { display: flex; gap: 2px; margin: -0.2rem 0 1rem; flex-wrap: wrap; }
  .sub button {
    background: none; border: none; border-bottom: 2px solid transparent;
    color: #6b7280; font: inherit; font-size: 0.76rem; padding: 0.25rem 0.5rem;
    cursor: pointer;
  }
  .sub button:hover { color: #9ca3af; }
  .sub button.activa { color: #e5e7eb; border-bottom-color: #c2410c; }

  h2 { margin: 0 0 0.5rem; font-size: 1.1rem; }
  h3 { margin: 1.4rem 0 0.4rem; font-size: 0.78rem; text-transform: uppercase;
       letter-spacing: 0.06em; color: #9ca3af; }
  .entrada { color: #9ca3af; font-size: 0.84rem; line-height: 1.6; margin: 0 0 0.9rem; }
  b { color: #e5e7eb; }

  ol.pasos, ol.limites { margin: 0; padding-left: 1.1rem; }
  ol.pasos li, ol.limites li {
    color: #9ca3af; font-size: 0.83rem; line-height: 1.6; margin-bottom: 0.8rem;
  }
  ol.pasos li::marker, ol.limites li::marker { color: #c2410c; font-weight: 700; }

  .caja {
    margin-top: 1rem; padding: 0.7rem 0.85rem; background: #12151d;
    border-left: 3px solid #c2410c; font-size: 0.82rem; line-height: 1.6; color: #9ca3af;
  }

  table { width: 100%; border-collapse: collapse; font-size: 0.79rem; }
  th {
    text-align: left; color: #6b7280; font-weight: 600; font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.05em; padding-bottom: 0.3rem;
  }
  td { padding: 0.3rem 0; border-top: 1px solid #1f2430; color: #9ca3af; vertical-align: top; }
  td:first-child { color: #e5e7eb; width: 42%; padding-right: 0.6rem; }

  ul.umbrales { list-style: none; margin: 0; padding: 0; }
  ul.umbrales li { padding: 0.4rem 0; border-top: 1px solid #1f2430; font-size: 0.8rem; }
  .el { color: #e5e7eb; }
  .mm { color: #f87171; margin-left: 0.4rem; }
  .orig { display: block; color: #6b7280; font-size: 0.72rem; margin-top: 0.1rem; }

  td.num { color: #9ca3af; width: auto; text-align: right;
           font-variant-numeric: tabular-nums; }
  td.falta { color: #f87171; }
  th:not(:first-child) { text-align: right; }

  dl { margin: 0; }
  dt { color: #e5e7eb; font-size: 0.82rem; font-weight: 600; margin-top: 0.7rem; }
  dd { margin: 0.15rem 0 0; color: #9ca3af; font-size: 0.8rem; line-height: 1.55; }
</style>
