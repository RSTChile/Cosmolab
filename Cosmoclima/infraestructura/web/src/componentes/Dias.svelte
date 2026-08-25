<script>
  import { COLORES, nivelDe } from '../lib/riesgo.js';
  import { diaMes, fechaCorta } from '../lib/fechas.js';

  let { pronostico, serie = null, dia = $bindable(0), desdeVigente = 0 } = $props();

  // Cuando no hay comuna elegida, la barra muestra el día pero sin alturas:
  // inventar una serie «promedio nacional» sería un número que no significa nada.
  const acumulados = $derived.by(() => {
    if (!serie) return null;
    return serie.map((_, i) => serie.slice(Math.max(0, i - 2), i + 1).reduce((a, b) => a + b, 0));
  });
  const tope = $derived(acumulados ? Math.max(10, ...acumulados) : 10);

  const corto = diaMes;
</script>

<div class="barra">
  {#each pronostico.fechas as fecha, i}
    {@const mm = acumulados?.[i]}
    <button
      class="dia"
      class:activo={i === dia}
      class:vencido={i < desdeVigente}
      onclick={() => (dia = i)}
      title={i < desdeVigente
        ? `${fechaCorta(fecha)} · día ya transcurrido`
        : mm != null
        ? `${fechaCorta(fecha)} · ${mm.toFixed(1)} mm en 72 h`
        : fechaCorta(fecha)}
    >
      <span class="tallo">
        {#if mm != null}
          <span
            class="relleno"
            style="height: {Math.max(2, (mm / tope) * 100)}%;
                   background: {COLORES[nivelDe(mm).clave]}"
          ></span>
        {/if}
      </span>
      <span class="fecha">{corto(fecha)}</span>
    </button>
  {/each}
</div>

<style>
  .barra {
    display: flex;
    gap: 2px;
    align-items: flex-end;
    height: 100%;
  }
  .dia {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    gap: 3px;
    background: none;
    border: none;
    padding: 0 0 2px;
    cursor: pointer;
    border-radius: 3px;
    min-width: 0;
  }
  .dia:hover { background: #161b26; }
  .dia.activo { background: #1f2937; }
  /* ★ Los días ya transcurridos siguen visibles pero apagados: quitarlos
     desplazaría la barra cada día y se perdería la referencia; dejarlos igual
     que los demás los haría pasar por previsión. */
  .dia.vencido { opacity: 0.32; }
  .dia.vencido .tallo { filter: grayscale(1); }
  .tallo {
    display: flex;
    align-items: flex-end;
    width: 100%;
    height: 42px;
    background: #12151d;
    border-radius: 2px;
    overflow: hidden;
  }
  .relleno { width: 100%; border-radius: 2px 2px 0 0; }
  .fecha {
    font-size: 0.62rem;
    color: #6b7280;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .dia.activo .fecha { color: #e5e7eb; }
</style>
