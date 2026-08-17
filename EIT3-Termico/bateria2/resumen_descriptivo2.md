# Resumen descriptivo — segunda batería EIT-3 Térmico κ_H (v7.5)

Solo estadística descriptiva. Ninguna conclusión sobre si hay o no "hallazgo":
eso es del investigador principal.

## Cobertura

| experimento | barridos | filas | 
|---|---|---|
| D · modo de reinicio (10 semillas × 2 modos) | 20 | 1.200 |
| A' · repetición (30 semillas, modo=parada) | 30 | 1.800 |
| B' · multivariable (108 combinaciones × 10 semillas, modo=parada) | 1.080 | 64.800 |
| C · barajado sobre A' | 30 series × 1.000 barajes | 30 filas de resumen |
| C · barajado sobre B' | 1.080 series × 1.000 barajes | 1.080 filas de resumen |

0 filas con `NaN`/`Infinity` en D, A' y B'. 0 barridos fallidos de 1.130 corridos.
Tiempo real: ~12,7 h con 14 procesos (la máquina entró en throttling térmico
sostenido — `CPU_Speed_Limit` bajó a 35% en algún momento de la corrida — así
que el tiempo no es directamente comparable al de la primera batería).

## Experimento D — el modo de reinicio como factor

| modo | posición del mínimo de huella (media±desv, 10 semillas) | correlación huella↔entropía (media±desv) |
|---|---|---|
| `parada`  | 0,8551 ± ~0 (idéntico en las 10 semillas) | 0,367 ± 0,040 |
| `inicio`  | 1,3449 ± ~0 (idéntico en las 10 semillas) | 0,509 ± 0,024 |

**La predicción del encargo era: en `inicio` la posición NO se mueve, en
`parada` SÍ se mueve. Salió al revés de lo esperado en un sentido preciso:
NINGUNO de los dos modos se mueve entre semillas** — ambos dan una posición
del mínimo idéntica a precisión de punto flotante en las 10 semillas de su
modo (sí difieren ENTRE modos: 0,8551 vs 1,3449). Ver nota técnica en
`defectos_encontrados2.md` sobre por qué esto es esperable dado cómo está
escrito el motor: `computeDaisyworld()` (la función que gobierna el
crecimiento/muerte de `black`/`white`) no llama al generador de azar en
ningún punto — es una recursión determinística sobre luminosidad/tOpt/ruido
únicamente. Eso no decide si hay o no un hallazgo, solo explica el patrón
observado en los datos.

## Experimento A' — repetición con reinicio corregido

| | esta batería (A', modo=parada, 30 semillas) | batería anterior (A, método viejo) |
|---|---|---|
| correlación huella↔entropía | 0,375 ± 0,039 (rango 0,285 a 0,443) | −0,236 ± 0,073 |
| posición del mínimo de huella | 0,8551, idéntica en las 30 semillas | invariante (mismo síntoma, con el defecto de arrastre ya presente) |

La correlación no solo cambió de magnitud sino de **signo** respecto a la
batería anterior. La posición del mínimo sigue sin moverse entre semillas,
pero ahora con el reinicio correcto — ver nota técnica arriba sobre por qué
`computeDaisyworld` no depende de la semilla.

## Experimento C — barajado

| | A' (esta batería) | referencia de la batería anterior |
|---|---|---|
| percentil medio del r real en la nula | 99,72 | — |
| fuera del percentil 95% | 30/30 (100%) | 72,1% |
| fuera del percentil 99% | 25/30 (83,3%) | 28,5% |

Sobre B' (1.080 series, con y sin filtrar combinaciones >10% saturadas):

| | B' completo (1.080) | B' sin combinaciones saturadas (600) |
|---|---|---|
| r medio ± desviación | −0,025 ± 0,327 | 0,078 ± 0,174 |
| percentil medio | 52,8 | 61,4 |
| fuera del percentil 95% | 468/1.080 (43,3%) | 136/600 (22,7%) |
| fuera del percentil 99% | 338/1.080 (31,3%) | 74/600 (12,3%) |

A diferencia de A' (percentil ~100, la correlación sobrevive el barajado casi
siempre), en B' el barrido "típico" tiene un percentil cercano a 50 — la
correlación real es, en promedio, indistinguible de una serie barajada al
azar. La dispersión de `r` en B' es grande (±0,327), consistente con que
distintas combinaciones de parámetros dan correlaciones de signo y magnitud
muy distintos entre sí.

## Experimento B' — saturación y desplazamiento de la frontera

**Saturación:** 48 de 108 combinaciones (44,4%) tienen más del 10% de sus 60
puntos con `saturacion_sensor=1`. Esas combinaciones quedan en el CSV crudo
(marcadas), pero se excluyeron de los promedios de la tabla de abajo, tal
como pide el encargo. Casi todas las combinaciones con `ptcSharp=6,0`
(exponente_ptc) cayeron en esta categoría — de la tabla filtrada por nivel,
el nivel `exponente_ptc=6` desaparece casi por completo (no queda ninguna
combinación limpia con ese nivel para calcular su fila).

**Desplazamiento de la frontera por parámetro** (media sobre combinaciones
limpias, ver tabla completa en `analisis_v75_stdout.log` / `analisis_v75_completo.json`):

| parámetro | nivel bajo | nivel medio | nivel alto | ¿se mueve? |
|---|---|---|---|---|
| beta (persistencia) | 0,897 (β=0,80) | 0,902 (β=0,94) | 0,903 (β=0,98) | apenas (rango ~0,006) |
| tOpt (t_optima) | 0,747 (tOpt=22) | 0,866 (tOpt=25) | 1,011 (tOpt=28) | sí, monótono, rango ~0,264 |
| ptcSharp (exponente_ptc) | 0,886 (sharp=3) | 0,922 (sharp=4,1) | — (sharp=6 sin datos limpios) | sí entre los dos niveles disponibles |
| potencia_base | 0,871 (pB=0,30) | 0,901 (pB=0,47) | 0,929 (pB=0,65) | sí, monótono, rango ~0,058 |

A diferencia de la primera batería (donde la frontera no se movía con NINGÚN
parámetro de la grilla — beta, sigma, potencia_base — porque sigma no tiene
vía causal y beta resultó no alcanzar), acá **tOpt y potencia_base sí mueven
la frontera de forma clara y monótona**; beta sigue moviéndola muy poco.

## Archivos de esta batería (v7.5)

- `motor_v75.mjs`, `correr_barrido_v75.mjs`, `shim_v75.mjs` — motor validado bit a bit (ver `v75_validacion_y_presupuesto.md`)
- `experimento_D_reinicio.csv` (1.200 filas), `experimento_Aprima_repeticion.csv` (1.800 filas), `experimento_Bprima_multivariable.csv` (64.800 filas)
- `experimento_C_barajado.csv` (30 filas, sobre A'), `experimento_C_barajado_Bprima.csv` (1.080 filas, sobre B')
- `Bprima_resumen_por_combinacion.csv` (108 filas: frontera, saturación y r por combinación)
- `analisis_v75_completo.json`, `analisis_v75_stdout.log` — detalle completo para auditoría
- `defectos_encontrados2.md`, `validacion2.md`, `v75_validacion_y_presupuesto.md`, `paso1_sensibilidad_settle.md` — validación y decisiones
