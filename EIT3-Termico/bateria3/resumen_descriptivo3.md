# Resumen descriptivo — tercera batería EIT-3 Térmico κ_H (v7.6.1)

Solo estadística descriptiva. Ninguna conclusión sobre si hay o no "hallazgo":
eso es del investigador principal.

**Nota de comparabilidad con `bateria2/` (v7.5):** acá el eje de luminosidad es
0,60→1,40 (el rango que v7.6.1 declara como el único válido para ese control),
no 0,25→1,95 como en `bateria2/`. Además `TOPE_EQ`/`TOPE_REC` están rebajados
a 6.000/3.000 (antes 20.000/20.000), decisión de esta batería respaldada por
datos (ver `topes_investigacion.md`), sin afectar ningún valor medido —0
reclasificaciones—. Las comparaciones de abajo son entre diseños distintos
(eje angosto vs. eje ancho), no una repetición exacta.

## Cobertura

| experimento | barridos | filas |
|---|---|---|
| D · modo de reinicio (10 semillas × 2 modos) | 20 | 1.200 |
| A' · repetición (30 semillas, modo=parada) | 30 | 1.800 |
| B' · multivariable (108 combinaciones × 10 semillas) | 1.080 | 64.800 |
| C · barajado sobre A' | 30 series × 1.000 barajes | 30 filas |
| C · barajado sobre B' | 1.080 series × 1.000 barajes | 1.080 filas |

0 filas con `NaN`/`Infinity`. 0 barridos fallidos de 1.130. Tiempo real total:
**15,41 h** con 14 procesos (throttling térmico confirmado de nuevo,
`CPU_Speed_Limit` bajó a 39% durante la corrida — con los topes viejos
hubiera sido bastante más, la reducción de topes sí ayudó aunque no alcanzó
la proyección optimista de ~6,1h).

## Experimento D — modo de reinicio

| modo | posición del mínimo de huella (10 semillas) | correlación huella↔entropía |
|---|---|---|
| `parada` | 0,8169, idéntica en las 10 semillas | 0,044 ± 0,166 |
| `inicio` | 0,8169, idéntica en las 10 semillas | 0,061 ± 0,168 |

Igual que en `bateria2`: ninguno de los dos modos mueve la posición del
mínimo entre semillas (misma explicación técnica: `computeDaisyworld()` no
consume el generador de azar). A diferencia de `bateria2` (parada r=0,367,
inicio r=0,509 — ambas claramente positivas), acá **la correlación es
prácticamente cero en los dos modos**, con desviaciones (±0,166) del mismo
orden que la propia media.

## Experimento A' — repetición

| | esta batería (eje 0,60-1,40) | bateria2 (eje 0,25-1,95) |
|---|---|---|
| correlación huella↔entropía | 0,008 ± 0,134 (rango −0,237 a 0,267) | 0,375 ± 0,039 |
| posición del mínimo | 0,8169, idéntica en las 30 semillas | 0,8551, idéntica en las 30 semillas |

La correlación **cambió radicalmente**: de un valor positivo consistente y
con poca dispersión entre semillas (bateria2) a un valor centrado en cero con
dispersión grande y que cruza de signo según la semilla (esta batería). El
mínimo de huella sigue sin moverse entre semillas en ambos casos.

## Experimento C — barajado

| | A' esta batería | A' bateria2 | referencia original (primera batería) |
|---|---|---|---|
| percentil medio del r real | 51,6 | 99,72 | — |
| fuera del percentil 95% | 2/30 (6,7%) | 30/30 (100%) | 72,1% |
| fuera del percentil 99% | 0/30 (0%) | 25/30 (83,3%) | 28,5% |

En esta batería, el r real de A' es —en promedio— **indistinguible de una
serie barajada al azar** (percentil ~52, centro de la nula). Es la inversa
del patrón de `bateria2`, donde el r real caía casi siempre en el extremo de
la distribución nula.

| | B' esta batería (1.080) | B' bateria2 (1.080) |
|---|---|---|
| r medio ± desviación | −0,013 ± 0,149 | −0,025 ± 0,327 |
| percentil medio | 47,7 | 52,8 |
| fuera del percentil 95% | 108/1.080 (10,0%) | 468/1.080 (43,3%) |
| fuera del percentil 99% | 26/1.080 (2,4%) | 338/1.080 (31,3%) |

También en B' el efecto es mucho más chico que en `bateria2`: 10% fuera del
95% (vs. el 5% esperable por azar puro) en vez de 43,3%.

## Experimento B' — saturación y desplazamiento de la frontera

**Saturación: 2 de 108 combinaciones (1,9%)** superan el 10% de puntos
saturados — muchísimo menos que en `bateria2` (44,4%). Consistente con que el
eje angosto (0,60-1,40) es justamente el rango donde v7.6.1 declara que el
sensor PTC responde bien; el eje ancho de `bateria2` empujaba el sistema a
los extremos con más frecuencia. Excluir esas 2 combinaciones de los
promedios cambia muy poco el resultado agregado (ver `analisis_v76_stdout.log`).

**Desplazamiento de la frontera por parámetro** (media sobre las 108
combinaciones — casi ninguna se excluye por saturación esta vez):

| parámetro | nivel bajo | nivel medio | nivel alto | ¿se mueve? |
|---|---|---|---|---|
| beta | 0,819 (β=0,80) | 0,821 (β=0,94) | 0,822 (β=0,98) | prácticamente no (rango ~0,003) |
| tOpt | 0,661 (tOpt=22) | 0,819 (tOpt=25) | 0,982 (tOpt=28) | sí, monótono, rango ~0,320 (más marcado que en bateria2, ~0,264) |
| ptcSharp | 0,848 (sharp=3) | 0,819 (sharp=4,1) | 0,795 (sharp=6) | sí, monótono, rango ~0,054 |
| potencia_base | 0,799 (pB=0,30) | 0,819 (pB=0,47) | 0,844 (pB=0,65) | sí, monótono, rango ~0,045 |

Mismo patrón cualitativo que `bateria2`: tOpt domina el desplazamiento,
potencia_base y ptcSharp lo mueven de forma moderada, beta casi no influye.

## Archivos de esta batería (v7.6.1)

- `motor_v76.mjs`, `correr_barrido_v76.mjs`, `shim_v76.mjs`, `orquestador_v76.mjs` — motor validado bit a bit (`validacion3.md`, `validacion3_topes_bajados.md`)
- `experimento_D_reinicio.csv` (1.200 filas), `experimento_Aprima_repeticion.csv` (1.800 filas), `experimento_Bprima_multivariable.csv` (64.800 filas)
- `experimento_C_barajado.csv` (30 filas, sobre A'), `experimento_C_barajado_Bprima.csv` (1.080 filas, sobre B')
- `Bprima_resumen_por_combinacion.csv` (108 filas: frontera, saturación y r por combinación)
- `analisis_v76_completo.json`, `analisis_v76_stdout.log` — detalle completo para auditoría
- `topes_investigacion.md`, `defectos_encontrados3.md`, `validacion3.md`, `v76_validacion_y_presupuesto.md` — validación y decisiones
