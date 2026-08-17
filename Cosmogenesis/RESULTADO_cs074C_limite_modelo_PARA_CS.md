# cs074-C — ¿Relación y proceso sí, números físicos no?

**Fecha:** 2026-07-26 · Análisis sobre 4180 corridas ya hechas (cs074+cs074-A+cs074-B), sin física nueva.

---

## Desviación declarada, encontrada al implementar (no escondida)

Dos de los tres números físicos pre-registrados (razón protón:neutrón, razón de masa
protón/electrón) resultaron ser **constantes estructurales** del motor — no varían en
NINGUNA corrida del barrido disponible (dependen de masas fijas del catálogo y de
`tasa_expansion`, que nunca se barrió). El método pre-registrado (distancia contra un
control de azar re-muestreado del barrido) es matemáticamente inaplicable a un número sin
varianza — no hay barrido del que resamplear. Se reportan igual, marcados "no evaluables
por este método", en vez de forzar una comparación que no tiene sentido.

## Columna "NO" — números físicos

| Número | Valor real | Lo que da el modelo | ¿Evaluable? | Veredicto |
|---|---|---|---|---|
| Fracción de materia | 4,9% / 31,5% | distancia mínima 0,04pp / 0,19pp sobre 4180 puntos | Sí | **z=1,37 — NO significativo.** Tan cerca como cualquier punto cae por puro volumen del barrido (mismo patrón que ya vimos en E5.3-1 y en cs074 original: puntos aislados "cerca" que no resisten el chequeo). |
| Razón protón:neutrón | 7,1 | 7,095 | No (constante, sin barrido) | **Caso aparte, no cuenta como excepción.** La fórmula (`freeze_out.py`) usa la diferencia de masa real (1,293 MeV) y la vida del neutrón real (880s) como ENTRADA — que salga cerca de 7,1 es esperable de implementar bien la física de congelamiento conocida con constantes reales, no un hallazgo emergente del modelo. |
| Razón masa protón/electrón | 1836,15 | 18,43 | No (constante, y de otra naturaleza) | Lejos, y comparando cosas distintas: `masa_trio` es la suma de masas DESNUDAS de quarks — el motor basal no le suma la energía de ligadura que da ~99% de la masa real del protón. |

## Columna "SÍ" — relaciones y procesos (6 de 7 sostienen con control real)

| Relación | Control | Resultado |
|---|---|---|
| La contabilidad de energía cierra exacto | gravedad pura | 1,7% de fuga (límite 5%) |
| El costo de ligadura tiene efecto causal | presupuesto finito vs infinito | 29,3% de celdas difieren, justo donde debía |
| Muerte térmica ≠ Nada | E=0 por construcción | retiene ~100% del presupuesto al morir |
| La expansión rescata estructura | con/sin expansión | 88,4% sin vs 60,7% con (compite contra el colapso) |
| El techo no-monótono en ε es real | energía finita vs infinita | curva casi idéntica con y sin presupuesto |
| La gravedad es indispensable | apagarla | 60,7% → 2,0% |
| ~~El enfriamiento H₂ fragmenta~~ | barajado, 10× intensidad | **NO sostiene** — 0/11 niveles con separación |

## Lectura

La sospecha del director queda **confirmada, con matices honestos**: el único número
físico que se pudo testear correctamente (fracción de materia) no coincide de forma
significativa — es indistinguible del azar del propio barrido. Los otros dos números no
se pudieron testear con el método pre-registrado (son constantes del motor, no salidas
del barrido), así que no cuentan ni como confirmación ni como excepción — quedan
abiertos si alguna vez se barre `tasa_expansion` o se le agrega energía de ligadura a la
masa bariónica. El modelo sí sostiene 6 de 7 relaciones/comportamientos con control real y
significativo — el "SÍ da relación y proceso" queda bien respaldado; el "NO da números"
queda confirmado donde se pudo medir, abierto donde no.

**Archivos:** `PROTOCOLO_cs074C_limite_del_modelo_PREREGISTRO.md`, `cs074C_limite_modelo.py`,
`resultados_cs074C_limite_modelo/cs074C_result.json`.
