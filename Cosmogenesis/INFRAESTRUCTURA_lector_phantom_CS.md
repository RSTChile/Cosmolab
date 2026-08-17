# Infraestructura: lector de volcados binarios de Phantom en Python

**Fecha:** 5-ago-2026 · **Tipo:** infraestructura (herramienta, no experimento — no requiere adjudicación ni autorización de cierre)

## El bloqueo que resuelve

`DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md` (nodo C-N4) y el roadmap
multi-IA del 5-ago-2026 (frente #11, prioridad P0) señalaban que Cosmogénesis no tenía
forma de leer, en Python, las posiciones y densidades de **todas** las partículas de gas
de una corrida de Phantom — sólo los `.sink` (sumideros) eran legibles como texto plano.
Esto bloqueaba tres frentes: la delimitación C-N4, la jerarquía de controles NULL-1 a
NULL-5 para blindar CS073, y una replicación rigurosa de κ_V con datos de partícula
completos.

## Qué funcionó

**La librería `sarracen`** (paquete Python de la propia comunidad Phantom/SPH,
`pip install sarracen`) lee directamente el binario Fortran de Phantom — no hizo falta
compilar ninguna utilidad de Fortran (`phantom2gadget`, `phantom2hdf5`, etc.).

**Único tropiezo, resuelto:** `pip install sarracen` intentaba compilar `llvmlite` desde
código fuente (sin wheel para Python 3.13 con la versión que pip resolvía por defecto),
porque `sarracen` depende de `numba>=0.55.1` sin techo de versión, pero pip elegía una
versión antigua de numba. Se resolvió instalando primero `numba` y `llvmlite` más
recientes con `pip install --only-binary=:all: numba` (trae `llvmlite==0.45.1`, que sí
tiene wheel para macOS/Python 3.13), y luego `sarracen` sin problema.

**Verificado sobre datos reales del proyecto** — `ic_real` e `ic_null1` de
`/Users/alexis/phantom_cs073/bateria_n2000/`, volcado intermedio de cada corrida
(`cosmog_00250`):

| Corrida | Partículas gas | Sumideros | NaN | ρ mín | ρ mediana | ρ máx |
|---|---|---|---|---|---|---|
| ic_real | 1856 | 8 | No | 0.00056 | 0.749 | 7.45 |
| ic_null1 | 1957 | 6 | No | 0.00299 | 0.0285 | 94.5 |

Rangos físicamente sensatos (positivos, sin NaN, sin outliers imposibles), consistente
con lo esperado: real tiene más masa concentrada en menos partículas gas libres (más
sumideros, ρ mediana más alta) — coherente con el propio hallazgo z=48.69 de CS073.

**Verificación adicional, más fuerte que rangos razonables — conservación de masa exacta.**
En `ic_null1/cosmog_00500` (volcado final, n0=2000, masa por partícula de gas=9.4): el
gas quedó en 1922 partículas (78 acretadas → 78×9.4 = 733.2 de masa "perdida" del
reservorio de gas). La suma de masas de los 8 sumideros en ese mismo volcado da
**exactamente 733.2**. Esto no es un rango plausible — es una igualdad exacta entre dos
cantidades leídas de dos bloques distintos del mismo archivo binario (bloque de gas vs.
bloque de sumideros), lo que descarta que el parser esté leyendo campos desalineados o
basura con apariencia física razonable por casualidad.

## Un detalle no trivial que hay que tener presente al reusar esto

`sarracen.read_phantom()` devuelve un **DataFrame único** si el volcado todavía no
tiene sumideros formados (típico cerca de t=0), pero una **lista `[gas, sinks]`** una
vez que nacieron sumideros. `leer_dump()` normaliza esto: siempre devuelve la tupla
`(gas, sinks_o_None)`. Cualquier script nuevo que use este lector debe manejar el caso
`sinks is None` explícitamente (no asumir que siempre hay sumideros).

## Entregable

`leer_volcado_phantom.py` — dos funciones: `leer_dump(path)` y `listar_dumps(carpeta)`,
con smoke test incorporado (`python leer_volcado_phantom.py` corre y verifica solo).
Código autodescriptivo con docstring de módulo.

## Qué falta para escalar (no resuelto acá a propósito)

- Leer **todos** los pasos temporales de una corrida (no sólo uno) para análisis de
  evolución — usar `listar_dumps()` + loop, con cuidado de memoria si son muchos dumps.
- No se midió tiempo de lectura para corridas grandes (N8550, test_massiva) — antes de
  asumir que un barrido temporal completo sobre esas corridas es barato, cronometrarlo.
- No se generó todavía ningún resultado de Teoría con esto — sólo la herramienta. El
  siguiente paso (delimitación C-N4, jerarquía NULL de CS073) es una tarea aparte.

## Estado

**Listo para usar.** Desbloquea el frente #11 (C-N4) y el frente #6 (jerarquía NULL-1 a
NULL-5 de CS073) del roadmap — ambos quedan pendientes de una decisión de Alexis sobre
si conviene correr esos análisis ahora (son de solo lectura sobre datos ya existentes,
no requieren nuevas corridas de Phantom) o esperar a la Fase II completa.
