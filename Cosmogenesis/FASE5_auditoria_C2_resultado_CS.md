# Auditoría de `kcap` (eje C2) — ¿es relacional puro o esconde escala/geometría?

**Fecha:** 10-ago-2026 · Ejecuta: CC (Claude), tarea encargada por Alexis a partir de la duda de los dos
analistas externos sobre `A2-B0-C2` antes de pasar a Phantom. No se corrió Phantom. No se tocó
`cs090_fase5_generador.py` ni `cs090_fase5_motor.py` (se puede verificar con `git diff` — cero cambios en
esos dos archivos). Se creó un archivo nuevo: `cs090_fase5_auditoria_C2.py`. No se declara cierre ni
veredicto sobre si A2-B0-C2 "es válido" — se reportan números, la lectura final es de Alexis.

## Resumen de una línea

**PASS en la pregunta que importaba** (kcap no lee coordenadas, distancia, layout ni tamaño de caja —
depende de grado y de vecinos compartidos, cantidades puramente relacionales, y el costo que lo
acompaña es invariante a reescalado de unidades) — **con una salvedad documentada**: la poda tiene un
detalle de desempate por índice de nodo que rompe la invarianza EXACTA bajo renombrado, pero su efecto
medido sobre la clasificación final es del mismo tamaño que el ruido estocástico normal del candidato,
no una fuente adicional de sesgo geométrico.

---

## Parte 1 — Auditoría de código (lectura línea por línea)

### La pregunta: ¿de qué depende `kcap`?

**Generación** — `cs090_fase5_generador.py`:
```
línea 45:  RANGO_KCAP     = (4, 7)         # límite de escala duro (C2): grado máximo por nodo
línea 77:      kcap=_sample(rng, *RANGO_KCAP, entero=True),
```
`kcap` es un **entero sampleado UNA vez por regla**, uniforme en {4,5,6,7}, dentro de `generar_regla()`.
No hay ninguna expresión `kcap = f(N)`, `kcap = f(distancia)` ni `kcap = f(coordenadas)` en todo el
archivo — es un parámetro plano, igual que `K` o `J`.

**Aplicación** — `cs090_fase5_motor.py`, función `_enforce_kcap` (líneas 136-148):
```python
def _enforce_kcap(adj, N, kcap):
    """C2 — límite de escala duro: ningún nodo conserva más de `kcap` vecinos; si excede, se quedan
    los `kcap` de mayor soporte local (vecinos compartidos), MISMO criterio de ranking que
    gate_localidad/cs081 (ya validado), aligerado."""
    for i in range(N):
        nb = list(adj[i])
        if len(nb) <= kcap:
            continue
        sup = sorted(((len(adj[i] & adj[j]), j) for j in nb), reverse=True)
        mantener = set(j for _, j in sup[:kcap])
        for j in nb:
            if j not in mantener:
                adj[i].discard(j); adj[j].discard(i)
```
Llamada desde `dinamica_B0` (líneas 210-211):
```python
if costo_nivel == "C2" and step % 4 == 0:
    _enforce_kcap(adj, N, p["kcap"])
```

**Lo que lee esta función, literalmente:**
- `len(nb)` — el **grado** del nodo i (cuántos vecinos tiene ahora mismo en el grafo).
- `len(adj[i] & adj[j])` — el **soporte local** de la arista (i,j): cuántos vecinos COMPARTEN i y j.
  Esto es una cantidad puramente topológica (se calcula con intersección de conjuntos de adyacencia, no
  con posiciones), el mismo criterio que ya usa `gate_localidad` de `cs081_poda_dinamica.py`
  (script congelado anterior, reusado en espíritu, no en código).
- `kcap` — el entero fijo por regla.

**Lo que NO lee en ningún punto:** ningún arreglo de coordenadas, ninguna distancia euclidiana, ninguna
posición de layout, ningún tamaño de caja. Confirmado por `grep -inE "coord|pos\[|posicion|xy=|embedding|
distanc|euclid|layout|caja|box"` sobre `cs090_fase5_generador.py` y `cs090_fase5_motor.py` completos: los
únicos resultados son comentarios/docstrings que EXPLICAN que no hay coordenadas (ver P3 de
`chequear_P3_localidad`, el propio chequeo automatizado del filtro de admisión), o las funciones de
coarse-graining (`cajas_bfs`/`grafo_grueso`, usadas SÓLO para medir después de que la dinámica ya corrió,
nunca para decidir la poda de kcap).

### Respuesta a la pregunta de auditoría (Parte 1)

**(a) — kcap limita algo relacional puro: grado + soporte local (vecinos compartidos).** No es **(b)**:
no hay distancia geométrica externa, coordenadas, tamaño de caja, ni ningún valor elegido para forzar un
resultado — es un entero de 4 a 7 muestreado al azar, y el criterio de "a quién podar" es un conteo de
vecinos compartidos, que es tan relacional como el grado mismo. **Pasa P3 (localidad relacional sin
coordenadas) y P5 (sin escala física horneada)** por inspección de código.

**Analogía simple:** pensá en kcap como "cada persona puede tener como máximo 5 amigos cercanos". Cuando
una persona tiene más de 5, se queda con los 5 que tienen MÁS amigos en común con ella (no con los que
viven más cerca en el mapa — no hay mapa). Eso es 100% información de la red de amistades (quién conoce
a quién), nunca de geografía.

---

## Parte 2 — Tests de invarianza empíricos (anti-Shannon)

Script nuevo `cs090_fase5_auditoria_C2.py`. Reusa `cs090_fase5_generador.py`, `cs090_fase5_motor.py`,
`cs090_fase5_clasificador.py` y `cs080_renormalizacion.py` **sin editarlos** (import directo; donde hace
falta sustituir una pieza para el test, se hace por inyección de un input distinto o por monkeypatch de
un nombre a nivel de módulo EN MEMORIA de este proceso — el archivo en disco no cambia).

Se usaron las **5 reglas Clase III reales** de `FASE5A_profundizar_A2B0C2_resultado_CS.md`
(`A2-B0-C2-r2, r4, r7, r13, r16`, mismos seeds, mismos K/J/noise/meandeg/kcap), reproducidas
determinísticamente con `generar_regla()`.

### Test 1 — renombrado de nodos, mecanismo `_enforce_kcap` aislado (exacto, determinístico)

`_enforce_kcap` no tiene azar propio, así que se puede comparar EXACTO: podar-y-luego-renombrar vs.
renombrar-y-luego-podar, sobre 30 grafos aleatorios (N=80).

**Resultado: 0/30 exactamente iguales.** Diagnóstico: en un grafo de ejemplo (N=80, kcap=5), **39 de 80
nodos tenían un empate exacto de "soporte" justo en la frontera de corte** (el vecino kcap-ésimo y el
kcap+1-ésimo comparten el mismo número de vecinos comunes). El desempate lo resuelve
`sorted(..., reverse=True)` sobre la tupla `(soporte, índice_j)` — **a igualdad de soporte, gana el nodo
con índice numérico más alto**. Eso SÍ es una dependencia del índice/etiqueta bruta del nodo, no de una
cantidad relacional — un detalle de implementación (desempate arbitrario, común en algoritmos greedy), no
una fuga de coordenadas/distancia, pero tampoco 100% relacional en sentido estricto.

### Test 1b — renombrado de nodos, pipeline COMPLETO (construcción→dinámica→clasificación)

Se renombran los nodos de A2 justo después de construir el grafo inicial (antes de que corra la dinámica),
con el resto del pipeline sin tocar.

| regla | baseline | renombrado | ¿igual? |
|---|---|---|---|
| A2-B0-C2-r2  | III | III | sí |
| A2-B0-C2-r4  | III | III | sí |
| A2-B0-C2-r7  | III | **I** | **no** |
| A2-B0-C2-r13 | III | **intermedio** | **no** |
| A2-B0-C2-r16 | III | III | sí |

**2/5 cambiaron de clase.**

### Test 2 — orden de recorrido de nodos dentro de `_enforce_kcap` (barajado en vez de secuencial)

| regla | baseline | orden barajado | ¿igual? |
|---|---|---|---|
| A2-B0-C2-r2  | III | III | sí |
| A2-B0-C2-r4  | III | III | sí |
| A2-B0-C2-r7  | III | **I** | **no** |
| A2-B0-C2-r13 | III | III | sí |
| A2-B0-C2-r16 | III | III | sí |

**1/5 cambió de clase.**

### Control — ¿es esto ruido estocástico normal, o algo específico de kcap?

Antes de sacar conclusiones de Test 1/1b/2, hacía falta descartar la explicación más simple: que
A2-B0-C2 sea sencillamente RUIDOSO cerca del umbral I/III (ya lo decía
`FASE5A_profundizar_A2B0C2_resultado_CS.md`: "el resultado sigue viéndose parcialmente estocástico").
Se corrieron las mismas 5 reglas **sin renombrar nada, sin reordenar nada — sólo con una semilla de azar
distinta** (misma K/J/noise/meandeg/kcap):

| regla | baseline | semilla distinta (sin tocar nada más) | ¿igual? |
|---|---|---|---|
| A2-B0-C2-r2  | III | III | sí |
| A2-B0-C2-r4  | III | **I** | **no** |
| A2-B0-C2-r7  | III | III | sí |
| A2-B0-C2-r13 | III | III | sí |
| A2-B0-C2-r16 | III | III | sí |

**1/5 cambió de clase — con SOLO cambiar la semilla, nada de renombrado ni de orden.**

**Comparación de tasas de cambio:** renombrado=2/5, orden barajado=1/5, control (sólo semilla)=1/5 — del
mismo orden de magnitud. **Lectura honesta:** la inestabilidad que se ve en Test 1b/2 no es claramente
mayor que el ruido estocástico normal del candidato (r4 cambió SOLO con la semilla, sin renombrar ni
reordenar nada; r7 cambió con renombrado y con orden pero NO con la semilla control usada). El desempate
por índice detectado en Test 1 es real y mide algo genuino (no es 100% relacional en sentido estricto),
pero su huella en el resultado final (Clase I vs III) es del mismo tamaño que la sensibilidad ordinaria
del candidato cerca del umbral de clasificación — no se ve como una fuente adicional, sistemática, de
sesgo geométrico escondido.

**Analogía simple:** es como preguntarse si una moneda ligeramente abollada cae más veces cara porque la
abolladura pesa, o porque las monedas ya de por sí caen distinto cada vez que las tirás. Acá, tirar la
moneda "distinto" (semilla nueva, sin tocar la abolladura) cambia el resultado casi tanto como cambiar la
abolladura (renombrar/reordenar) — así que no se puede culpar a la abolladura específicamente de la
inestabilidad que se observa.

### Test 3 — reescalado arbitrario de las unidades de costo (historial de flips, holonomía)

`_costo_y_podar` (la función que también actúa en C1/C2 podando por costo, además del límite duro de
kcap) normaliza por z-score antes de podar por percentil. Matemáticamente, `z(c·x) = z(x)` para
cualquier constante `c>0` — la poda por percentil debería ser exactamente invariante al reescalado de
unidades. Se verificó empíricamente: 20 pruebas, multiplicando el historial de flips por constantes entre
0.01x y 1000x.

**Resultado: 20/20 exactamente el mismo conjunto de aristas conservadas.** Confirma que ni el costo ni el
"presupuesto" que co-actúa con kcap dependen de en qué unidades se midan — pasa el test de reescalado.

### Test 4 — ¿kcap está atado a un tamaño N específico?

Por código: `RANGO_KCAP=(4,7)` se samplea una sola vez por regla (generador, líneas 45/77); en el motor,
`p["kcap"]` es el MISMO valor en las 4 tallas del piloto (500/1000/1500/2000) porque `correr_regla()`
reusa el mismo dict `p` en cada talla — no hay recálculo. Confirmación empírica: se corrió la regla
`A2-B0-C2-r2` (kcap=5) en N=300/600/900/1200 con el mismo kcap en las cuatro, sin ningún ajuste.
**kcap es un conteo absoluto de grado (un número chico, 4 a 7), no una fracción ni una función de N.**
Pasa el test — no hay una escala absoluta ligada en secreto a un N particular.

---

## Veredicto de la auditoría (no de A2-B0-C2 como candidato — eso es lectura de Alexis)

| pregunta | resultado |
|---|---|
| Parte 1 — ¿kcap lee coordenadas/distancia/layout/caja? | **NO — PASS.** Sólo grado y soporte (vecinos compartidos). |
| Test 1 — ¿poda exactamente invariante a renombrado? | **NO, exacto.** Desempate por índice de nodo en casos de empate de soporte (39/80 nodos en el ejemplo). Detalle de implementación, no geometría. |
| Test 1b/2 — ¿la CLASIFICACIÓN final cambia con renombrado/orden? | Sí, en 1-2 de 5 reglas — **pero el control (sólo cambiar semilla, sin tocar nada) cambia 1/5 también**, misma magnitud. |
| Test 3 — ¿costo invariante a reescalado de unidades? | **SÍ — PASS**, exacto (20/20). |
| Test 4 — ¿kcap atado a un N específico? | **NO — PASS.** Conteo absoluto fijo por regla, igual en todas las tallas. |

**En síntesis:** la duda concreta que trajeron los dos analistas (¿kcap esconde coordenadas o distancia
disfrazada?) tiene respuesta clara de **PASS** — no hay ninguna lectura de posición/geometría en el
mecanismo de kcap ni en el costo que lo acompaña. Lo que SÍ se encontró, no pedido pero honesto de
reportar, es un detalle de implementación menor (desempate por índice de nodo, no por otra cantidad
relacional) que rompe la invarianza EXACTA bajo renombrado — pero su efecto medible sobre si una regla
cae en Clase I o III es del mismo orden que el ruido estocástico normal del candidato (que ya estaba
documentado como "parcialmente estocástico" en `FASE5A_profundizar_A2B0C2_resultado_CS.md`), no una
fuente sistemática adicional de sesgo. Si Alexis quiere blindar esto más allá de lo que pidió esta
auditoría, el arreglo sería trivial (desempatar por otra cantidad relacional, ej. costo acumulado, o por
azar explícito en vez de por índice) — no se tocó nada, queda anotado para que Alexis decida si vale la
pena.

## Archivos de esta tarea

- `cs090_fase5_auditoria_C2.py` — script nuevo (único archivo de código de esta tarea; no modifica
  ningún script congelado, verificable con `git diff cs090_fase5_generador.py cs090_fase5_motor.py`).
- Este informe.

No se corrió Phantom. No se declaró cierre ni veredicto sobre A2-B0-C2. No se hicieron commits de git.
