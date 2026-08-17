# PROTOCOLO E5.3-2 — Eficiencia vs intensidad de ligadura, rango nula-a-total

**Congelado (pre-registro):** 2026-07-24 16:38 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.3-2, batería Enfoque 5, corrida en paralelo con 29 agentes más)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.3-2"

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación
respecto de lo aquí escrito se reporta como desviación explícita, no se edita
retroactivamente (T3).

**Verificación de reutilización (obligatoria antes de definir nada propio):** se
comprobó en disco `Cosmogenesis/BATERIA_ENFOQUE5/E5_3_1_eficiencia_12decadas/` — la
carpeta existe pero está VACÍA (sin protocolo ni motor) en el momento de este
pre-registro (comprobado dos veces: 16:37 y 16:40). No hay definición de E5.3-1 que
heredar. Por lo tanto este protocolo define su PROPIA definición de eficiencia/E_ligada,
declarada como tal (no heredada), siguiendo el MISMO principio exigido para E5.3-1 por
el documento madre: **SALIDA medida, jamás ajustada hacia 4.9%/31.5%.** Para máxima
comparabilidad futura con E5.3-1 (si aparece), se reutiliza el mismo campo φ, la misma
física de difusión/expansión de `cs074_rcruz.py`, y la misma grilla de ε que usó el
agente hermano E5.1-1 (verificada en disco: `E5_1_1_supervivencia_exergia/PROTOCOLO_E5.1-1_PREREGISTRO.md`).

---

## 1. Pregunta

Al variar cuán fuerte "liga" la estructura (cuánto resiste a romperse por la expansión),
¿cambia la fracción de energía que queda atrapada (ligada) como estructura, y cómo es esa
curva eficiencia(intensidad_ligadura)?

## 2. Modelo (heredado de cs074_rcruz.py, motor propio bajo mi prefijo)

Mismo campo escalar φ en anillo de N=200 sitios, misma `campo_inicial`, misma
`paso_difusion` (relajación local, SOLO por aristas vivas), reutilizadas SIN editar el
archivo base (se importan como funciones).

**La novedad de este experimento — intensidad de ligadura L:**

El código base (`paso_expansion`) corta cada arista viva con probabilidad de Bernoulli H
por paso (H=tasa de expansión). Ese H es el "cuánto se rompe" ya definido en cs074_rcruz.
Aquí se introduce el eje que el documento madre pide y que NO existe en el código base
literal (no hay `H_TOPO` ni `ALPHA_CUT` en `cs074_rcruz.py` — se construye el análogo
pedido): la **intensidad de ligadura L** modula H hacia abajo, representando resistencia
a romperse:

    H_eff(L) = H0 / (1 + L)

- L→0 (1e-3): ligadura nula → H_eff ≈ H0 (la estructura se rompe al ritmo natural, sin
  resistencia).
- L→∞ (1e2): ligadura casi total → H_eff ≈ H0/101 (~1% del ritmo natural: la estructura
  resiste casi toda la expansión).

**H0 (tasa natural de referencia, NO ajustada a mano):** H0 = D medido por (ε, semilla)
— el mismo anclaje r=1 que `cs074_rcruz.py` identificó como el punto de transición
natural expansión/difusión (ver docstring del archivo base, líneas 19-20: "r pre-registrados
que CRUZAN 1"). Se reutiliza ese punto ya identificado por el código base en vez de
inventar una constante nueva. D se mide con `medir_D()` de la base, sin editar.

**Ruido dinámico (T7):** en cada paso de evolución se suma al campo ruido gaussiano de
amplitud NOISE_REL·ε, NOISE_REL=0.02 (misma constante que usó E5.1-1, declarada aquí
ANTES de correr, no ajustada después). Con ε=0 el ruido es exactamente 0.

**pasos:** calibrados UNA vez con `medir_pasos_lavado()` de la base (ε=1e-3, H=0,
umbral P<0.05, margen ×1.15), reusados fijos en toda la grilla — mismo método que el
modo "produccion" de la base y que usó E5.1-1.

## 3. Axiomas declarados (E1/E2, NO física real)

- **E1 (conservación declarada):** E_total se fija UNA vez por (ε, semilla) como la
  energía de desviación inicial, E_total = Σ(φ₀ − mean(φ₀))², medida del propio campo
  ANTES de evolucionar. Nunca se retoca. Sirve de denominador fijo para la eficiencia.
- **E2 (redistribución, no creación):** la expansión (cortar aristas) no crea energía;
  aísla regiones y así congela diferencias que la difusión habría borrado. Este
  experimento MIDE cuánta de esa energía queda atrapada como diferencia-entre-piezas
  (ligada) vs. cuánta sigue como estructura dentro de una pieza (aún relajable).

## 4. Observable — Eficiencia = E_ligada / E_total

Al final de la evolución (`activo` final define las aristas vivas → segmentos =
tramos contiguos del anillo separados por aristas cortadas):

    Para cada segmento k (nodos i∈k): μ_k = mean(φ_i, i∈k), n_k = |k|
    E_ligada  = Σ_k n_k · (μ_k − mean_global(φ))²         (varianza ENTRE segmentos)
    E_dentro  = Σ_k Σ_{i∈k} (φ_i − μ_k)²                   (varianza DENTRO de cada segmento)
    Identidad de auditoría (ANOVA): E_ligada + E_dentro = Σ(φ − mean_global(φ))² = E_final

    eficiencia = E_ligada / E_total          (E_total fijado en §3, nunca el mismo E_final)

Justificación: la energía "ligada" es precisamente la que quedó ATRAPADA porque una
arista se cortó (ligadura actuó) y ya no puede promediarse con el resto — es la huella
medible de la ligadura, no una elección arbitraria. Si nunca se corta nada (L→∞, un solo
segmento), E_ligada=0 por construcción (no hay "entre" sin al menos 2 piezas) — esto es
una predicción de mecanismo, no un resultado buscado.

**Juez ≠ observable (T2):** el veredicto usa la curva completa eficiencia(L) por ε
contra NULL, no un único número.

**Segundo observable de verificación cruzada (regla de ejecución #4):** número de
segmentos final (fragmentación) — debe subir monótono al bajar L, si el mecanismo es el
descrito; se reporta junto a la eficiencia como chequeo independiente.

## 5. NULL

Se permutan los VALORES de φ al final (idéntico principio a `evolucionar(...,
null=True)` de la base) manteniendo la MISMA partición en segmentos (`activo` real) —
esto aísla si la eficiencia observada depende de la estructura espacial real dentro de
cada segmento o sería igual de alta con cualquier asignación de valores a esa misma
partición (T4: el NULL debe morder). Misma semilla, mismo L, mismo ε; difieren solo en
el barajado final.

## 6. Barrido (sobredimensionado, regla del director)

| Eje | Rango | Puntos |
|---|---|---|
| L = intensidad_ligadura | logspace(1e-3, 1e2) | 10 (5 décadas exactas, según spec) |
| ε | {0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.3, 1.0} | 9 (idéntica a E5.1-1, comparabilidad entre-tema) |
| semillas | 0..11 | 12 (mínimo exigido por spec) |
| ruido dinámico | NOISE_REL=0.02·ε cada paso | fijo, declarado |
| N | 200 (igual que modo "produccion" de la base) | — |
| pasos | calibrado una vez (lavado P<0.05, ε=1e-3, H=0) | — |

Total combinaciones (L, ε) = 10×9 = 90. Cada combinación: 12 semillas × {REAL, NULL} =
24 → **2160 evoluciones de campo** + calibración de lavado.

## 7. PASS / criterios de lectura (congelados antes de correr)

- **ε=0 → eficiencia=0** a todo L (sin diferencia inicial no hay nada que quede ligado;
  E_total=0 → eficiencia se reporta como 0 por convención, no NaN).
- **L→0 (H_eff≈H0, ritmo natural) → fragmentación alta, eficiencia debería ser
  relativamente ALTA** (el campo se congela rápido, antes de difundirse, preservando casi
  todo el patrón inicial repartido en piezas aisladas).
- **L→∞ (H_eff→0, casi nunca corta) → un solo segmento (o pocos) → eficiencia debería
  caer hacia 0** (sin piezas que dividan la estructura, la varianza "entre" no existe;
  toda la energía que sobrevive es "dentro", y el ring completo tiende a difundirse con
  `pasos` calibrados para lavar).
- Esta es una PREDICCIÓN DE MECANISMO explícita, no el resultado deseado — se reporta la
  curva real, monótona o no.
- **NULL debe caer más bajo que REAL** en la región donde REAL muestra eficiencia
  apreciable (T4); si el NULL iguala al REAL, el hallazgo es artefacto de la partición
  (tamaños de segmento), no de la estructura espacial real.
- **Ningún coeficiente se toca para acercar la curva a 4.9%/31.5%** — esos valores NO
  entran en el motor en ningún punto (regla de ejecución #6).
- Si cualquiera de estos falla, se reporta como tal, sin reinterpretar ni ajustar
  después de ver los datos (T3).

## 8. Verificación cruzada (regla de ejecución #4)

1. NULL propio (permutación intra-partición), por celda (L, ε).
2. Segundo observable: número de segmentos final (fragmentación), reportado en paralelo.
3. Auditoría de identidad ANOVA (E_ligada + E_dentro == E_final, tolerancia 1e-9) en
   cada corrida — si falla, es error de cómputo, se reporta y se para (T6).
4. Auditoría de conservación E1: E_total nunca se recalcula tras evolucionar (fijado al
   inicio); se reporta también E_final/E_total (¿crece o decrece la energía total del
   campo con la evolución? — dato honesto, no forzado a 1).

## 9. Salidas

- `E5_3_2_motor.py` — motor (escrito DESPUÉS de este pre-registro).
- `E5_3_2_resultado_crudo.json` — filas completas del barrido (L, ε, H0, H_eff, D,
  pasos, eficiencia_real media/std por semilla, eficiencia_null media/std, z,
  n_segmentos_real medio, auditoría ANOVA, E_final/E_total).
- Reporte final verbatim a CS (este agente no adjudica el veredicto de la batería).

## 10. Trampas explícitamente evitadas

- T0: nada discreto puesto a mano — N y pasos vienen del modelo base y de calibración
  medida.
- T1: NOISE_REL=0.02 es constante declarada aquí, heredada de E5.1-1 para consistencia
  entre agentes, no ajustada después de ver resultados. H0=D es una medida, no una
  constante inventada.
- T2: eficiencia es una fórmula fija (ANOVA entre-segmentos); el veredicto lo da la
  curva completa contra NULL.
- T5: se reporta la curva eficiencia(L) entera por cada ε, no un gate binario.
- T6: se audita la identidad ANOVA y E_final/E_total cada corrida.
- T7: ruido dinámico presente en cada paso, además de 12 semillas.

No se corre nada del motor hasta que este archivo esté guardado en disco.
