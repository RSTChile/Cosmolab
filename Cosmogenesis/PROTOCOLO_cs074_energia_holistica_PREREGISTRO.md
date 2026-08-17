# PROTOCOLO cs074 — Energía como capa transversal, experimento holístico

**Congelado (pre-registro):** 2026-07-25 · **Ejecutor:** CC · **Director:** Alexis López Tapia
**Diseño base:** `DISENO_experimento_holistico_todos_los_factores_PARA_CC.md` (leído entero,
congelado antes de este documento).
**Código base reusado (leído, NO editado):** `cs072_modulos/` (motor validado, ver su README:
"no se edita la física") y `cs073_cierre_holistico.py` (patrón de orquestador nuevo, tampoco
editado — se reutilizan sus piezas `Expansion`, `GravedadGeneral`/`energia_total`,
`EnfriamientoH2`, `MateriaOscuraHalo`, no su lógica de bucle).

Este documento se escribe y congela ANTES de escribir `cs074_energia_holistica.py`. Cualquier
desviación respecto de lo aquí escrito se reporta como desviación explícita (T3).

---

## 1. Pregunta

Cuando la contabilidad de energía (exergía/degradada/ligada) corre JUNTO con la dinámica de
formación de estructura (gravedad general + expansión + enfriamiento H₂ + materia oscura) —no
antes, no después, no aparte— ¿qué fracción del presupuesto energético termina ligada en
estructura estable? ¿Depende esa fracción de cuán escaso o abundante es el presupuesto
(`E_reserva`)? ¿El presupuesto tiene un efecto CAUSAL real sobre el resultado (prueba de
admisibilidad) o es contabilidad decorativa?

**T-target (recordatorio permanente):** 4,9%/31,5% es SOLO test de salida. No entra en ninguna
fórmula, no se ajusta nada para acercarse.

## 2. Arquitectura (por qué no se toca nada validado)

`cs072_modulos/` está cerrado (validado CS057→CS072). Sus 23 piezas + el proceso quark→átomo
NO tienen bucle editable desde afuera sin modificar ese paquete — así que la capa de energía
NO se integra ahí. Se integra en la ÚNICA fase con bucle paso-a-paso propio y editable: la
dinámica de formación de estructura que `cs073_cierre_holistico.py` ya construyó (gravedad
general + expansión + H₂ + CDM, actuando juntos). `cs074_energia_holistica.py` (nuevo, raíz):
1. Llama `cs072_modulos.nucleo.corre()` UNA vez (determinista) para obtener bariones/masa/
   densidad — igual que hace CS073.
2. Escribe SU PROPIO bucle de dinámica (mismas piezas reusadas — `Expansion`,
   `GravedadGeneral`, `EnfriamientoH2`, `MateriaOscuraHalo` — importadas, no reimplementadas;
   misma convención G_ADIM=1.0, SOFTENING=0.3 que CS073), con la capa de energía añadida.

## 3. La capa de energía — cómo se opera cada regla

**Ledger** (un dict simple, acompaña el bucle; NO se mete dentro de `Estado`, que pertenece a
la fase validada y no se toca):
`E_total(0) = KE(0) + PE(0) + E_reserva` — `E_reserva` es el presupuesto latente NUEVO
(barrido, ver §5), más allá de la energía mecánica del escenario inicial.

- **Regla 1 (conservación exacta), el chequeo duro:** cada paso se MIDE (nunca se asume)
  `KE(t)+PE(t)` con `energia_total()` (la misma función ya escrita en
  `p_gravedad_general.py`, mismo softening que usan las aceleraciones). Se audita:
  `residual(t) = E_total(0) − [KE(t)+PE(t)] − reserva_restante(t) − ligada_acum(t)`.
  `reserva_restante` y `ligada_acum` sólo cambian por los cobros de Regla 4 (abajo) — son
  exactos por construcción, no estimados. `residual(t)` absorbe TODO lo demás sin
  descomponerlo en canales separados (pérdida térmica de H₂, efecto del estiramiento
  isótropo sobre PE, error numérico de integración) — **decisión de implementación,
  declarada aquí, no escondida:** separar esos tres canales con precisión requeriría
  instrumentar `EnfriamientoH2`/`Expansion` por dentro (no se hace, son piezas reusadas tal
  cual). En su lugar, `residual(t)` es UN solo bucket auditado, y el criterio de cordura es:
  - **Control (smoke test, sin expansión ni enfriamiento, gravedad pura):**
    `|residual(t)|/E_total(0)` debe quedarse cerca de cero (error numérico puro — el mismo
    diagnóstico de cordura que el docstring de `energia_total()` ya declara).
  - **Con expansión/enfriamiento activos:** `residual(t)` puede ser distinto de cero (son
    mecanismos "por diseño", como ya advierte esa misma función) — se reporta la curva
    completa, no se le exige cero.
  - **Umbral de falla declarado:** si en el tramo de CONTROL (gravedad pura, sin
    expansión/cooling) `max_t |residual(t)|/E_total(0) > 0.05` (5%), la corrida se marca
    **FALLADA** (contabilidad rota, no se reporta como física). 5% se elige por ser holgado
    frente al error de integración esperado de un leapfrog con subpasos (ya usado y
    verificado en CS073) pero ajustado para cazar un bug real de contabilidad — declarado
    ANTES de correr, no ajustado después de ver el número.

- **Regla 2 (exergía = depende de las diferencias):** `X(t) = mean[(ρ_local(t)/⟨ρ_local(t)⟝ −
  1)²]` sobre los bariones — misma familia funcional que la definición canónica de Enfoque 5
  (`_observables_homologadas.py`: desviación cuadrática de un reparto uniforme), aplicada
  aquí sobre la densidad local dinámica que `EnfriamientoH2._densidad_local_dinamica()` YA
  calcula (se reutiliza esa función, no se reimplementa el estimador).

- **Regla 3 (la expansión rescata exergía):** NO es un canal nuevo del ledger — es una
  predicción cualitativa a verificar con los datos del barrido: `X` retenida al final debe
  ser mayor con `expansion_on=True` que con `False`, a igualdad de todo lo demás
  (`expansion_on` ya es un interruptor existente en el patrón CS073, se barre como palanca).

- **Regla 4 (costo de ligadura), el mecanismo causal nuevo:** cada paso, tras `_fof()`
  (friends-of-friends, `min_miembros=2` — un par ya es ligadura, más laxo que el
  `min_miembros=5` que CS073 usa para su propio observable de Jeans, que se mantiene aparte
  sin cambios para no perder comparabilidad con esa corrida), para cada cluster candidato:
  - **Criterio de ligadura (energético, virial):** `KE_interno` (relativo al centro de masa
    del cluster, así no cuenta el movimiento de conjunto) `+ PE_interno < 0`. `PE_interno`
    se obtiene de `energia_total(...)` restringida a los miembros del cluster, restando la
    `KE_total` (no relativa) que esa función ya incluye — álgebra directa, no fórmula nueva.
  - **Reparto del costo (simplificación declarada):** de los miembros del cluster, sólo los
    que NUNCA fueron acreditados antes (`nuevos`) reciben cobro, prorrateado:
    `costo = |PE_interno| × (n_nuevos / n_miembros)`. Evita cobrar dos veces al mismo átomo
    cuando clusters chicos se fusionan en uno grande, sin necesitar recalcular PE de
    subconjuntos históricos.
  - **El gate:** si `costo ≤ reserva_restante` → se cobra (`reserva −= costo`,
    `ligada += costo`) y esos átomos quedan ACREDITADOS (cuentan como "materia" en el
    balance final). Si `costo > reserva_restante` → NO se acreditan (siguen existiendo
    dinámicamente — la gravedad no se altera, ninguna trayectoria cambia — pero no cuentan
    en la fracción final de estructura ligada). Este gate es la ÚNICA forma en que el
    presupuesto afecta el resultado — sin fuerzas nuevas, sin tocar la dinámica validada.

## 4. Observable holístico (§3 del diseño)

Al final de cada corrida, fracciones de `E_total(0)`:
`{exergía libre (X final, informativa, no es parte del balance de energía sino de estructura),
mecánica residual (KE+PE final), reserva no gastada, ligada en estructura}`.
La fracción **ligada en estructura** = candidata a "materia" — comparada contra 4,9%/31,5%
SOLO al reportar (§1, T-target).

## 5. Barrido (sobredimensionado, todo junto por punto — nunca piezas por separado)

| Eje | Rango | Puntos |
|---|---|---|
| `amp_rugosidad` (ε) | {0.5, 1.0, 1.5, 2.5, 4.0} | 5 |
| `E_reserva` | logspace(1e-3, 1e3) × (KE(0)+\|PE(0)\|) de la corrida de referencia — 7 puntos, escaso→abundante | 7 |
| `cdm_on`, `cooling_on`, `expansion_on`, `gravedad_on` | {True, False} cada uno, sólo el punto base (todos True) se cruza con ε×E_reserva completo; el resto son celdas de control puntuales (apagar de a una, regla de admisibilidad de piezas) | 4 controles + 1 base |
| semillas (layout/CDM) | 8 (mínimo ya usado por CS073 para NULL) | 8 |

Total aproximado: 5×7×8 = 280 corridas del barrido principal + 4×8 = 32 controles de
admisibilidad de piezas + la comparación `E_reserva` finita vs infinita (§6) por cada punto
base. Sobredimensionado a propósito (regla del director); se reporta lo que salga.

## 6. Prueba de admisibilidad de la capa de energía (§5 del diseño)

Mismo punto (misma semilla, mismos ε/palancas), corrido dos veces:
(a) `E_reserva` finita (la del barrido), (b) `E_reserva = inf` (el gate nunca bloquea, todo
cluster energéticamente ligado se acredita — equivalente a `energia_on=False`).
Se compara la fracción ligada final (a) vs (b).

**Declarado ANTES de correr (honestidad del instrumento):** dado que el gate de Regla 4 es,
por construcción, la única vía por la que `E_reserva` puede afectar el resultado, una
diferencia entre (a) y (b) es esperable casi mecánicamente cuando `E_reserva` es escasa. Esto
NO invalida la prueba (sigue siendo cierto que si `E_reserva=inf` y `E_reserva` escasa dieran
el MISMO resultado, significaría que el gate nunca se activa en la práctica — sí sería un
hallazgo real de "la energía no actúa"). Lo que se reporta como hallazgo científico no es el
mero hecho de que difieran, sino la MAGNITUD de la fracción ligada bajo una reserva razonable
(ni absurdamente escasa ni absurdamente abundante) y su comparación de salida con 4,9%/31,5%.

## 7. Trampas — checklist

- **T-holística:** el observable de materia se lee sólo de la corrida completa
  (`correr_holistico_energia()`), nunca de una pieza aislada.
- **T1:** `E_reserva` se barre en rango amplio (log-espaciado, 6 décadas), nunca fijada a
  mano para pegarle a un número.
- **T-conservación:** el umbral duro del §3/Regla 1 (5% en el tramo de control) — falla
  explícita, no se sigue corriendo con contabilidad rota.
- **T-target:** 4,9%/31,5% sólo en el reporte final.
- **T-admisibilidad:** §6 (energía) + controles de apagado de una pieza a la vez (§5, ya
  parte del barrido) para las piezas reusadas (`cdm_on`, `cooling_on`, `expansion_on`,
  `gravedad_on`), coherente con el mecanismo que `pieza_base.py` ya exige para las 23
  piezas del motor validado.

## 7b. ADENDA de implementación (post-verificación, 2026-07-25) — no se edita §4 arriba

Al correr el smoke test (§Verificación paso 2/3) se encontró que `frac_ligada_estructura`
(ligada_acum / denom_frac, con `denom_frac = mecanica_ref + E_reserva_abs`) **no sirve para
comparar entre corridas con distinto `E_reserva`**: el denominador crece con la propia
`E_reserva` barrida, así que a `E_reserva=inf` (brazo "sin energía" de §6) da 0,0
trivialmente sin importar cuánta estructura se acreditó de verdad — inflaba artificialmente
el resultado de "difieren" en la prueba de admisibilidad, no por el mecanismo (que sí
funciona: `n_particulas_acreditadas` subió de 9 a 47 al pasar de reserva chica a grande en
el chequeo aislado), sino por un denominador mal elegido.

**Corrección:** se agrega `frac_masa_ligada = masa_acreditada / masa_bariones_total` —
denominador FIJO (la masa bariónica total NO depende de `E_reserva`) — como el observable
PRIMARIO para §4 (balance) y §6 (admisibilidad). `frac_ligada_estructura` (energía) se
conserva como diagnóstico secundario, documentado con esta limitación. La comparación con
4,9%/31,5% (§1, T-target) se hace sobre `frac_masa_ligada`, no sobre la versión de energía.

## 8. Qué se entrega a CS (yo), sin adjudicar

- Este documento (pre-registro).
- `cs074_energia_holistica.py`.
- Resultado crudo del smoke test (control de conservación en gravedad pura).
- Resultado crudo del barrido completo + la comparación de admisibilidad.
- Balance energético completo, tal como salga — ningún ajuste hacia 4,9%/31,5%.
- **NO se adjudica ni se cierra el experimento aquí** — CS lee qué emergió del conjunto,
  y el director decide.
