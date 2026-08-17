# HALLAZGO ABIERTO — Etapa 7 / V6: “masa” no es independiente del linaje

**Fecha de registro:** 2026-07-23  
**Origen:** auditoría del director (Alexis) sobre disco + Memanto; confirmada por revisión de código y JSON.  
**Estatuto:** **ABIERTO — no adjudicar como cierre; no vender chain_pass=True como cierre de la cadena 1→7.**  
**Localización:** etapa 7 del motor 1a7 / suite épocas masa v6. Las etapas **1–6** no muestran este patrón (r-cruz, ρ, E3, nulls).

---

## 1. Cronología en disco (por mtime; no por narrativa)

| orden | evento |
|-------|--------|
| 1 | `PROTOCOLO_1A7_PREREGISTRO.md` fija etapa 7 con **suite v5**, umbral rate≥0.55 |
| 2 | Corre **v5** → rate e4_lineage_pass = **0.40** (no PASS) |
| 3 | Motor 1→7 producción con v5 → **`chain_pass: false`** (log de producción) |
| 4 | Se escribe `PROTOCOLO_V6_...` y el propio texto justifica v6 **porque v5 falló** en mass_obs |
| 5 | Corre **v6** → rate = **0.80** → se etiqueta PASS |
| 6 | Se edita `pipeline.py` para que etapa 7 use **v6** en vez de v5 → nueva corrida → **`chain_pass: true`** |

**Lectura:** el juez pre-registrado del motor (v5) falló; se sustituyó el juez/observable de masa y con eso se recuperó el PASS de la cadena. Eso **no** es un pre-registro limpio del pipeline 1→7: el criterio de la etapa 7 se movió **después** de ver el negativo.

---

## 2. El problema de fondo: forma funcional, no solo “otro umbral”

### Fórmula v6 de `mass_obs` (REAL)

```text
mass_obs = dens × n_groups × n_long_co × co_member_score × gyr / N_atoms
```

### Criterio `lineage_wins` (heredado de v5, sin cambio)

Gana el linaje si (entre otras):

- `co_member_REAL / co_member_SHUFFLE ≥ 1.15`, **o**
- fusión REAL≻SHUFFLE, **o**
- `n_long_co` REAL≻SHUFFLE …

Es decir: **las mismas variables de linaje** (`co_member_score`, `n_long_co`, …) entran en:

1. el veredicto “¿hay linaje causal?”, **y**
2. la magnitud llamada “masa”.

No se midió una cantidad **independiente** que resultara correlacionada con el linaje. Se **construyó** “masa” multiplicando (parte de) lo que ya se sabía que separaba REAL de SHUFFLE.

### Comprobación en el JSON de producción v6

- En las **10 semillas**, `e4_lineage_pass` coincide **semilla a semilla** con `lineage_ok`.
- El umbral `mass_E4 ≥ 0.3` **nunca decide**: mass REAL ∈ ~43–284 en todas las semillas (incluidas las 2 que fallan: fallan por linaje, no por masa baja).
- El “gate de masa” es **decorativo**.
- El salto **0.40 → 0.80** es, en la práctica, la tasa de **`lineage_ok`** que **ya se conocía** en v4/v5 (~0.80–0.90), no un descubrimiento nuevo de que “la gravedad produce masa”.

**Analogía del director (aceptada):** definir masa = “cuánto atrae” y celebrar que la atracción predice la masa.  
**Familia del problema:** misma que observables construidos para el resultado (p.ej. debates de conteo/artefacto en CS072), aquí vía **forma del observable post-fallo**, no vía coeficiente numérico a mano.

---

## 3. Qué se retrae / qué se mantiene

### RETRACTADO como cierre

| claim previo | nuevo estatuto |
|--------------|----------------|
| v6 `E3_OK_E4_LINEAGE_CAUSAL_OK` = masa E4 causal probada | **NO** — como mucho: linaje REAL≻SHUFFLE + renombre |
| motor 1→7 `chain_pass=True` = cadena cerrada | **NO** — etapa 7 **abierta**; el true depende del v6 inválido como prueba de masa |
| “salto 0.40→0.80 sin bajar umbrales = mejora física” | **NO** — mejora de **definición**, no de mecanismo |

### SE MANTIENE (no contaminado por este hallazgo)

| tramo | por qué |
|-------|---------|
| CS074-rcruz + robustez N | NULL muerde; r=0 lava; umbral no se mueve con N |
| TEST_RHO (estiramiento / ρ) | controles propios |
| E0–E3 mass_obs=0, nulls OFF/SHUFFLE/INVERT=0 | siguen siendo resultados de modo, no tautología linaje↔masa |
| E3 rate 1.00 (átomo estricto) | independiente de la fórmula v6 |
| v4/v5: **linaje** co_member R/S≈1.42, rate lineage_ok alto | eso **sí** se midió *antes* de v6; el error es **llamarlo masa** y usarlo para cerrar la cadena |

### LO QUE SÍ SE PUEDE AFIRMAR CON HONESTIDAD SOBRE V6

- Como **ingeniería de etiquetas**: mass_obs REAL>0 siempre que hay linaje medible (por construcción).  
- Como **física de “la gravedad engendra masa”**: **no probado** por v6.

---

## 4. Qué haría falta para reabrir etapa 7 (diseño, no implementación aquí)

Criterio mínimo de independencia (propuesta para mesa, no ejecutada):

1. **Observable de masa** pre-registrado que **no** sea función monótona de `co_member` / `n_long_co` / fusión (las mismas que el juez de linaje).  
2. O bien: un solo nombre honesto — p.ej. solo se claima **linaje causal**, y se **retira** la palabra “masa” hasta tener otro canal (respuesta inercial, virial, scale-free mass ratio, etc.).  
3. El protocolo del motor 1→7 debe fijar la suite de etapa 7 **antes** del primer `chain_pass`, y un cambio de suite tras un FAIL es **nueva pre-inscripción**, no el mismo cierre.

---

## 5. Acción en el repo (esta sesión)

- Este archivo: marca el hallazgo **abierto**.  
- Se actualizan resúmenes de v6, motor 1a7 y canon de masa para **no** presentar chain_pass/v6 como cierre.  
- No se adjudica; no se reescribe v7 en silencio.

— Registro a partir de la auditoría del director; Grok (Diotallevi) documenta, no cierra.
