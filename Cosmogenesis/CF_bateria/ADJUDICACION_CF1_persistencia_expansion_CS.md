# CF-1 — Adjudicación: ¿la diferencia persiste bajo expansión?

**Serie:** CF (Cosmo-Física) · **ID:** CF-1  
**Fecha adjudicación (Grok, entrega a CS):** 2026-07-23  
**Pregunta simple:** una diferencia mínima en un campo caliente, ¿no se borra si el todo se expande más rápido de lo que se reabsorbe?  
**Estatuto de diseño:** re-sellado de **CS074-rcruz** existente — **sin reimplementar**.

---

## Artefactos (disco, no palabra)

| Archivo | Rol |
|---------|-----|
| `cs074_rcruz.py` | Código (Bernoulli expansion, pasos calibrados al lavado) |
| `cs074_rcruz_produccion_resultado.json` | N=200, 8 semillas, 8 ε, 10 r |
| `cs074_rcruz_robustez400_resultado.json` | N=400, misma grilla |
| `cs074_rcruz_chico_resultado.json` | N=100 smoke |
| `cs074_rcruz_comparacion_N.json` | Tablas N |
| `RESUMEN_CS074_rcruz_PARA_CS.md` | Informe previo |
| `RESUMEN_CS074_rcruz_robustez_N_PARA_CS.md` | Robustez N |

**No es CF-1:** `cs074_persistencia_campo.py` producción original (H fijo → r≫1, control H=0 roto). Queda como prehistoria.

---

## Criterios de la pregunta (lo que se sellaría)

1. ε=0 → P=0 (nada que preservar).  
2. r=0 (H=0) → difusión **lava** (P bajo).  
3. r↑ → P_real sube; REAL ≫ NULL (z alto).  
4. El umbral en r **no** se mueve de forma salvaje con N (mecanismo, no artefacto de tamaño).  
5. NULL de forma (permutación φ) muerde (T4).

---

## Evidencia numérica (media ε>0)

### Controles

| N | pasos | control r=0 (P_real) | ε=0 → P=0 | D medido |
|---|-------|----------------------|-----------|----------|
| 100 | 1553 | **0.032** | sí | 3.3e-3 |
| 200 | 6095 | **0.034** | sí | 8.4e-4 |
| 400 | 24380 | **0.034** | sí | 2.1e-4 |

### Curva P_real(r) — N=200 (producción)

| r | 0 | 0.1 | 0.5 | 1 | 10 | 100 |
|---|---|-----|-----|---|----|-----|
| P_real | **0.034** | 0.621 | 0.812 | 0.867 | 0.960 | 0.988 |
| z | 0.3 | **4.9** | 6.4 | 6.8 | 7.6 | 7.8 |

### Robustez N (umbrales de ubicación)

| métrica | N=100 | N=200 | N=400 |
|---------|-------|-------|-------|
| r con P>0.5 | 0.1 | 0.1 | 0.1 |
| r half-rise | 0.1 | 0.1 | 0.1 |
| max \|ΔP\| N200–N400 | — | **0.064** | — |

---

## Lectura para CS (cruda; no es sello final del director)

**Lo que la curva muestra (hecho de laboratorio):**

1. Sin expansión, la diferencia se reabsorbe.  
2. Con expansión relativa r=H/D creciente, la diferencia se congela y gana al NULL de forma.  
3. Independencia de ε (ε≥1e-9): misma curva (linealidad del juguete).  
4. Ubicación del cruce en el **mismo bin de r** al cambiar N 100→400.

**Caveats (honestos, no maquillar):**

- Grid de r grueso (0 → 0.1): no hay r\* continuo fino.  
- D un-paso subestima lavado multi-paso → el “r=1” nominal no es el punto crítico exacto; el **mecanismo** sí se ve.  
- N=800 del CS074 original no se re-corrió en r-cruz (coste); 100–400 cubren factor 4.  
- Expansión Bernoulli (corrección de `round(H·N)→0`) es parte del instrumento r-cruz, no del script original.

**Trampas T0–T7 en este re-sello:**

| T | ¿Respeta? |
|---|-----------|
| T1 | Sí — no se fija 7:1/GeV |
| T2 | Sí — P = forma×magnitud, no el juez de linaje |
| T3 | Sí — no se cambió el criterio tras la robustez N |
| T4 | Sí — NULL permuta forma |
| T5 | r=0 vs r>0 a ambos lados del comportamiento |
| T6 | control_r0_lava puede fallar (y en el original fallaba) |
| T7 | multi ε, multi r, multi seed, multi N |

---

## Propuesta de veredicto CF-1 (para sello del director/CS)

**`CF1_PASS_MECANISMO_PERSISTENCIA_EXPANSION`** (cualificado)

- Mecanismo: **probado** en el instrumento r-cruz.  
- No se afirma: “inflación del ME”, “η bariónica”, ni identidad con física de partículas.  
- Capa: **topológica / pre-partícula** (fila 1–2 de la línea de tiempo masa). Base legítima de la batería CF, no es CF-4.

**Acción:** sin código nuevo. El director/CS puede firmar o suspender. Grok **no** re-corre producción salvo pedido.

---

## Relación con la batería CF

CF-1 cierra (si se sella) el eslabón *diferencia + expansión*.  
CF-2 repara el eslabón *enfriar = expandir / dens*.  
CF-3/4 reubican la **masa** en épocas ME (Higgs / ligadura), fuera de E4 post-átomo.
