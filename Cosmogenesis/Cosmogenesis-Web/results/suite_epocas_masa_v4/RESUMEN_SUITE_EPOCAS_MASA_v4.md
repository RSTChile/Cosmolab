# SUITE ÉPOCAS MASA v4 — pares mutuos + linaje de fusión

**Fecha:** 2026-07-22/23  
**Código:** `codigo/suite_epocas_masa/suite_epocas_masa_v4_fusion_lineage.py`  
**JSON:** `results/suite_epocas_masa_v4/suite_epocas_masa_v4_result.json`  
**Tiempo:** ~2412 s · sin Rm-as-masa · umbrales no inflados a mano

---

## Veredicto global

### `E3_OK_E4_PARTIAL_lineage_weak`

| pieza | estado |
|-------|--------|
| E3 átomo estricto | **OK** (rate **1.00**) |
| mass_obs pre-E4 = 0 | **OK** |
| mass OFF / SHUFFLE / INVERT = 0 | **OK (nulls limpios)** |
| mass solo en gravedad REAL | **OK (gate)** — 5/10 seeds con mass REAL>0 |
| enlace mutuo REAL ≻ SHUFFLE (mutR/S ≥ 1.25) | **DÉBIL** (rate causal conjunto **0.30**) |
| **linaje / co-membresía REAL ≻ SHUFFLE** | **FUERTE** (rate lineage **0.90**) |
| dens REAL ≻ dens SHUFFLE | **aún débil** (~41 vs ~41) |

---

## Qué cambió respecto a v3

| v3 | v4 |
|----|-----|
| Fuerza all-to-all (softening) | **FORCE_CUTOFF=8** (no todo atrae a todo) |
| E_bind total entre todos los pares | **E_mutual** solo pares con co-proximidad ≥5 pasos (por ID) |
| Juez = bind_strength R/S | Juez = mutual_bind R/S **+** (n_mutual R/S **o** co_member R/S) |
| mass_obs ∝ E_bind × dens × grupos | mass_obs ∝ **E_mutual** × dens × grupos (solo si n_mutual≥1) |
| — | **Linaje:** co-membresía de grupos a lo largo de E4 + eventos de fusión |

Bug de medición v4.0 (corregido antes del run canónico): mass_obs se anulaba por producto n_mutual×n_groups y por muestrear solo hist cada 20 pasos → se usa `mass_obs_max` en todos los pasos E4.

---

## Controles (10 seeds, G=0.20)

| métrica | REAL | SHUFFLE | OFF / INVERT |
|---------|------|---------|--------------|
| mass | **~1173** (media; 5 seeds >0) | **0** | **0** |
| mutual_bind | ~32.6 | ~43.9 | — |
| n_mutual_stable | ~11.7 | ~11.4 | — |
| co_member_score | **0.121** | 0.088 | — |
| fusion_events | ~23 | ~26 | — |
| dens_enhance | ~41.4 | ~40.7 | — |

### Rates

| rate | valor | umbral PASS |
|------|-------|-------------|
| E3 | **1.00** | 0.55 |
| E4 causal (stack completo) | **0.30** | 0.55 |
| lineage_causal | **0.90** | 0.55 |
| gyr_causal | 0.40 | — |

**Seeds e4_causal=True:** 42, 99, 99991 (3/10).  
**Seeds lineage_causal=True:** 9/10 (solo falla 2025).  
**Seeds mass REAL>0 con nulls limpios:** 42, 99, 8191, 99991, 54321 (5/10).

### Ratios finitos (sin ∞ por mutS=0)

| ratio | media finita | n seeds finitos |
|-------|--------------|-----------------|
| mutual_bind R/S | **0.74** | 9 |
| n_mutual R/S | **0.56** | 9 |
| co_member R/S | **1.42** | 10 |
| fusion R/S | **1.20** | 10 |

*(Los ∞ del JSON cuando SHUFFLE mutual=0 inflan la media cruda; la media finita es la lectura honesta.)*

---

## Lectura (lo importante)

1. **Canal de masa sigue limpio:** sin gravedad REAL no hay `mass_obs`. El contrato temporal E4 se sostiene a nivel de nomenclatura.

2. **E_mutual no es el discriminante que esperábamos.** En media, SHUFFLE tiene *más* mutual_bind que REAL (43.9 vs 32.6). El shuffle de identidades **sí** puede formar pares próximos estables — la energía de pares mutuos **no** prueba por sí sola “gravedad del hidrógeno”.

3. **El linaje de co-membresía sí separa.** `co_member_score` REAL/SHUFFLE ≈ **1.42**, rate lineage **0.90**. REAL mantiene a los mismos átomos juntos en el mismo grupo a lo largo de E4 más que el grafo con fuentes barajadas. Eso es progreso real vs v3 (donde el juez era solo E_bind total).

4. **El stack causal completo sigue en 0.30** porque exige *también* mutual_bind R/S ≥ 1.25 (que falla a menudo). Si el juez se basara solo en linaje+mass nulls, el rate subiría — **pero no redefinimos el umbral a posteriori para fabricar PASS.** Se reporta partial y se deja el juez pre-registrado.

5. **Rm/v3** leak ~0.19 → sigue **no** siendo masa.

6. Barrido G: causal G>0 ~0.28; lineage G>0 ~0.75. G=0 → mass=0 y causal=0. L smoke causal ~0.17 (más exigente).

---

## Cadena del relato (estado tras v4)

```text
E3 átomos H estrictos              →  OK (1.00)
E4 mass solo REAL                  →  OK (nulls limpios)
E4 mutual energy REAL ≻ SHUFFLE    →  NO robusto (media R/S < 1)
E4 co-membresía / linaje R ≻ S     →  SÍ (rate 0.90, R/S≈1.42)
E4 stack causal pre-registrado     →  PARTIAL 0.30 (igual orden que v3)
Rm/v3                              →  no masa
```

---

## Siguiente natural (si se sigue) — sin sintonía SM

1. **Juez pre-registrado v5:** promover co-membresía / árbol de fusión como métrica principal y dejar mutual_bind como diagnóstico (no al revés). Eso es cambio de *protocolo*, no de umbral a ojo tras ver el dato.
2. Softening + cutoff acotados al radio atómico medido (no constante 8).
3. Conservación de momento en N-body (hoy hay fuerza sin reacción simétrica perfecta en shuffle).
4. No reabrir Rm como masa.

---

## Artefactos

- `codigo/suite_epocas_masa/suite_epocas_masa_v4_fusion_lineage.py`
- `results/suite_epocas_masa_v4/suite_epocas_masa_v4_result.json`
- `results/suite_epocas_masa_v4/run.log`
- este resumen
