# SUITE ÉPOCAS MASA v5 — juez por linaje (pre-registrado)

**Fecha:** 2026-07-23  
**Protocolo:** `V5_LINEAGE_PRIMARY_2026-07-23`  
**Pre-registro:** `codigo/suite_epocas_masa/PROTOCOLO_V5_LINAJE_PREREGISTRO.md` (antes de la corrida)  
**Código:** `suite_epocas_masa_v5_linaje.py`  
**Tiempo:** ~2195 s  

---

## Veredicto global

### `E3_OK_E4_PARTIAL_lineage`

| pieza | valor | umbral |
|-------|-------|--------|
| rate E3 | **1.00** | 0.55 |
| mass nulls limpios | **True** | — |
| **rate e4_lineage_pass (juez v5)** | **0.40** | **0.55 → no PASS** |
| rate lineage_ok (sin mass) | **0.80** | — |
| rate v4 stack (diagnóstico) | 0.30 | (no gatea) |
| co_member R/S media | **1.42** | 1.15 |
| mutual R/S media (diag) | no confiable (∞ en seeds mutS=0) | — |

Seeds **e4_lineage_pass=True:** 42, 8191, 99991, 54321 (4/10).

---

## Qué cambió respecto a v4

| v4 | v5 |
|----|-----|
| PASS exigía mutual_bind R/S ≥ 1.25 | **No** — mutual es diagnóstico |
| lineage era sub-flag (0.90) | **Juez primario** (pre-registrado) |
| rate stack 0.30 | rate lineage-pass **0.40** (sube, no llega a 0.55) |

No se bajó COMEM_VS_SHUFFLE_MIN ni MASS_REAL_MIN tras ver el dato.

---

## Lectura

1. **Nulls de masa siguen limpios** — el contrato temporal se sostiene.  
2. **El linaje separa** (rate_lineage_ok 0.80, coR/S 1.42).  
3. **El PASS completo se queda en 0.40** porque exige *también* mass_obs REAL>0: varias semillas tienen linaje bueno y **mass=0** (E_mutual / n_mutual no encienden mass_obs).  
4. Seed 99: mass alta pero coR/S=1.00 exacto → falla linaje (empate con SHUFFLE).  
5. Barrido G: a G∈{0.05,0.10,0.15} e4_lin=1.00 en las 4 seeds de smoke; a G=0.20 del control global baja (dependencia de seed set).  
6. **No es PASS 0.55** — no se reclama cierre de E4 causal; se reclama que el juez correcto es linaje y el cuello restante es **asignación de mass_obs**, no la co-membresía.

---

## Siguiente natural (si se sigue)

- mass_obs acoplada a linaje (p.ej. ∝ co_member × dens × grupos) **pre-registrado en v6**, no improvisado.  
- O aceptar PARTIAL y cerrar el claim como: “linaje de fusión REAL≻SHUFFLE reproducible; masa numérica aún frágil en el juguete”.

## Artefactos

- `suite_epocas_masa_v5_result.json`  
- `run.log`  
- este resumen  
