# SUITE ÉPOCAS MASA v6 — mass_obs acoplada al linaje

**Fecha corrida:** 2026-07-23  
**Protocolo:** `V6_MASS_LINAJE_2026-07-23`  
**Código:** `suite_epocas_masa_v6_mass_linaje.py`  
**Tiempo:** ~2499 s  

---

## ⚠ ESTATUTO (auditoría director 2026-07-23): **NO ES CIERRE DE MASA**

Ver: `Cosmogenesis/HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md`

### Veredicto de corrida (números en disco) — **RETRACTADO como claim de masa**

El JSON reportó `E3_OK_E4_LINEAGE_CAUSAL_OK` con rate **0.80**.  
Eso **no** se acepta como “masa E4 causal probada”.

| hecho | detalle |
|-------|---------|
| Fórmula mass_obs | `dens × n_groups × n_long_co × co_member × gyr / N` |
| Juez lineage_wins | usa **co_member** y **n_long_co** (mismas piezas) |
| e4_lineage_pass vs lineage_ok | **idénticos** en las 10 semillas |
| mass_E4 ≥ 0.3 | **nunca decide** (mass REAL ~43–284 siempre) |
| 0.40 (v5) → 0.80 (v6) | = tasa de lineage_ok ya vista en v4/v5, no descubrimiento nuevo |

**Claim honesto residual de v6:** se reetiquetó linaje como “masa” y, por construcción, “masa” acompaña al linaje.  
**No claim:** que la gravedad produzca una magnitud de masa independiente.

### Cronología viciada respecto al motor 1→7

1. Protocolo 1a7 pre-registró etapa 7 = **v5**.  
2. v5 falló (0.40) → chain_pass false.  
3. Apareció v6 **porque v5 falló** → se sustituyó en el pipeline → chain_pass true.  

Eso es **cambio de juez post-fallo**, no pre-registro del cierre de cadena.

---

## Números de corrida (referencia; no “PASS de masa”)

| métrica | valor |
|---------|--------|
| rate E3 | 1.00 |
| mass nulls (modo OFF/SHUFFLE/INVERT) | limpios (sí: gate de **modo**, no tautología) |
| rate e4_lineage_pass (= lineage_ok) | 0.80 |
| co_member R/S | 1.42 |
| seeds fail | 99, 2025 (linaje empatado, no “masa baja”) |

---

## Qué se mantiene útil

- Nulls por **modo** (solo REAL puede tener mass_obs ≠ 0) siguen siendo disciplina de etiqueta.  
- El **linaje** como fenómeno (REAL≻SHUFFLE en co-membresía) viene de v4/v5 y no depende de llamarlo masa.  
- El error es el **cierre retórico** y el **uso de v6 para chain_pass**, no la existencia del JSON.

## Artefactos

- `suite_epocas_masa_v6_result.json` (números crudos)  
- `HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md` (estatuto)  
