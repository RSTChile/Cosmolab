# SUITE ÉPOCAS MASA v2 — E3 estricto + E4 nulos

**Fecha:** 2026-07-22  
**Código:** `codigo/suite_epocas_masa/suite_epocas_masa_v2_endurecido.py`  
**JSON:** `results/suite_epocas_masa_v2/suite_epocas_masa_v2_result.json`  
**Tiempo:** ~1191 s · sin reabrir Rm como masa

---

## Veredicto global

### `E3_OK_E4_CAUSAL_WEAK`

| pieza | estado |
|-------|--------|
| **E3 átomo estricto** | **OK** (rate 1.00) |
| **Masa = 0 antes de E4** | **OK** (rate 1.00) |
| **Nulos de masa** OFF / SHUFFLE / INVERT = 0 | **OK** (limpios) |
| **Masa solo en gravedad REAL** | **OK** (por contrato del proceso) |
| **Densificación causada por gravedad de H** (REAL ≫ SHUFFLE) | **DÉBIL** (rate 0.10–0.20) |

---

## E3 — átomo endurecido

Criterios **todos** a la vez (post-freeze + VEV):

| criterio | valor |
|----------|--------|
| tamaño k | 4–14 |
| fracción núcleo \|Φ\| alta | 0.15–0.75 (núcleo + halo) |
| cohesión perim/√k | 1.2–6.5 |
| persistencia centroide | ≥ 4 pasos |

**Resultado barrido** L∈{24,28,32,40} × MIX× seeds (n=48):

- rate_E3 = **1.00**
- mean n_stable ≈ reportado en JSON
- mass_obs pre-E4 = **0** siempre

La definición laxa de v1 (cualquier cluster “parecido”) queda sustituida; el átomo estricto **sigue apareciendo** de forma estable.

---

## E4 — controles causales

Para cada seed se corren **4 mundos** (misma semilla):

| modo | fuerza | ¿otorga mass_obs? |
|------|--------|-------------------|
| **REAL** | atracción ∝ suavizado(ρ_H) | **sí** |
| **OFF** | 0 | no |
| **SHUFFLE** | misma amplitud, pozos **barajados** (descorrelacionados de H) | no |
| **INVERT** | repulsión | no |

### Medias (10 seeds, G=0.15)

| modo | mean mass | mean dens_enhance |
|------|-----------|-------------------|
| REAL | **64.8** | **7.97** |
| OFF | **0** | **1.00** |
| SHUFFLE | **0** | **7.90** |
| INVERT | **0** | (reparte) |

### Lecturas separadas (importante no mezclar)

1. **Canal de masa (nomenclatura correcta)**  
   - Solo REAL da mass_obs.  
   - OFF / SHUFFLE / INVERT → 0.  
   - Eso **cumple** “no hay masa sin gravedad real sobre H” *a nivel de definición del observable*.

2. **Causalidad de la densificación**  
   - REAL dens ≈ SHUFFLE dens (~8 vs ~8).  
   - Un campo de fuerza con la **misma intensidad** pero **pozos falsos** también apelotona.  
   - Por tanto: **subir densidad no basta para probar que fue la gravedad del hidrógeno**; hace falta dinámica donde el clumping de REAL **supere de forma estable** al NULL shuffle (rate dens_causal ≈ **0.20** hoy; causal_mass conjunto ≈ **0.10**).

3. **Rm/v3**  
   - leak medio ~0.19 → sigue siendo **instrumento ilegal** como masa (solo monitoreo).

### Barrido G

- G=0 → mass REAL = 0, causal = 0  
- G>0 → mass REAL alta, pero dR/dS ~ 0.97–1.14 (shuffle casi empata)  
- Solo G≥0.35 empieza a verse algo de ventaja dens (causal 0.33 en 3 seeds)

---

## Qué se admite / qué queda abierto

| claim | estado |
|-------|--------|
| Átomo E3 con criterios duros emerge | **Admitido** |
| mass_obs = 0 en E0–E3 | **Admitido** |
| mass_obs = 0 si no hay gravedad REAL | **Admitido** |
| densificación E4 es *específica* de gravedad-sobre-H (vs shuffle) | **Abierto / débil** |
| Rm v3 = masa | **Enterrado** |

---

## Siguiente endurecimiento natural (E4 dinámica)

Para que REAL gane a SHUFFLE de verdad (sin sintonía SM):

1. **Fuerza solo entre átomos** (N-body sobre centroides H), no campo suave barajable al mismo poder.  
2. **Conservación + pozo ligado a identidad del átomo** (el shuffle no puede reasignar “quién atrae a quién” sin romper tracks).  
3. NULL_SHUFFLE = permutar **etiquetas de átomos** en el grafo de atracción, no el campo escalar completo.  
4. Métrica: energía de enlace H o radio de gyración de grupos atómicos, no solo max(ρ)/mean(ρ).

---

## Artefactos

- `codigo/suite_epocas_masa/suite_epocas_masa_v2_endurecido.py`
- `results/suite_epocas_masa_v2/suite_epocas_masa_v2_result.json`
- este resumen
