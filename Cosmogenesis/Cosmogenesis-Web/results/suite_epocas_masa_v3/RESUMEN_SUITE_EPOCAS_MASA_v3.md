# SUITE ÉPOCAS MASA v3 — N-body atómico

**Fecha:** 2026-07-22  
**Código:** `codigo/suite_epocas_masa/suite_epocas_masa_v3_atomic_nb.py`  
**JSON:** `results/suite_epocas_masa_v3/suite_epocas_masa_v3_result.json`  
**Tiempo:** ~1574 s · sin Rm-as-masa

---

## Veredicto global

### `E3_OK_E4_PARTIAL_bind_sep_weak`

| pieza | estado |
|-------|--------|
| E3 átomo estricto | **OK** (1.00) |
| mass_obs pre-E4 = 0 | **OK** |
| mass OFF / SHUFFLE / INVERT = 0 | **OK (nulls limpios)** |
| mass solo en gravedad REAL entre átomos | **OK (gate)** |
| enlace REAL > SHUFFLE de identidades (causal) | **PARCIAL** (rate **0.30**, media bindR/S **1.11**, umbral 1.25) |
| dens REAL > dens SHUFFLE | **aún débil** (a veces SHUFFLE densifica más) |

---

## Qué cambió respecto a v2

| v2 (campo suave) | v3 (N-body atómico) |
|------------------|---------------------|
| Fuerza = gradiente de ρ_H suavizado | Fuerza **par a par** entre centroides de átomos H |
| SHUFFLE barajaba el **campo** (misma amplitud) | SHUFFLE permuta **quién es fuente de masa** en el grafo de atracción |
| dens REAL ≈ dens SHUFFLE → no causal | mass nulls **limpios**; bind REAL/SHUFFLE **puede** separarse (hasta ×1.78 en seeds buenos) |

Progreso real: el canal de **masa** ya no se “escapa” por el null de pozos; el cuello es la **separación de enlace** (aún ~30% de seeds en el umbral pre-registrado).

---

## Controles (10 seeds, G=0.20)

| modo | mean mass | mean \|E_bind\| | dens |
|------|-----------|-----------------|------|
| **REAL** | **~5028** | **~298** | ~41 |
| **OFF** | **0** | ~0 | — |
| **SHUFFLE** | **0** | ~282 | ~45 |
| **INVERT** | **0** | — | — |

Seeds con **e4_causal=True** (bindR/S ≥ 1.25 + mass REAL + nulls):  
**42** (1.31), **3141** (1.37), **99991** (1.78) → **3/10**.

---

## Barridos

- **G=0:** mass=0, causal=0  
- **G>0:** mass REAL alta; rate causal promedio **~0.25** (variable con G; picos 0.50 en G=0.05, 0.20, 0.35)  
- **L:** causal ~0.08 en smoke (más exigente en redes grandes con menos densidad de pares)

Escala de `mass_obs` es **arbitraria del juguete** (producto enlace×dens×grupos); lo que importa es **contraste vs nulos**, no el número absoluto ni 1/1836.

---

## Lectura en el lenguaje de la Teoría

1. **Átomos H** (E3 estricto) están.  
2. **Masa** no se asigna hasta que hay **gravedad entre esos átomos** (E4 REAL).  
3. Si apagas la gravedad, o **barajas quién atrae a quién**, o inviertes: **masa = 0**.  
4. Eso es más fiel a “gravedad sobre el hidrógeno” que el campo suave de v2.  
5. Aún no es robusto al 55%+ en **enlace causal** REAL≻SHUFFLE: a veces el grafo barajado produce |E_bind| comparable (fuerzas de la misma familia de magnitudes).  
6. Rm/v3 sigue siendo leak de nombre (~0.19), **no masa**.

---

## Siguiente endurecimiento posible (si se sigue)

- Energía de enlace **solo entre pares que permanecen mutuos** ( foreing-source shuffle no crea pares “estables” en tracks).  
- Métrica de **árbol de fusión** de grupos H (REAL forma linajes; SHUFFLE no).  
- Softening / cutoff de fuerza acotado al radio atómico (menos “todo atrae a todo”).  
- No subir umbrales a mano para fabricar PASS.

---

## Artefactos

- `codigo/suite_epocas_masa/suite_epocas_masa_v3_atomic_nb.py`
- `results/suite_epocas_masa_v3/suite_epocas_masa_v3_result.json`
- este resumen
