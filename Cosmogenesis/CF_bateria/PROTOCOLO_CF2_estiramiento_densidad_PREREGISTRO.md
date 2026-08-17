# PROTOCOLO PRE-REGISTRADO — CF-2

**Fecha/hora pre-registro:** 2026-07-23 (ANTES de producción CF-2)  
**Serie:** CF · **ID:** CF-2  
**Pregunta simple:** al expandirse el espacio, ¿la temperatura/contraste se suaviza en espacio físico porque enfriar **es** expandir (estiramiento + dilución ρ∝a⁻³)?  

**Por qué se rehace:** `TEST_RHO_DISPERSION` tenía 1 semilla, sin barrido multi-seed de a, y el pipeline 1a7 lo verificaba de forma frágil (T7/T6).

---

## Barrido pre-registrado (congelado)

| Eje | Valores |
|-----|---------|
| Semillas | 7, 42, 99, 777, 2025, 3141, 8191, 99991 (**8**) |
| H_EXP (fija a_final = exp(H_EXP)) | **1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0** → a ∈ [e, e⁸] (~2.7 … ~2980) = varias décadas |
| L, PASOS, D0, W0 | 64, 400, 0.12, 1.2 (sello geométrico fijo; no se retocan post-dato) |

**Brazos (por cada seed × H_EXP):**

| Brazo | a(t) | ρ | D |
|-------|------|---|---|
| **REAL** | exp(H_EXP·t_g) | ρ0/a³ | D0·(ρ/ρ0) |
| **NULL_RHO_FIXED** | igual a(t) | ρ≡ρ0 | D0 constante |
| **NULL_A_FIXED** | a≡1 | ρ≡ρ0 | D0 |

(NULL_STRETCH opcional de reporte; no entra al PASS principal.)

---

## Observables

Por corrida, a lo largo de t_g:

- `A_phys(t) = A_comov(t) / a(t)` (abruptness del gradiente en espacio físico)  
- `w_phys(t)` ancho físico de la transición  
- `A_phys_ratio = A_phys_final / A_phys_init`  
- `w_phys_ratio = w_phys_final / w_phys_init`  

Agregados por (H_EXP, brazo) sobre semillas: media y std.

---

## PASS pre-registrado (congelado; no bajar tras ver datos)

Definiciones:

- **stretch_seed(H, seed):** en REAL, `A_phys_ratio < 0.25` **y** `w_phys_ratio > 2.0` (mismos umbrales de espíritu del test viejo; ahora multi-seed).  
- **rho_sep_seed(H, seed):**  
  `A_comov_final(REAL)` y `A_comov_final(NULL_RHO_FIXED)` difieren:  
  `|A_cR - A_cN| / max(A_cR, A_cN, 1e-12) ≥ 0.08`  
  (la dilución cambia el perfil comóvil).  
- **mono_REAL(H, seed):** a lo largo de muestras de la trayectoria REAL, `A_phys` es **no-creciente** en media por bins de a (pendiente de regresión log a → log A_phys ≤ 0).  
- **null_no_stretch_like_REAL:** en NULL_A_FIXED, `A_phys_ratio` **no** cumple stretch (ratio ≥ 0.25 o w_ratio ≤ 2) en ≥70% de semillas (el no-expandir no “estira”).

**PASS de un H_EXP:**  
`rate_stretch ≥ 0.70` (entre 8 semillas) **y** `rate_rho_sep ≥ 0.70` **y** `rate_mono ≥ 0.70`.

**PASS global CF-2:**  
Al menos **5 de 8** valores de H_EXP en el barrido dan PASS de H, **incluyendo** al menos un H_EXP con a_final > 50 y uno con a_final < 20 (rango, no un solo régimen).

**Si falla:** se reporta FAIL con la curva completa. No se edita este protocolo (T3).

---

## NULL que debe morder (T4)

- REAL vs NULL_RHO_FIXED: separación en A_comov o en suavizado (rho_sep).  
- REAL vs NULL_A_FIXED: stretch solo con expansión.  
Si REAL ≈ NULL_RHO_FIXED en todo el barrido → instrumento no discrimina dilución → **reportar**, no maquillar.

---

## Anti-trampas

| T | Cómo se evita aquí |
|---|-------------------|
| T1 | No GeV ni 1/1836 |
| T2 | Observable = gradiente físico, no linaje |
| T3 | Este archivo **antes** de producción |
| T4 | NULL dens fija + a fija |
| T5 | H_EXP bajos pueden no stretch → gate puede fallar |
| T6 | rate_stretch puede ser <0.70 |
| T7 | 8 H_EXP × 8 seeds × 3 brazos |
| T0 | Campo T(x) continuo; sin k impuesto |

---

## Entrega

- JSON crudo completo  
- Tabla A_phys_ratio(H_EXP) por brazo  
- Veredicto automático según criterios de arriba (sin narrativa cosmológica)

**Firma pre-registro Grok:** 2026-07-23, antes de `cf2_estiramiento_densidad.py` producción.
