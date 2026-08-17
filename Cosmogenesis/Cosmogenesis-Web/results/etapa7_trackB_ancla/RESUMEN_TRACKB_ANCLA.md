# TRACK B — masa inercial operacional por ancla gravitatoria externa

**Fecha:** 2026-07-23
**Pre-registro:** `codigo/suite_epocas_masa/PROTOCOLO_TRACKB_ANCLA_PREREGISTRO.md`
**Código:** `codigo/suite_epocas_masa/etapa7_trackB_ancla_masa_inercial.py`
**JSON:** `results/etapa7_trackB_ancla/trackB_smoke_result.json`,
`results/etapa7_trackB_ancla/trackB_production_result.json`

---

## 0. Cronología (pre-registro ANTES de correr)

| hora | evento |
|------|--------|
| 05:11 | `PROTOCOLO_TRACKB_ANCLA_PREREGISTRO.md` escrito y congelado (criterio de PASS, fórmulas, semillas, barrido) |
| 05:16:00–05:16:16 | smoke (4 semillas, 2 posiciones de ancla, 2 `M_anchor`, 4 repeticiones shuffle) |
| 05:16:36–05:17:13 | producción (10 semillas, 5 posiciones, 5 `M_anchor`, 8 repeticiones shuffle) |

El smoke no reveló ningún problema estructural (nulls OFF limpios, chequeo de signo INVERT
limpio) → se corrió producción **con los mismos umbrales**, sin tocarlos tras ver datos.

---

## 1. Fórmula y criterio pre-registrado (verbatim)

```
F_known(r) = G_GRAV * M_anchor / (r^2 + SOFTENING^2)      # NO depende de mass_proxy del átomo
REAL:    a_obs_i = F_known(r_i) / mass_proxy_i             # F=ma con masa PROPIA
SHUFFLE: a_obs_i = F_known(r_i) / mass_proxy_perm(i)       # F=ma con masa AJENA permutada (x8 repeticiones)
OFF:     M_anchor = 0 -> a_obs_i = 0
INVERT:  a_obs_i = -F_known(r_i) / mass_proxy_i            # repulsivo, chequeo de signo

Regresión primaria (pooled sobre átomos x semillas x M_anchor x posición-ancla, r_i<=FORCE_CUTOFF):
  y = 1/a_obs_i   vs   x = mass_proxy_i   (lineal, mínimos cuadrados)

T1(semilla) = slope_REAL > 0
T2(semilla) = R2_REAL >= 0.30  AND  (R2_REAL - R2_SHUFFLE) >= 0.15
PASS_B(semilla) = T1 AND T2

rate_B = fracción de semillas con PASS_B=True (de las semillas con >=3 átomos en snapshot)
PASS si rate_B>=0.55 y frac_off_zero>=0.99 y >=5 semillas con datos suficientes
PARTIAL si 0.30<=rate_B<0.55 (nulls limpios)
FAIL si rate_B<0.30 (nulls limpios)
INCONCLUSO si nulls sucios o <5 semillas con datos
```

Ancla: **no** es un átomo — no tiene id de track, no entra a `match_persist`, ningún cálculo de
grupos/co-membresía/fusión la toca. `co_member_score`, `n_long_co_pairs`, `fusion_events` **no
se calculan en `etapa7_trackB_ancla_masa_inercial.py`** — no solo no se usan como ingrediente,
no existen en el archivo. Confirmado por lectura del código: cero apariciones de esos tres
nombres.

---

## 2. Semillas y barrido

- **Semillas estándar:** 7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321 (las 10 mismas de
  v1–v6, usadas solo por comparabilidad de régimen físico del motor E0–E3; ningún output de
  linaje de esas corridas se usa aquí).
- **Barrido `M_anchor`:** {5, 10, 20, 40, 80} (calibrado antes de fijar el criterio: `mass_proxy`
  típico 4.6–14.6 en el snapshot E4).
- **Posiciones de ancla:** centro `(14,14)` + 4 puntos `(7,7),(7,21),(21,7),(21,21)` del grid
  `L=28`.
- Producción: 10 semillas × 5 posiciones × 5 `M_anchor` = 250 combinaciones/semilla,
  `N_SHUFFLE_REPEATS=8`.

---

## 3. Números crudos — producción (10 semillas, criterio pre-registrado)

| seed | n_atoms | n_inrange | T1 (slope>0) | R²pool REAL | R²pool SHUF | gap | PASS_B |
|-----:|--------:|----------:|:---:|------:|-------:|-------:|:---:|
| 7 | 17 | 135 | True | 0.015 | 0.018 | -0.003 | False |
| 42 | 20 | 130 | True | 0.117 | 0.001 | 0.116 | False |
| 99 | 17 | 115 | True | 0.045 | 0.002 | 0.043 | False |
| 777 | 23 | 165 | True | 0.051 | 0.001 | 0.050 | False |
| 2025 | 14 | 95 | True | 0.081 | 0.001 | 0.080 | False |
| 3141 | 18 | 105 | True | 0.085 | 0.001 | 0.084 | False |
| 8191 | 20 | 120 | True | 0.042 | 0.000 | 0.042 | False |
| 99991 | 16 | 90 | True | 0.031 | 0.000 | 0.031 | False |
| 12345 | 22 | 165 | True | 0.049 | 0.000 | 0.049 | False |
| 54321 | 25 | 160 | True | 0.042 | 0.000 | 0.042 | False |

- **rate_B = 0.0** (0/10 seeds con PASS_B=True) — T1 se cumple 10/10, T2 falla 10/10 porque
  `R²pool_real < 0.30` en las 10 semillas (rango 0.015–0.117), aun cuando en 9/10 semillas
  REAL supera claramente a SHUFFLE en magnitud relativa (gap positivo, hasta ×100 de ratio),
  solo insuficiente frente al umbral absoluto pre-registrado de 0.30.
- **frac_off_zero = 1.00** (10/10) — el null OFF es perfecto: `a_obs=0` en el 100% de las
  muestras cuando `M_anchor=0`, en las 10 semillas.
- **n_invert_sign_ok = 10/10** — INVERT siempre produce aceleración de signo opuesto a REAL en
  igualdad de átomo/ancla/`M_anchor` (chequeo de consistencia, no gatea PASS).
- Tiempo: smoke 16.1 s (4 semillas), producción 36.5 s (10 semillas).

### Diagnóstico (pre-registrado como NO-gating, log-log controlando por r)

| seed | β_mass REAL | R²diag REAL | β_mass SHUFFLE | R²diag SHUF |
|-----:|------:|------:|------:|------:|
| 7 | -0.996 | 0.215 | -0.044 | 0.230 |
| 42 | -1.012 | 0.436 | -0.015 | 0.401 |
| 99 | -0.994 | 0.527 | -0.002 | 0.520 |
| 777 | -0.982 | 0.445 | 0.001 | 0.418 |
| 2025 | -0.951 | 0.493 | 0.035 | 0.387 |
| 3141 | -1.014 | 0.296 | -0.024 | 0.265 |
| 8191 | -0.859 | 0.507 | 0.104 | 0.461 |
| 99991 | -1.018 | 0.191 | 0.039 | 0.177 |
| 12345 | -0.987 | 0.365 | 0.010 | 0.340 |
| 54321 | -0.852 | 0.423 | 0.150 | 0.377 |

`β_mass` (exponente de `mass_proxy` en `log(a_obs) ~ β_mass·log(mass_proxy)+β_r·log(r)+const`)
es **muy estable en −1** (rango −0.85 a −1.02) para REAL en las 10/10 semillas — coincide
exactamente con lo que predice F=ma (`a=F/m`, dado que así se construyó `a_obs` en modo REAL).
Bajo SHUFFLE, `β_mass` colapsa a **≈0** (rango −0.044 a +0.150) en las 10/10 semillas: usar la
masa de OTRO átomo del snapshot destruye por completo la relación entre `mass_proxy` verdadero
y la aceleración observada. Este contraste es limpio y reproducible, pero **no gatea el
veredicto** (se pre-registró como diagnóstico) y tiene la reserva de circularidad de la §4.

---

## 4. Confirmación de no-uso de ingredientes prohibidos

Se verificó por lectura del archivo `etapa7_trackB_ancla_masa_inercial.py` completo: **no
aparece ninguna vez** `co_member_score`, `n_long_co_pairs` ni `fusion_events`, ni funciones
derivadas de ellos. El ancla no tiene id de track, no participa de `match_persist`, y no existe
ningún cálculo de grupos/co-membresía en este archivo. El único ingrediente de identidad es el
id de track de cada átomo dentro de su propio snapshot (continuidad de objeto, no linaje).

---

## 5. Veredicto honesto: **FAIL** (del criterio pre-registrado), con reservas importantes

- El criterio primario pre-registrado (regresión lineal pooled de `1/a_obs` vs `mass_proxy`,
  sin controlar por `r`) **FAIL** limpio: `rate_B=0.0/10`, nulls limpios, datos suficientes en
  las 10 semillas. No se retocó ningún umbral después de correr.
- **Causa identificada, no umbral movido:** la regresión pooled mezcla muestras de MUCHOS `r`
  distintos (barrido de posición de ancla y `M_anchor`) sin controlar por distancia. Dado que la
  relación verdadera es `1/a_obs = r²/(G·M_anchor)·mass_proxy` (hiperbólica en r, lineal en masa
  solo a `r` fijo), pooling sin controlar `r` diluye la señal — eso es lo que muestra el
  R²pool bajo (0.015–0.117) pese a que el diagnóstico log-log (que sí controla `r`) recupera
  `β_mass≈-1` casi exacto en las 10 semillas. Esta es una debilidad del **instrumento**
  (la métrica candidata sugerida en el mandato, tal como se pre-registró), no evidencia de que
  la hipótesis F=ma sea falsa en este juguete.
- **Reserva de circularidad (registrada ANTES de correr, no post-hoc):** el modo REAL se define
  literalmente como `a_obs = F_known/mass_proxy_propio` — su ajuste consigo mismo es
  cuasi-tautológico por construcción (no es una medición independiente de una dinámica externa
  al código que escribimos). El peso probatorio real de este diseño está en REAL vs SHUFFLE: la
  masa AJENA (permutada) destruye la señal (β_mass SHUFFLE≈0 vs REAL≈-1; R²pool SHUFFLE
  virtualmente cero en 9/10 semillas), lo que muestra que el snapshot (posiciones + masas
  emergentes de E0–E3) tiene suficiente varianza real para que la asignación CORRECTA de masa a
  átomo importe — un resultado no trivial, pero de "consistencia interna del código", no un
  descubrimiento de una física nueva emergente del motor (a diferencia del hallazgo de linaje,
  que emerge de dinámica N-body real entre átomos, no de una fórmula que nosotros mismos
  dividimos).
- **No se afirma** que esto sea equivalente al hallazgo de linaje (`co_member` R/S≈1.42): aquel
  es un patrón EMERGENTE de la dinámica mutua N-body ya existente en v6; este es una
  verificación de que una regla F=ma que **nosotros mismos implementamos** (fuera de
  `nbody_step`, que nunca divide por masa) es internamente consistente y distinguible de una
  mala asignación de etiquetas. Es un resultado modesto y honesto, no un cierre de "masa
  inercial confirmada".
- **Conclusión:** con el criterio tal como se pre-registró, Track B **no pasa**. El diagnóstico
  (no gating) sugiere que una versión mejor especificada del mismo diseño (controlando `r`
  explícitamente en el criterio primario, en vez del pooled sin controlar) probablemente sí
  pasaría — pero eso sería un **protocolo nuevo**, no una reinterpretación del actual, y no se
  ejecuta aquí sin autorización explícita del director (regla del proyecto: ningún cierre de
  arco sin autorización, y no se baja/cambia el criterio tras ver el dato).

---

## 6. Archivos escritos

- `Cosmogenesis-Web/codigo/suite_epocas_masa/PROTOCOLO_TRACKB_ANCLA_PREREGISTRO.md`
- `Cosmogenesis-Web/codigo/suite_epocas_masa/etapa7_trackB_ancla_masa_inercial.py`
- `Cosmogenesis-Web/results/etapa7_trackB_ancla/trackB_smoke_result.json`
- `Cosmogenesis-Web/results/etapa7_trackB_ancla/trackB_production_result.json`
- `Cosmogenesis-Web/results/etapa7_trackB_ancla/RESUMEN_TRACKB_ANCLA.md` (este archivo)
