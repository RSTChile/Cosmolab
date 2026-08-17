# PRE-REGISTRO — Etapa 7 / Track B: masa inercial operacional por ancla gravitatoria externa

**Escrito:** 2026-07-23 05:11 (hora local, ANTES de correr smoke o producción).
**Autor:** agente delegado, sesión Cosmogénesis (dirección: Alexis López Tapia).
**Motor:** copia nueva `etapa7_trackB_ancla_masa_inercial.py`, prefijo `etapa7_trackB_ancla_`.
No se edita v1–v6 ni motor_1a7. El motor de campo/átomos (`medium_norm`, `weighted_cut`,
`components_strict`, `match_persist`, `toroidal_delta`) es una copia literal de v6 (E0–E3,
sin cambios de comportamiento) porque ese motor no se toca.

**Calibración previa (no toca el criterio de PASS, solo el rango del barrido):** se corrió
un smoke exploratorio de solo-E0-E3 (5 semillas, sin ancla) para ver cuántos átomos estables
emergen y su rango de `mass_proxy`, y elegir un barrido de `M_anchor` con escala razonable.
Resultado: 14–23 átomos por semilla al llegar a E4, `mass_proxy` ∈ [4.6, 14.6] típico. Esto
fija el barrido de `M_anchor` de abajo. Ningún umbral de PASS se decidió mirando resultados
del ancla — el ancla no se había corrido todavía cuando se fijaron los umbrales de este archivo.

---

## 0. Objetivo y por qué es un canal distinto del linaje

`co_member_score`, `n_long_co_pairs`, `fusion_events` **no se calculan en este script** (no
solo no se usan: no existen en el código). El único ingrediente de identidad usado es el id de
track de cada átomo (para saber que el átomo i del paso t es el mismo átomo físico), que no es
co-membresía ni linaje — es continuidad de objeto, igual que en Track D.

Idea (mandato de la misión): en física real, la masa inercial se mide operacionalmente aplicando
una fuerza CONOCIDA e independiente y midiendo la aceleración resultante (`m = F/a`). Aquí: un
**ancla gravitatoria externa** de masa `M_anchor` fija y conocida, en una o varias posiciones
fijas del grid, que **no es un átomo** — no participa de `match_persist`, no entra a ningún
cálculo de grupos/co-membresía/fusión, no tiene id de track. Ejerce la misma ley de fuerza que
usa `nbody_step` de v6 (mismo `G_GRAV`, `SOFTENING=1.2`, `FORCE_CUTOFF=8.0`), pero el ancla
**no acelera** (masa fija, posición fija — nunca se mueve, no hay reacción sobre ella).

---

## 1. Observable primario

Snapshot de átomos estables: se corre el motor E0→E3 (copia v6) hasta el primer paso dentro de
la ventana E4 (`step >= grav_start` y `frozen`) en que hay ≥ 2 átomos estables (`age >=
PERSIST_STEPS`). Ese snapshot (posiciones `(cy,cx)` y `mass_proxy` de cada átomo, ids de track)
se usa **una sola vez por semilla** — no se sigue evolucionando el campo φ/Φ más allá de ese
punto; el ancla se aplica sobre ese snapshot congelado (aislamos el efecto del ancla de la
dinámica mutua N-body entre átomos y de la evolución posterior del campo, que son harina de otro
costal — canal deliberadamente aislado, ver nota de diseño más abajo).

Para cada átomo `i` del snapshot, cada posición de ancla `(y_a,x_a)` del barrido, y cada
`M_anchor` del barrido:

```
dy = toroidal_delta(y_a, cy_i, L);  dx = toroidal_delta(x_a, cx_i, L)
r_i = hypot(dy, dx)
```

Si `r_i > FORCE_CUTOFF (8.0)`: el átomo está fuera de rango, `F_known=0`, se excluye de la
regresión (se cuenta y reporta, no se descarta silenciosamente).

Si `r_i <= FORCE_CUTOFF`:

```
F_known(r_i) = G_GRAV * M_anchor / (r_i^2 + SOFTENING^2)      [SOFTENING=1.2, igual que nbody_step]
```

`F_known` es "conocida e independiente": depende solo de `M_anchor`, `G_GRAV`, `r_i` — **nunca**
del `mass_proxy` del átomo que la recibe (a diferencia de `nbody_step`, donde la fuerza entre dos
átomos SIEMPRE es producto de ambas masas). Esa es la diferencia de diseño explícita frente a v6:
el ancla no "pesa" según quién la sienta, exactamente como un campo gravitatorio de prueba.

**Respuesta (aceleración observada) por modo:**

| modo | fórmula | lee |
|------|---------|-----|
| **REAL** | `a_obs_i = F_known(r_i) / mass_proxy_i` | F=ma con la masa PROPIA del átomo como divisor inercial |
| **SHUFFLE** | `a_obs_i = F_known(r_i) / mass_proxy_perm(i)` | F=ma con la masa de OTRO átomo del mismo snapshot (permutación aleatoria dentro del snapshot), repetido `N_SHUFFLE_REPEATS=8` veces con permutaciones distintas, se reporta la media |
| **OFF** | `M_anchor=0` → `F_known=0` → `a_obs_i=0` | control nulo — ningún ancla, ninguna fuerza |
| **INVERT** | `a_obs_i = -F_known(r_i) / mass_proxy_i` | repulsivo, misma masa propia — chequeo de signo, no de inercia |

Esta es una integración **explícita, nueva, propia de Track B** (no una llamada a `nbody_step`):
`nbody_step` de v6 usa `strength = G*m_i*m_j/r²` **directamente como aceleración, sin dividir por
masa** (convención propia de ese motor para fuerza mutua entre átomos, que no tocamos). Track B
**sí** divide por `mass_proxy_i` porque el mandato de la misión es probar F=ma explícitamente con
una fuerza que no dependa de la masa del receptor — si reusáramos la convención de `nbody_step`
sin dividir, la "aceleración" resultante sería `∝ mass_proxy_i` (más masa ⇒ más desplazamiento),
que es lo opuesto de inercia por construcción del código, no algo que dependa de los datos. Se
documenta esta decisión aquí, antes de correr nada, precisamente para que no parezca elegida
después de ver el resultado.

**Advertencia de circularidad, registrada de antemano:** dado que REAL se define exactamente como
`a_obs = F_known/mass_proxy_propio`, el ajuste de REAL contra su propio `mass_proxy` es casi
tautológico (coincide por álgebra, no por medición independiente) — el peso probatorio real de
este diseño está en la comparación **REAL vs SHUFFLE**: ¿el ajuste con la masa CORRECTA predice
mejor que con una masa AJENA (permutada) del mismo snapshot? Eso sí es una pregunta empírica
genuina (depende de cuánta varianza real de `mass_proxy` hay en cada snapshot y de si las
permutaciones producen ajustes visiblemente peores). Se reporta esta reserva en el resumen final,
no se oculta.

---

## 2. Métrica y regresión pre-registrada

Por semilla, pool sobre todas las combinaciones (posición de ancla × `M_anchor`) y todos los
átomos con `r_i <= FORCE_CUTOFF`:

```
y = 1 / a_obs_i     x = mass_proxy_i (SIEMPRE la etiqueta verdadera del átomo, incluso al evaluar SHUFFLE)
```

Ajuste lineal por mínimos cuadrados (`np.polyfit` grado 1): `slope`, `intercept`, `R²`.
Se calcula por separado para las muestras generadas en modo REAL y en modo SHUFFLE (media de las
`N_SHUFFLE_REPEATS=8` permutaciones).

**Diagnóstico adicional (no gatea PASS):** regresión log-log `log(a_obs) ~ β_m·log(mass_proxy) +
β_r·log(r) + const`, para reportar el exponente de masa `β_m` observado (bajo F=ma con fuerza
conocida se espera `β_m ≈ -1`; bajo la convención cruda de `nbody_step` sin dividir se esperaría
`β_m ≈ +1`). Se reporta, no decide el veredicto.

---

## 3. Sub-tests por semilla (solo REAL vs SHUFFLE, umbrales fijados aquí)

- **T1 — pendiente consistente:** `slope_REAL > 0` (F=ma con masa propia predice que `1/a_obs`
  crece con `mass_proxy` — más masa, menos aceleración, luego más "1/a").
- **T2 — ajuste y ventaja sobre SHUFFLE:** `R²_REAL >= R2_MIN (0.30)` **y**
  `(R²_REAL - R²_SHUFFLE) >= GAP_MIN (0.15)`.

`PASS_B(semilla) = T1 AND T2` (ambos, no "o" — a diferencia de Track D, aquí T1/T2 son
condiciones conjuntas de la misma hipótesis F=ma, no señales alternativas).

Requisito de datos mínimos por semilla: `n_atoms_snapshot >= 3` con `mass_proxy` distinto (si no,
la semilla se marca `insufficient_atoms` y no cuenta ni a favor ni en contra en `rate_B`, pero se
reporta).

---

## 4. Controles nulos (reportados; OFF gatea colapso)

- **OFF:** `a_obs_i == 0` para el 100% de los átomos en el 100% de las semillas (control
  determinista — con `M_anchor=0`, `F_known=0` siempre; si no da exactamente cero hay un bug, no
  un resultado). `frac_off_zero >= OFF_NULL_CLEAN_FRAC (0.99)` o el veredicto es INCONCLUSO.
- **INVERT:** dirección opuesta a REAL (`acc_i · (pos_ancla − pos_i) < 0` cuando en REAL es `> 0`,
  para el mismo átomo/ancla/M_anchor), magnitud `|a_obs|` idéntica a REAL — chequeo de signo, no
  gatea PASS/FAIL de la hipótesis de inercia.

---

## 5. Veredicto agregado (10 semillas estándar)

```
rate_B = fracción de semillas con PASS_B=True, sobre las semillas con datos suficientes
```

- **PASS** si `rate_B >= RATE_PASS (0.55)` y `frac_off_zero >= 0.99` y `>=5` semillas con datos
  suficientes.
- **PARTIAL** si `0.30 <= rate_B < 0.55` (nulls limpios, datos suficientes).
- **FAIL** si `rate_B < 0.30` (nulls limpios, datos suficientes).
- **INCONCLUSO** si `frac_off_zero < 0.99`, o si `<5` semillas tienen datos suficientes.

Umbrales (`R2_MIN=0.30`, `GAP_MIN=0.15`, `RATE_PASS=0.55`, `OFF_NULL_CLEAN_FRAC=0.99`,
`N_SHUFFLE_REPEATS=8`) se fijan en este archivo, antes de producción, y no se tocan después de
ver el JSON de producción. No se afinan solo en semillas que fallan.

---

## 6. Semillas y barrido

- **Semillas estándar:** 7, 42, 99, 777, 2025, 3141, 8191, 99991, 12345, 54321 (las 10 mismas de
  v1–v6/Track D, por comparabilidad de régimen físico — no hay dependencia circular: Track B no
  usa ningún output de linaje de esas corridas).
- **Barrido `M_anchor`:** `{5, 10, 20, 40, 80}` (elegido por la calibración de §0: `mass_proxy`
  típico 4.6–14.6, `G_GRAV=0.20`, `SOFTENING=1.2` ⇒ este rango da `F_known` ni despreciable ni
  saturante en `r` típicos 1–8).
- **Posiciones de ancla:** 5 puntos fijos del grid `L=28`: centro `(14,14)` + 4 puntos a un
  cuarto de cada esquina `(7,7), (7,21), (21,7), (21,21)` — cubren el toro con solapamiento
  parcial dado `FORCE_CUTOFF=8`.
- **Producción:** 10 semillas × 5 posiciones × 5 `M_anchor` = 250 combinaciones de ancla por
  semilla (pool de docenas a cientos de pares átomo×combinación por semilla, dado 14–23 átomos
  típicos).
- **Smoke previo:** 4 semillas (`7, 42, 99, 2025`), 2 posiciones (centro + `(7,7)`), 2 `M_anchor`
  (`10, 40`), `N_SHUFFLE_REPEATS=4` — antes de la corrida de producción completa.

---

## 7. Registro de iteración

- Hora de escritura de este documento: ver encabezado (2026-07-23 05:11).
- Hora de la corrida de smoke: 2026-07-23 05:16:00 → 05:16:16 (16.1 s, 4 semillas).
- Hora de la corrida de producción: 2026-07-23 05:16:36 → 05:17:13 (36.5 s, 10 semillas).
- El smoke no obligó a cambiar nada estructural de este documento: nulls OFF limpios (100%),
  chequeo de signo INVERT limpio (100%), sin errores — se corrió producción con los mismos
  umbrales fijados arriba, sin tocarlos. Ver `RESUMEN_TRACKB_ANCLA.md` para el veredicto.
