# E5.2-2 · Anticorrelación exergía↔entropía: ¿X baja exactamente lo que S_ent sube?

**Pre-registro fechado.** Fecha/hora de redacción (UTC): 2026-07-24T20:37:17Z, ANTES de correr
el motor. Regla T3: si algo falla, se reporta — no se edita esto después.

**Base física:** motor de `cs074_rcruz.py` (NO editado — se importa como módulo: `paso_difusion`,
`paso_expansion`, `campo_inicial`, `medir_D`, `medir_pasos_lavado`). Mismo campo continuo φ en
anillo de N nodos, difusión solo por aristas vivas, expansión = corte Bernoulli de aristas con
probabilidad H por paso, r = H/D (razón expansión/difusión), H = min(r·D, 1) con D MEDIDO del
propio campo (no impuesto).

Axiomas E1 (conservación del presupuesto total) y E2 (la expansión redistribuye E latente en
exergía, no la crea) se heredan del motor tal cual — no se tocan ni se activan/desactivan aquí
(eso es objeto de E5.3-4, no de este experimento).

---

## 1. Definiciones exactas (ANTES de correr)

### X(t) — Exergía (desviación cuadrática del equilibrio uniforme)

```
X(t) = (1/N) · Σ_i  (φ_i(t) − 1)²
```

φ_eq = 1 es el "estado muerto" de referencia: el valor del fondo uniforme (`fondo = np.ones(N)`)
con el que arranca `campo_inicial` antes de sumar la perturbación ε·pert. X mide la capacidad de
trabajo disponible como el momento cuadrático de la desviación respecto de ese equilibrio fijo
— es la forma estándar de exergía cerca de equilibrio para un potencial cuadrático (análoga a
energía libre ∝ (Δφ)²). Es una cantidad de MOMENTOS (suma de cuadrados respecto a una
referencia fija).

### S_ent(t) — Entropía (Shannon de la densidad de energía espacial)

```
p_i(t) = φ_i(t)² / Σ_j φ_j(t)²
S_ent(t) = − Σ_i p_i(t) · ln(p_i(t))
```

Se usa φ² (no φ) como "densidad de energía" para garantizar p_i ≥ 0 siempre (φ puede volverse
negativo para ε grande; φ² no). p_i es la fracción de "energía" (φ²) que reside en el sitio i;
S_ent es la entropía de Shannon de esa distribución de probabilidad SOBRE EL ESPACIO (no sobre
los valores). Campo uniforme (equilibrio) → p_i uniforme → S_ent → ln(N) (MÁXIMO). Campo muy
estructurado/concentrado → p_i concentrado en pocos sitios → S_ent bajo. Esta es una cantidad de
PROBABILIDAD/LOGARITMO (no de momentos).

### Independencia (anti-T2)

X y S_ent se calculan por vías algebraicas distintas: X es una suma de cuadrados respecto a una
constante fija externa (φ_eq=1, no depende de la distribución de φ); S_ent es una entropía de
Shannon sobre una distribución normalizada de φ². Ninguna se define en términos de la fórmula
de la otra — no hay division circular, no hay X=f(S_ent) ni S_ent=g(X) algebraico. La relación
entre ambas (si existe) es un HALLAZGO EMPÍRICO del barrido, no una identidad impuesta. (Nota:
para campos gaussianos cerca de equilibrio ambas cantidades típicamente covarían con la
dispersión del campo — eso es precisamente lo que este experimento pone a prueba, no lo que
asume.)

Juez ≠ observable: ninguna de las dos cantidades decide por sí misma el PASS; el juez es la
correlación de Pearson entre las dos series temporales, evaluada contra su propio NULL.

---

## 2. Barrido (sobredimensionado, regla del director)

- **N = 200** (misma escala que `cs074_rcruz.py modo=produccion`).
- **ε** ∈ {0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0} — 8 valores, 12 décadas de rango (igual
  rango extremo que E5.1-1/E5.3-1), incluyendo ε=0 como control degenerado.
- **r** ∈ {0, 1e-3, 1e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000} — 12 valores, rango extremo
  1e-3…1e3 (igual que pide la spec) + r=0 como control (sin expansión).
- **Semillas:** 16 por celda (ε,r), semillas base 5000..5015.
- **Pasos por corrida:** calibrados (no a mano) igual que el motor base: se mide el lavado
  (`medir_pasos_lavado`, P_LAVADO=0.05) a ε=1e-2 (representativo, ni extremo ni degenerado) y se
  usa `pasos = ceil(mediana_lavado × 1.15)` fijo para todo el barrido (mismo criterio que
  `cs074_rcruz.py modo=produccion`). Se registra el valor calibrado en el resultado.
- **Grid total:** 8 ε × 12 r × 16 semillas = 1536 corridas, cada una con serie temporal
  {X(t), S_ent(t)} de longitud = pasos, un punto por paso.

## 3. NULL — barajado temporal

Por cada corrida (ε, r, semilla) ya evolucionada, se obtienen las dos series reales
{X(t_k)}, {S_ent(t_k)}, k=1..pasos. El NULL se construye permutando el ORDEN temporal de una de
las series (X) con una permutación aleatoria π (RNG derivado de la misma semilla + sufijo
"null", reproducible) manteniendo S_ent(t) en su orden original:

```
NULL: corr(X[π(1..T)], S_ent[1..T])
```

Esto "reordena los pasos" (rompe la correspondencia t↔t que sería la relación causal
instantánea) pero "conserva los valores" (el conjunto de valores de X y de S_ent no cambia, solo
su emparejamiento temporal). Se reporta también el barajado inverso (permutar S_ent en vez de X)
como verificación secundaria.

## 4. Juez y umbral (PASS congelado antes de correr)

Por cada corrida: `r_pearson_real = corr(X(t), S_ent(t))` y `r_pearson_null = corr(X[π](t), S_ent(t))`.

- **PASS (anticorrelación fuerte y específica):** r_pearson_real < −0.9 en REAL, agregando por
  (ε,r) sobre las 16 semillas (media y mediana), Y r_pearson_null claramente por encima de −0.9
  (ausente) en las mismas celdas.
- **Negativo honesto:** si r_real no cruza −0.9 en ningún régimen, o si r_null también cae
  fuerte (el barajado no logra romper la anticorrelación → sospecha de artefacto de definición,
  T2), se reporta tal cual, sin ajustar.
- Casos degenerados (ε=0 o ε=1e-12 con X≈0 constante ⇒ varianza≈0 ⇒ correlación indefinida) se
  marcan NaN y se excluyen del agregado, reportados aparte.

## 5. Qué se entrega crudo a CS

- Tabla completa r_pearson_real y r_pearson_null por (ε, r), media/mediana/std entre 16
  semillas.
- Al menos 3 trayectorias X(t) vs S_ent(t) de ejemplo (r bajo, r≈1, r alto) para inspección
  visual.
- Veredicto sin suavizar: ¿en qué región de (ε,r) el PASS se cumple, si en alguna?

## 6. Archivos

- Motor: `E5_2_2_motor.py` (importa `cs074_rcruz.py` sin editarlo).
- Resultados crudos: `E5_2_2_resultados.json`.
- Este pre-registro: `E5_2_2_PROTOCOLO_PREREGISTRO.md`.

**Firmado (pre-registro, antes de correr):** agente E5.2-2, 2026-07-24T20:37:17Z.
