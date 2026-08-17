# PROTOCOLO F2-4 — Expansión (H) y reabsorción (D) como ejes independientes
### ¿colapsa todo en r = H/D?

**Ejecuta:** CC · **Fecha/hora de fijación (UTC):** 2026-07-24T09:34:24Z
**Spec autoritativa:** `BATERIA_FUNDAMENTOS_F1_a_F4_PARA_CC.md`, sección F2-4 (líneas 152–159)
**Código base leído, NO editado:** `cs074_rcruz.py` (raíz de Cosmogenesis)

Este documento se fija ANTES de correr el motor (`F2_4_motor.py`). No se edita después
de ver resultados (T3). Cualquier desviación del plan se declara como tal en el reporte
final, no se disfraza cambiando el criterio.

---

## 1. Por qué esto no es solo "repetir cs074_rcruz con otro nombre"

En `cs074_rcruz.py`, **D nunca es una palanca barrida**: se MIDE del propio campo
(`medir_D`, fracción de contraste borrada en un paso de difusión pura, con el
coeficiente de mezcla fijo en 0.5 dentro de `paso_difusion`), y luego H se DERIVA de un
r objetivo × esa D medida (`H = min(r_target * D, 1.0)`). Es decir: **D está fijo en la
física (0.5), r se mueve moviendo H**. Todo el enfoque 2 hasta ahora barrió r como
cociente asumido — exactamente la trampa T1/T7 que este experimento existe para
cerrar.

Para que H y D sean dos ejes FÍSICAMENTE independientes (no una D medida y una H
derivada de un target-r), generalizamos el único lugar del modelo donde D vive: el
coeficiente de mezcla de la difusión. En `paso_difusion` de cs074_rcruz ese coeficiente
está fijado en la constante 0.5:

```
nuevo = phi + 0.5 * (media - phi)
```

Aquí se reemplaza el 0.5 por un parámetro `D` que se barre igual que `H`:

```
nuevo = phi + D * (media - phi)          # D ∈ (0, 1], reemplaza el 0.5 fijo
```

`D` sigue teniendo el mismo significado físico que en cs074_rcruz (fuerza de
reabsorción/relajación hacia el vecino por paso), y `H` sigue siendo exactamente la
misma probabilidad de corte de arista por paso que en cs074_rcruz
(`paso_expansion`, sin cambios de fórmula). El resto de la física (dominio periódico,
condición inicial de 5 modos de Fourier + fase aleatoria, `persistencia` = autocorr ×
varianza, NULL = permutación final) es IDÉNTICO a cs074_rcruz, no se toca nada más.

Como verificación adicional (no como criterio de PASS — solo diagnóstico), se mide
también `D_emp` = fracción de contraste borrada en un paso puro de difusión con ese `D`,
para confirmar que el parámetro barrido se comporta monótonamente como cabría esperar de
una tasa de reabsorción.

---

## 2. Diseño del grid (por qué esta forma exacta)

`H` y `D` usan la MISMA secuencia geométrica de 8 valores:

```
grid_8 = [0.080, 0.113, 0.160, 0.226, 0.319, 0.451, 0.637, 0.900]   (razón ≈1.413)
```

`H_list = grid_8`, `D_list = grid_8`. El barrido es el producto externo completo
8×8 = 64 combinaciones `(H_i, D_j)`.

**Por qué idénticas y no dos grids distintos:** con `H_list = D_list`, cada diagonal
`i − j = k` (k = −7…7) comparte EXACTAMENTE el mismo r = grid_8[i]/grid_8[j] = razón^k,
con escala absoluta de (H,D) distinta en cada punto de la diagonal. En particular la
diagonal k=0 da r=1 con 8 pares (H,D) DISTINTOS en magnitud absoluta (desde
H=D=0.080 hasta H=D=0.900). Esto es la prueba más directa posible de la pregunta del
experimento: mismos r, distintas (H,D) absolutas — si P difiere dentro de una misma
diagonal, r no es la variable que gobierna, aunque el cociente sea idéntico.

Rango de r resultante: r ∈ [0.089, 11.25] (7 diagonales a cada lado de k=0, cruza 1).
Es un rango más angosto que el de F2-1 (que resuelve r* fino) — aquí el objetivo no es
resolver el punto crítico sino testear el colapso mismo; se declara esta limitación de
alcance en el reporte.

**eps fijo, no barrido:** eps = 1e-2 (representativo, no extremo). F2-4 no pide barrer
eps (esa es la máquina de F1); barrerlo aquí multiplicaría el costo ×N sin agregar
información a la pregunta "¿colapsa H,D en r?". Se declara como alcance fijo.

**N fijo:** N = 200 (mismo N de "modo producción" de cs074_rcruz). No se barre N en
F2-4 (la ley r*(N) es objeto de F2-1). Alcance fijo, declarado.

**Semillas:** 12 (mínimo pedido por la spec, `≥12`). Cada semilla corre REAL y su NULL
pareado (misma semilla, mismo campo inicial, mismo camino de cortes) — igual que
cs074_rcruz.

**Perturbación dinámica (T7):** el corte de aristas es un proceso de Bernoulli
independiente por arista y por paso (`paso_expansion`), no un patrón fijo — cada una de
las 12 semillas recorre una TRAYECTORIA distinta de cortes a lo largo de los `pasos`
pasos, no solo una condición inicial distinta. Eso es la perturbación dinámica exigida;
F2-4 no agrega un tercer eje de amplitud de ruido porque la spec de este experimento no
lo pide (a diferencia de F1-5) y ya hay dos ejes físicos (H,D) más semilla-trayectoria.

---

## 3. Calibración de `pasos` (medida, no elegida para dar un resultado)

Se mide el tiempo de lavado (pasos para que P < `P_LAVADO=0.05` a H=0) en el D MÁS LENTO
del grid (D=0.080), con eps=1e-2, N=200, 12 semillas, igual método que
`medir_pasos_lavado` de cs074_rcruz generalizado a D paramétrico. `pasos_fijo` =
ceil(mediana × 1.15) — mismo margen que cs074_rcruz. Ese mismo `pasos_fijo` se usa para
las 64 combinaciones (H,D) del grid, para que todas las celdas tengan la MISMA
exposición temporal y la comparación entre celdas no esté confundida por tiempos de
corrida distintos.

Prueba de escritorio previa (banco de calibración, mismo código, sin fijar el número a
mano): D=0.08 → lavado ≈33 200 pasos (semilla única de prueba) → pasos_fijo esperado
≈38 000–40 000 tras medir con 12 semillas y margen. El valor FINAL que se usa es el que
imprima el propio script al correr (medido, no este estimado de escritorio).

---

## 4. Observable y NULL (idénticos a cs074_rcruz, sin tocar)

- `persistencia(phi, c0) = corr(phi, roll(phi,1))+ × (var(phi)/c0²)` — autocorrelación a
  primer vecino (clipeada a ≥0) por varianza relativa. Igual fórmula, sin cambios.
- NULL: permutación de `phi` al final de la corrida (mismo campo, mismo camino de
  cortes, se destruye la forma espacial preservando el histograma). Igual a
  cs074_rcruz.
- Por cada (H,D): `P_real` y `P_null` promediados sobre 12 semillas, `z = (P_real̄ −
  P_null̄) / sd_combinada` (misma fórmula que cs074_rcruz).

---

## 5. Criterios de PASS/FAIL (fijados AHORA, antes de correr)

### 5.1 Verificación previa (gate de validez, no la pregunta central)
El NULL debe morder: se reporta z por celda; si en ≥90% de las 64 celdas |z| es
consistente con REAL≻NULL en el régimen r≳1 (mismo patrón cualitativo que F2-1/CS074),
el grid es válido para interpretar. Si el NULL no muerde en general, se reporta como
hallazgo y el resto se interpreta con esa reserva explícita (T4).

### 5.2 Prueba directa por diagonales (r exacto, la prueba central del experimento)
Para cada diagonal k (grupo de puntos con el mismo r EXACTO por construcción del grid):
- `spread_k` = desviación estándar de `P_real` entre los distintos (H,D) de esa
  diagonal (solo diagonales con ≥2 puntos, es decir k=−6…6, 13 de las 15 diagonales).
- `sigma_semilla` = SEM pooled de P_real por celda = media, sobre las 64 celdas, de
  `P_real_std / sqrt(12)`.
- **Umbral de colapso (fijado):** `tol = max(3 × sigma_semilla, 0.03)`.
- **PASS (colapsa) por diagonal:** `spread_k ≤ tol`.
- **Veredicto global:** se reporta la fracción de diagonales que pasan, y CADA diagonal
  que falla se reporta con su magnitud (`spread_k`, cuántas veces excede `tol`, qué
  (H,D) específicos difieren más).
- Caso especial k=0 (r=1, 8 puntos, la prueba más fuerte): se reporta aparte siempre,
  pase o falle.

### 5.3 Prueba global bilineal (complementaria, cuantifica cuánto pesa cada eje)
Regresión lineal ponderada (peso = 1/SEM²) de `P_real` sobre `log10(H)` y `log10(D)`
más constante: `P ≈ a·log10(H) + b·log10(D) + c`.
- Si el colapso en r es exacto, `log10(r) = log10(H) − log10(D)` implica `a = −b`
  (P depende solo de la combinación log H − log D).
- **Métrica de colapso:** `desbalance = a + b` (0 si colapso perfecto; en unidades de
  "pendiente de P por década").
- **PASS (fijado):** `|a + b| ≤ 3 × SE(a+b)` (no distinguible de 0 con los datos).
- Se reporta `a`, `b`, `a+b`, `SE(a+b)`, y el valor tal cual salga — sin suavizar si no
  pasa.

### 5.4 Regla T3
Estos umbrales (`tol`, factor 3×SEM, `0.03`, `|a+b|≤3×SE`) NO se tocan después de ver
`F2_4_resultado.json`. Si el resultado no pasa, se reporta como FAIL con su magnitud —
no se relaja el umbral ni se cambia el observable.

---

## 6. Verificación cruzada (tres vías, obligatorio)

(a) **NULL:** z por celda, ver §5.1.
(b) **Segundo observable/método:** la propia prueba de colapso (§5.2 directa por
    diagonal EXACTA) actúa como el segundo método frente a la regresión bilineal
    (§5.3) — son dos formas matemáticamente distintas de preguntar lo mismo
    (comparación puntual vs. ajuste global); deben coincidir en veredicto. Si
    divergen, se reporta la divergencia explícitamente (no se elige la que convenga).
(c) **Auditoría en disco:** código (`F2_4_motor.py`) + JSON crudo
    (`F2_4_resultado.json`) con las 64 filas, `P_real`, `P_null`, `P_real_std`,
    `P_null_std`, `z`, `D_emp`, quedan en
    `BATERIA_FUNDAMENTOS/F2_4_H_D_desacoplados/` para que alguien que no escribió el
    código pueda revisar.

---

## 7. Qué NO se hace en F2-4 (fuera de alcance, para no pisar a otros agentes)

No se toca `cs074_rcruz.py`. No se resuelve r* fino (F2-1). No se mide D multi-paso
(F2-2). No se prueba reconexión de aristas (F2-3). No se barre perfil H(t) (F2-5). No
se usa NULL alternativo de secuencia de cortes (F2-6). No hay topología. No hay commits.
Todo el output vive bajo `F2_4_` en esta carpeta.
