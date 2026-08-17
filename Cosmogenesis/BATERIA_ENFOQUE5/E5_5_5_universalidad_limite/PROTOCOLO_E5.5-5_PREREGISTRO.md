# PROTOCOLO E5.5-5 — Universalidad del límite: ¿todos los ε→0 llegan al mismo estado?

**Congelado (pre-registro):** 2026-07-24 20:45 (America/Santiago, UTC-4)
**Ejecutor:** CC (agente E5.5-5, batería Enfoque 5, corrida en paralelo con 29 agentes más — prefijo propio `E5_5_5_`, no se toca nada fuera de este prefijo)
**Base de código leída (NO editada):** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_rcruz.py`
**Documento madre:** `BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, sección "E5.5-5" (Tema 5)
**Referencia de familias (leída, NO editada):** `BATERIA_FUNDAMENTOS/F1_4_familias_forma/F1_4_motor.py` y su
`PROTOCOLO_F1-4_PREREGISTRO.md` — el experimento F1-4 (Enfoque de Fundamentos) ya definió y congeló 6
familias de forma inicial para el mismo código base. Se reusan aquí IDÉNTICAS (mismas fórmulas, mismos
parámetros libres por semilla) para comparabilidad directa entre baterías — no se inventan familias nuevas
que puedan sesgar el resultado (T0/T1). Al momento de escribir este documento, F1-4 no tiene
`F1_4_produccion_resultado.json` terminado en disco (solo smoke + logs de una corrida de producción sin
resultado final escrito) — no hay resultado previo que leer como referencia numérica, solo las definiciones
de familia, que sí se reusan.

Este documento se escribe y congela ANTES de tocar el motor. Cualquier desviación respecto de lo aquí
escrito se reporta como desviación explícita, no se edita retroactivamente (T3).

---

## 1. Pregunta

Distintas formas iniciales de diferencia (6+ familias de perturbación), dejadas evolucionar hasta que la
diferencia se disuelve (el límite "ε→0" entendido como el estado de muerte térmica alcanzado por la
dinámica, no solo como el parámetro de amplitud inicial ε llevado a 0): ¿mueren todas en el mismo estado de
equilibrio (mismo E, X, S_ent), o hay dependencia de la forma inicial (dispersión entre familias)?

Dos lecturas del "límite ε→0" se cubren en el mismo barrido, sin privilegiar una a mano:
(a) el parámetro de amplitud inicial ε barrido en varias décadas hacia 0 (igual espíritu que E5.1-1/E5.5-1),
(b) la dinámica corrida hasta un número de pasos calibrado (medido, no puesto a mano) para que el campo
    quede cerca de su equilibrio de difusión — el "estado de muerte térmica" per se, sea cual sea el ε
    de partida.

## 2. Modelo físico (heredado de `cs074_rcruz.py`, IMPORTADO sin modificar)

Se importa el módulo `cs074_rcruz.py` con `importlib` (igual mecanismo que usa F1_4_motor.py) y se reusan,
sin copiar a mano ni modificar:

- `paso_difusion(phi, activo)` — difusión local por aristas vivas.
- `paso_expansion(activo, H, rng)` — corte de aristas vivas por Bernoulli(H).
- `persistencia(phi, contraste0)` — fórmula de exergía `X = c·v` (ver §4).
- `detectar_cuantizacion`, `temperatura_fisica`, `T_SING` — se reportan como metadatos, no son objeto de
  este experimento.

**Régimen de expansión — fijo, NO barrido:** r = H/D = 0 (H=0, sin expansión) en TODA la corrida. Justificación:
el objeto de este experimento es el límite de "muerte térmica" (T5 de la batería), que solo se alcanza si la
difusión puede lavar completamente el campo sin que la expansión aísle y congele estructura (eso es el objeto
de E5.5-2/E5.1-1, no de este experimento — mantener r fuera del barrido evita duplicar esos experimentos y
mantiene E5.5-5 enfocado en su pregunta asignada: familia vs. estado final). Se documenta esto como limitación
de alcance declarada, no oculta.

## 3. Las 6 familias de forma inicial (congeladas — reusadas de F1-4, NO redefinidas a mano)

Normalización común a las 6 (idéntica a F1-4 y a `cs074_rcruz.campo_inicial`): `pert -= pert.mean()`, si
`std(pert)>0`: `pert /= std(pert)`, `φ = 1 + ε·pert`.

| # | Familia | Construcción | Parámetro libre por semilla |
|---|---|---|---|
| 1 | `multi_modo` | Baseline = código base sin tocar. Σ sin(2π·m·x + φ_m)/m, m=1..5 | fases φ_m ~ U(0,2π), 5 por semilla |
| 2 | `modo_unico` | Un solo modo de Fourier sin(2π·m·x + φ) | m ~ entero uniforme {1..8} + φ ~ U(0,2π) |
| 3 | `bulto_gaussiano` | exp(−d(x,x₀)²/(2σ²)), d = distancia circular | x₀ ~ U(0,1), σ ~ U(0.02,0.08) |
| 4 | `ruido_blanco` | Espectro de amplitud plano \|A(k)\|=k⁰, fase aleatoria, IFFT real | fases ~ U(0,2π) por modo |
| 5 | `ruido_rojo` | Espectro \|A(k)\|=k⁻¹ (potencia ∝1/k², domina escala grande) | ídem |
| 6 | `ruido_azul` | Espectro \|A(k)\|=k⁺¹ (potencia ∝k², domina escala chica) | ídem |

Implementación idéntica a `F1_4_motor.py::campo_familia` (reescrita aquí como copia literal de fórmula,
verificada línea a línea contra el original, ya que F1-4 pertenece a otra carpeta de la batería y no se
importa por path relativo entre baterías — pero el código es el mismo, no se altera ni un signo).

## 4. Observables — definiciones (E, X, S_ent), congeladas ANTES de correr

Notación: φ₀ = campo inicial (tras sembrar la familia), φ_f = campo al final de la evolución (pasos
calibrados, ver §6). N = 200 sitios.

### X — Exergía
**Idéntica a `persistencia()` de `cs074_rcruz.py`, importada sin modificar** (para comparabilidad directa con
E5.1-1 y el resto de la batería que ya usa esta fórmula como "X_final"):

    c = corr(φ_f, roll(φ_f,1))    (clip a ≥0 si no finito → 0)
    v = Var(φ_f) / Var(φ₀)
    X = c · v

X=0 si Var(φ₀)=0 (caso ε=0, sin estructura que evolucionar) — por construcción.

### E — Energía total (declarada conservada, axioma E1, AUDITADA no forzada)
    E_frac = mean(φ_f) / mean(φ₀)

Por construcción (perturbación de media cero antes de escalar por ε), mean(φ₀) ≈ 1 siempre. E_frac debería
mantenerse ≈1 en todo el barrido si la difusión conserva el promedio del campo (operador lineal de
promediado local). Se AUDITA: se reporta la deriva |E_frac − 1| por fila; si es grande, es hallazgo, no se
oculta ni se renormaliza el campo a mano (T6).

### S_ent — Entropía (Gibbs, sobre la distribución espacial de magnitud — NO histograma de valores)
Se define tratando φ_f como una distribución de peso sobre los N sitios (argumento tipo teorema-H: para la
ecuación de difusión, el funcional −Σp·log(p) con p_i = φ_i/Σφ_i es un Lyapunov que crece monótonamente
hacia el máximo en el estado uniforme — es la elección estándar, no una entropía de histograma de valores,
que iría en la dirección contraria: un campo uniforme concentraría todo su histograma de valores en un solo
bin y daría entropía de histograma BAJA, lo opuesto de lo que "muerte térmica ⇒ entropía máxima" exige).

    p_i = max(φ_i, φ_piso) / Σ_j max(φ_j, φ_piso),   φ_piso = 1e-9 (piso solo para esta medida, NO
                                                       modifica la dinámica; declarado antes de correr, T1)
    H(p) = −Σ p_i · log(p_i)
    S_ent = H(p) / log(N)          (normalizado: máximo posible = 1, campo uniforme exacto)

Se audita y reporta la fracción de sitios que requirieron el piso (φ_i≤0) por fila — si es grande, se
reporta como advertencia de validez de esta métrica en ese régimen de ε (relevante solo para ε grandes,
fuera del foco "→0" de este experimento).

**Nota de independencia (T2):** S_ent y E_frac son, por construcción, INVARIANTES a permutación espacial de
φ (dependen solo del multiconjunto de valores, no del orden) — por eso el NULL (que permuta φ) no aporta
información sobre E o S_ent; el NULL es relevante y se reporta SOLO para X (que sí depende del orden
espacial vía la autocorrelación c). Esto se declara aquí explícitamente para que no se lea como omisión.

## 5. Barrido (sobredimensionado, regla del director; familia es el eje nuevo de este experimento)

| Eje | Rango | Puntos |
|---|---|---|
| familia | las 6 de §3 | 6 (ninguna se omite) |
| ε (amplitud inicial) | {0} ∪ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1} | 8 (0 estricto + 6 décadas hacia arriba + un punto de contraste moderado 0.3, para verificar que el estado final converge también independientemente de CUÁN grande era la amplitud de partida, no solo de la forma) |
| r = H/D | 0 (fijo, ver §2) | 1 |
| semillas | 0..11 | 12 (≥12 pre-registrado) |
| pasos | calibrado por familia (ver §6, NO puesto a mano) con margen ×5 sobre el lavado medido, para garantizar profundidad de equilibrio, no solo cruzar el umbral | — |

Total combinaciones (familia × ε) = 6 × 8 = 48. Cada combinación: 12 semillas × {REAL, NULL} = 24 corridas
→ **1152 corridas de evolución** + calibración de lavado por familia + verificación de convergencia (§7).

## 6. Calibración de pasos ("hasta ε→0" = hasta profundidad de equilibrio medida, no un número puesto a mano)

Por familia (misma receta que `cs074_rcruz.medir_pasos_lavado` / `F1_4_motor.medir_pasos_lavado`):
1. Se mide, con H=0 y ε=1e-2 (representativo, 8 semillas de calibración), el número de pasos para que
   P(=persistencia) caiga bajo P_thr=0.05 (`check_every=50`, `max_steps=50000`).
2. `pasos_lavado_mediana` = mediana de esos tiempos.
3. **`pasos_muerte` = ceil(pasos_lavado_mediana × 5)** — margen ×5 (NO el ×1.15 del código base) porque
   aquí el objetivo no es solo cruzar el umbral P<0.05 sino estar demostrablemente DENTRO del régimen de
   equilibrio profundo ("hasta ε→0"), congelado antes de correr, no ajustado después de ver resultados.
4. `pasos_muerte` se usa igual para TODOS los ε de esa familia (igual convención que el resto de la
   batería: se calibra una vez por familia, no por cada ε, para no introducir un grado de libertad extra).

## 7. Verificación de convergencia (chequeo de plateau, adicional, declarado antes de correr)

Para UNA combinación representativa por familia (ε=1e-2, semillas 0-2), se corre también a
`2×pasos_muerte` y se compara X, S_ent, E_frac contra el valor a `pasos_muerte`. Si la diferencia relativa
es grande (>10%), se reporta como advertencia de que `pasos_muerte` no alcanza profundidad de equilibrio
para esa familia — no se re-ajusta el barrido principal después (T3), se reporta el hallazgo tal cual.

## 8. NULL

Permutación del campo final (`rng.permutation(phi)`), idéntica receta que el resto de la batería. Aplica
solo a X (ver nota de independencia en §4). Reportado por fila: X_real, X_null, z-score.

## 9. Criterio de PASS / lectura (congelado, no se cambia tras ver datos — T3)

**Por familia, en el ε>0 más pequeño del barrido (1e-6) y en ε=0.3 (extremos del rango de amplitud
barrido):**
- X debe ser bajo (cerca de 0, dado r=0 y pasos_muerte calibrado para lavar) — si no lo es, `pasos_muerte`
  no alcanzó el equilibrio para esa familia y se reporta como FAIL de esa familia, no se fuerza lectura.
- S_ent debe ser alto (cerca de 1).
- E_frac debe ser ≈1 (conservación E1 auditada).

**Veredicto global (tres lecturas pre-registradas, ninguna se descarta a mano):**
1. **Convergencia:** si, en el estado de equilibrio profundo, la dispersión ENTRE familias (std y rango de
   X, S_ent, E_frac a través de las 6 familias, a igual ε) es pequeña frente a la dispersión ENTRE
   SEMILLAS dentro de una misma familia — la forma inicial NO determina el estado final (universalidad del
   límite, PASS de convergencia).
2. **Dispersión real:** si la dispersión entre familias es comparable o mayor que la dispersión entre
   semillas — hay dependencia de la forma inicial; se reporta cuál(es) familia(s) se aparta(n) y en qué
   observable, sin promediar ni ocultar (D1 vivo).
3. **Ninguna familia lava (todas con X alto pese a r=0 y pasos_muerte):** falla de calibración/diseño, se
   reporta como tal, no se reinterpreta como "hallazgo de no-convergencia".

**No se ajusta este criterio después de correr.**

## 10. Verificación cruzada (regla de ejecución #4)

(a) NULL de X, por fila (§8).
(b) Segundo observable/método: comparación INTER-familia misma (si `multi_modo`, que es el código base sin
    tocar, converge al mismo punto que las otras 5, es la verificación cruzada específica de este
    experimento) + chequeo de plateau (§7) como segundo método temporal independiente.
(c) Auditoría en disco: JSON crudo con las 1152 corridas + metadatos de calibración + este protocolo +
    log de ejecución con timestamps, en `resultados/`.

## 11. Trampas explícitamente evitadas

- T0/T1: familias y P_thr/márgenes son valores declarados aquí ANTES de correr, reusados de F1-4 (no
  inventados para favorecer el resultado); `pasos_muerte` sale de una medición del propio campo (×5 sobre
  lavado medido), no puesto a mano.
- T2: X, E, S_ent son fórmulas fijas; el veredicto de convergencia lo da la dispersión entre familias
  comparada contra la dispersión entre semillas, no un número aislado.
- T3: este documento se congela antes de escribir el motor.
- T4: el NULL se reporta para X en cada fila.
- T5: se reporta la curva completa (familia × ε), no un gate binario único.
- T6: se audita conservación E1 (E_frac) en cada fila; se declara que r está fijo en 0 (limitación de
  alcance, no ocultamiento).
- T7: barrido de ε en 7 décadas + 12 semillas por celda (perturbación por semilla vía fases/parámetros
  libres de cada familia, igual que F1-4).

## 12. Archivos de salida

- `E5_5_5_motor.py` — motor (escrito DESPUÉS de este pre-registro).
- `resultados/E5_5_5_produccion_resultado.json` — barrido completo crudo (48 combinaciones × 12 semillas).
- `resultados/E5_5_5_log_ejecucion.txt` — log con timestamps.
- `resultados/E5_5_5_analisis.json` — agregados: dispersión entre familias vs. entre semillas, por ε;
  chequeo de plateau; veredicto de convergencia.

No se corre nada del motor hasta que este archivo esté guardado en disco.

---

## Addendum (post-smoke, PRE-producción — 2026-07-24 20:52, antes de correr el grid congelado)

El smoke test (N=100, 4 semillas, `resultados/E5_5_5_smoke_resultado.json`) reveló un problema de
**calibración** (no de definición de observable ni de criterio, ambos siguen intactos): para `modo_unico`
y familias con un parámetro libre por semilla que controla la tasa de decaimiento físico, la calibración de
`pasos_lavado` (§6, 8 semillas aleatorias) puede sortear por azar solo casos "rápidos" (m alto) y subestimar
`pasos_muerte` para semillas del grid que sí caigan en el caso lento (m=1, el modo más bajo, decae más
lento bajo difusión — propiedad física conocida, no una elección para mover el resultado). Evidencia:
`modo_unico` calibró `lavado_mediana=50` (→ `pasos_muerte=250`) pero el chequeo de plateau (§7) mostró
`dX_medio_abs=0.0395` entre 1x y 2x pasos_muerte — lejos de plano, seguía decayendo.

**Fix declarado (T1: no es ajuste hacia ningún valor esperado, es endurecer la calibración con el peor caso
físico conocido de cada familia paramétrica, ANTES de correr el grid de producción):** para `modo_unico`
(m_fijo=1) y `bulto_gaussiano` (sigma_fijo=0.08, el ancho más lento del rango U(0.02,0.08) declarado en
§3), se añade una calibración de lavado determinista adicional en ese caso peor, y
`mediana_efectiva = max(mediana_aleatoria, mediana_peor_caso)`. `pasos_muerte` se calcula sobre
`mediana_efectiva × MARGEN_MUERTE` como antes. **Las familias en sí (definiciones de §3, usadas en el grid
real) NO cambian** — el sorteo aleatorio de m/sigma por semilla en el grid sigue siendo el pre-registrado;
solo se corrige cuántos pasos se corren para llegar al equilibrio.

Este addendum se escribe ANTES de correr `produccion` (no después de verla) — cumple T3: el criterio de
PASS (§9) y las definiciones de observable (§4) no se tocan, solo la calibración de profundidad temporal.
Se vuelve a correr el smoke test tras el fix para confirmar plateau antes de congelar la corrida completa.

---

## Addendum #2 (post-smoke #2, PRE-producción — 2026-07-24 21:10)

El segundo smoke test (tras el Addendum #1) confirmó `modo_unico` corregido (dX_medio_abs=0.000000), pero
reveló que **`ruido_blanco`** (y por extensión física, probablemente `ruido_azul`) tampoco alcanzaba
plateau: `dX_medio_abs=0.007463` sobre `X_real=0.015391` (~48% relativo) entre `pasos_muerte=250` y
`2×pasos_muerte=500`. Diagnóstico: todo sorteo de `ruido_blanco`/`ruido_azul` contiene el modo k=1 con
amplitud FIJA (α=0 → |A(1)|=1; α=1 → |A(1)|=1, la más baja de la banda pero no cero) — el mismo mecanismo
físico de decaimiento lento que `modo_unico` con m=1, pero a diferencia de éste NO depende de qué semilla
se sortea (solo la FASE es aleatoria, no la amplitud de k=1) — por tanto no es un problema de "calibración
con mala suerte", sino de que el criterio de calibración (cruce de P<0.05, un umbral temprano) simplemente
no mide profundidad de plateau para NINGUNA familia con contenido espectral de baja frecuencia persistente.

**Fix (reemplaza, no parchea, el criterio de calibración — aplicado UNIFORMEMENTE a las 6 familias, sin
excepciones):** se sustituye "cruce de P<0.05 × margen fijo ×5" por una calibración ADAPTATIVA de plateau:
se duplica `pasos` (arrancando en 200) hasta que el cambio absoluto en X y en S_ent entre `P` y `2P` sea
`< TOL_PLATEAU=0.01`, tomando el PEOR CASO (máximo) entre 4 semillas de calibración aleatorias más (para
`modo_unico`/`bulto_gaussiano`) la réplica determinista del peor caso físico ya declarada en el Addendum #1.
`pasos_muerte` final = `2×P` en el punto de convergencia. El criterio de parada (`TOL_PLATEAU=0.01`) es el
MISMO número para las 6 familias — no se ajusta por familia, no es un ajuste hacia ningún valor esperado
(T1). La medición de "cruce P<0.05" se conserva en el JSON de salida como diagnóstico descriptivo
comparable con el resto de la batería, pero YA NO determina `pasos_muerte`.

Se vuelve a correr el smoke test una tercera vez tras este fix para confirmar plateau en las 6 familias
antes de congelar la corrida de producción. Ningún criterio de PASS (§9) ni definición de observable (§4)
se modifica — solo la profundidad temporal a la que se mide el estado final, que es precisamente lo que
"hasta ε→0" pide medir con rigor.

---

## Addendum #3 (POST-producción, hallazgo de verificación cruzada — 2026-07-25)

La corrida de producción (N=200, `resultados/E5_5_5_produccion_resultado.json`, inicio 2026-07-24 21:20:58,
fin 2026-07-24 23:16:22, 6924.2s ≈ 115.4 min) terminó y el grid CONGELADO muestra `ruido_blanco` con
X_real≈0.0150 en las 8 celdas de ε>0 — tres órdenes de magnitud por encima de las otras 5 familias
(X≈2e-6 a 1.5e-5), pese a que su calibración adaptativa (Addendum #2) declaró plateau a `pasos_muerte=400`.

**Verificación de auditoría (regla de ejecución #4, "auditoría en disco por quien no lo escribió" — aquí
autoverificación honesta antes de reportar a CS, no reemplaza la auditoría externa):** se corrió
`ruido_blanco` (eps=1e-2, semillas 1000-1002) a pasos crecientes POR FUERA del grid congelado (no toca
`E5_5_5_produccion_resultado.json`, es solo diagnóstico adicional):

| pasos | X_mean |
|---|---|
| 400 (= pasos_muerte usado en el grid) | 0.01502940 |
| 1600 | 0.00500687 |
| 6400 | 0.00042790 |
| 25600 (= pasos_muerte de las otras 4 familias lentas) | 0.00000003 |
| 102400 | 0.00000000 |

**Conclusión de la verificación:** el `X=0.0150` de `ruido_blanco` en el grid oficial es un **FALSO
PLATEAU** de la calibración adaptativa — la curva de decaimiento de esta familia tiene un "hombro" (el
cambio absoluto entre pasos=200 y pasos=400 cae por debajo de `TOL_PLATEAU=0.01` de forma transitoria,
sin haber llegado al régimen asintótico), y sigue decayendo con fuerza hasta, al menos, pasos=25600, donde
alcanza `X≈3e-8` — CONSISTENTE con las otras 5 familias, no distinto. La calibración adaptativa (Addendum
#2), pensada para corregir el problema de `modo_unico`, tiene entonces un punto ciego no anticipado: un
criterio de parada por doble-paso puede disparar en un hombro transitorio de la curva, no solo en el
plateau real.

**Esto NO se reporta como "arreglado" ni se sustituye el JSON de producción (T3: el grid congelado con
`ruido_blanco` en X=0.015 es el resultado real de ejecutar exactamente el protocolo pre-registrado,
incluyendo su calibración — se reporta tal cual, con esta nota de interpretación adjunta).** Se reporta a
CS íntegro: (a) el resultado crudo del grid oficial (`ruido_blanco` diverge de las otras 5 en la banda
ε>0 tal como quedó congelado), y (b) esta verificación post-hoc que indica que, de haberse corrido
`ruido_blanco` a la misma profundidad temporal que las otras 4 familias lentas, el resultado más probable
es que TAMBIÉN converja al mismo estado casi-nulo. La lectura correcta, sin suavizar, es: **el grid oficial
no puede usarse solo para afirmar "ruido_blanco no converge"** — esa lectura sería un artefacto de
calibración, no un hallazgo físico — y el criterio adaptativo de plateau (Addendum #2) queda documentado
como con un punto ciego conocido, no resuelto en este experimento (correspondería a un experimento de
diseño de calibración, fuera del alcance de E5.5-5).
