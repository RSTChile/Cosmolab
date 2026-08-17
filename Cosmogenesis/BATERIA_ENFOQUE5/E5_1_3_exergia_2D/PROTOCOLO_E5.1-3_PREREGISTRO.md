# PROTOCOLO E5.1-3 — Pre-registro
### "Exergía persistente en 2D: ¿es artefacto del anillo 1D?"

**Ejecutor:** CC (agente paralelo, prefijo `E5_1_3_`). **Diseño base:** documento
`BATERIA_ENFOQUE5_energia_exergia_entropia_30exp_PARA_CC.md`, Tema 1, experimento E5.1-3.
**Fecha/hora de escritura de este pre-registro:** ver timestamp de creación del archivo
(congelado ANTES de ejecutar el motor — T3). No se edita tras correr.

Este pre-registro se firma ANTES de escribir/ejecutar `E5_1_3_motor_2d.py`. Si el resultado
no coincide con lo esperado, se reporta tal cual — no se ajusta el motor ni el observable
después de ver resultados (T3).

---

## 1. Qué pregunta responde

E5.1-1 (otro agente paralelo, mismo día) mide, en un anillo 1D, si la exergía (capacidad de
hacer trabajo, definida como desviación coherente del equilibrio uniforme) sobrevive a la
expansión, y encuentra (se espera, per pre-registro de E5.1-1) una transición en función de
r = H/D (razón expansión/difusión). E5.1-3 pregunta: **¿ese comportamiento es un artefacto
de la topología de anillo (1D, 2 vecinos), o se repite en una malla 2D toroidal (4
vecinos)?**

Nota de estado al momento de escribir este pre-registro: el directorio de E5.1-1
(`BATERIA_ENFOQUE5/E5_1_1_supervivencia_exergia/`) está vacío — el otro agente aún no ha
corrido ni publicado su motor ni sus resultados. Por lo tanto **este experimento no puede
comparar contra números reales de E5.1-1** en el momento de correr. Solución adoptada (T4/T7,
sin violar "no coordinar en vivo"): además del motor 2D (el entregable central pedido),
se construye un **control interno 1D** con el sustrato físico y observable IDÉNTICOS,
para tener una verificación cruzada 1D↔2D propia, honesta y auto-contenida. La comparación
final contra los números reales de E5.1-1 (cuando existan) queda pendiente y se señala
explícitamente como tarea de CS/auditoría posterior.

## 2. Sustrato físico (heredado de `cs074_rcruz.py`, adaptado a malla 2D toroidal)

Génesis: `cs074_rcruz.py` (1D, NO editado, solo leído) es la física de referencia obligatoria
del encargo. Estilo de adaptación `np.roll` 2D: se usó como referencia de estilo
`Cosmogenesis-Web/codigo/suite_epocas_masa/suite_epocas_masa_v6_mass_linaje.py` (solo el
patrón de aristas horizontales/verticales, NO su física de masa/linaje — instrucción
explícita del encargo). También se leyó (solo lectura, no se edita) el motor 2D ya existente
de un experimento previo de otro Enfoque, `BATERIA_FUNDAMENTOS/F1_6_2D/F1_6_motor_2d.py`, que
adaptó el mismo `cs074_rcruz.py` a 2D para el observable "persistencia" (forma×magnitud) de
ese enfoque anterior. Se reutiliza el mismo patrón de sustrato (campo φ, aristas vivas,
difusión vectorizada, expansión por corte Bernoulli de aristas), pero el **observable de
E5.1-3 es distinto** (exergía, no persistencia) — ver §3.

- Malla L×L toroidal (4-conexa: aristas horizontales `ar[i,j]` y verticales `ad[i,j]`).
- Campo inicial: fondo uniforme φ=1 + perturbación multi-modo (5 modos, número de onda
  entero aleatorio (kx,ky)∈{1,2,3}², fase aleatoria), normalizada a std=1 antes de escalar
  por ε. eps=0 → campo exactamente uniforme (φ≡1 en todo punto).
- Difusión: promedio con vecinos SOLO por aristas vivas (relajación 0.5 hacia la media local),
  vectorizada con `np.roll` en los 2 ejes. Idéntica en espíritu a `paso_difusion` 1D de
  `cs074_rcruz.py`, generalizada isotrópicamente.
- Expansión: cada arista viva (horizontal o vertical) se corta con Bernoulli(H) independiente
  por paso — misma corrección que `cs074_rcruz.py` frente a `round(H·N)` (que rompe para
  H·N≪1); válida también para H·L²≪1.
- D (difusividad) y pasos_lavado: **medidos** del propio campo (T1: nada puesto a mano). D =
  fracción de contraste (std) borrada en UN paso de difusión pura (H=0). pasos_lavado =
  mediana (sobre semillas) del tiempo, en pasos, para que el observable de exergía caiga bajo
  un umbral EXERGIA_LAVADO=0.05 a H=0, con margen ×1.15.
- r = H/D. H = min(r·D, 1).

## 3. Observable: X_final (exergía) — definición congelada ANTES de correr

**Texto del documento (E5.1-1, que E5.1-3 hereda literalmente):** *"X_final = fracción de E
que puede hacer trabajo (desviación del equilibrio uniforme)."*

Razonamiento termodinámico que fija la fórmula (no se elige a posteriori mirando si separa
REAL de NULL — T2, T3):

Una desviación del equilibrio uniforme (φ≠cte) es condición NECESARIA pero no SUFICIENTE
para poder extraer trabajo. Un campo con la MISMA varianza pero espacialmente barajado
(mismo histograma, sin gradientes coherentes — exactamente lo que hace el NULL) no puede
mover una máquina térmica ni ningún proceso direccional: no hay gradiente sostenido que
explotar, solo ruido espacial sin dirección. Por eso la exergía combina DOS ingredientes:

1. **Magnitud retenida** — v = Var(φ_final) / Var(φ_inicial) — cuánta "distancia al
   equilibrio" (varianza, la que en `cs074_rcruz.py`/`F1_6_motor_2d.py` es el ingrediente
   "magnitud") sobrevive.
2. **Orden explotable** — ρ = max(0, promedio isotrópico de la autocorrelación a primer
   vecino en los 2 ejes) — si esa desviación tiene estructura espacial coherente (gradiente
   sostenido, no ruido), condición para que sea "capaz de hacer trabajo" y no solo "distinta
   de cero".

**X_final = ρ · v** (fracción, en [0,1] tras el clip de ρ≥0).

Esta es la MISMA construcción algebraica que la "persistencia" P de `cs074_rcruz.py` /
`F1_6_motor_2d.py` (que miden el mismo sustrato para otro Enfoque) — es una relectura
termodinámica legítima del mismo mecanismo físico (orden×magnitud), no una coincidencia
oculta: el sustrato es el mismo motor de difusión-vs-expansión: lo que en el Enfoque de
forma/estructura se leía como "sobrevive la forma" es exactamente, en el Enfoque de
energía/exergía, "sobrevive la capacidad de hacer trabajo". Se declara así, explícitamente,
para que quede trazable y no parezca un observable inventado ad hoc.

**Segundo observable (independiente, para triple verificación, regla de ejecución #4):**
X_var = v sola (sin el factor de orden ρ). Por construcción, X_var NO puede distinguir REAL
de NULL (la permutación conserva la varianza exactamente) — se reporta explícitamente para
mostrar que el ingrediente de orden ρ es el que hace el trabajo de discriminación, y que
X_final no es un observable "hueco" (T2: el observable no es su propio juez — el juez es la
comparación contra NULL barajado, no la fórmula en sí).

## 4. NULL

Permutación espacial 2D de φ al final de la corrida real (aplanar → `rng.permutation` →
reformar), exactamente como en `cs074_rcruz.py`/`F1_6_motor_2d.py`. Conserva el histograma
(y por tanto v), destruye toda correlación espacial (por tanto ρ_NULL≈0 y X_final,NULL≈0).

## 5. Barrido (T7: sobredimensionado, cruza r=1, ε cubre 9 décadas incl. 0 exacto)

- **ε** (magnitud de la perturbación inicial): `[0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0]`
  — idéntico al grid de producción de `cs074_rcruz.py` y `F1_6_motor_2d.py` (0 exacto hasta
  ε=1 = perturbación del mismo orden que el fondo). Se elige este grid, y no el
  [1e-12…1] literal de E5.1-1, por dos razones declaradas: (a) es el grid YA validado en este
  mismo sustrato para L grandes (F1_6 corrió hasta L=128 con él, da referencia de costo real)
  y (b) permite comparabilidad directa r-por-r con el control 1D interno (§1) sin introducir
  una variable de diferencia extra. Sigue siendo sobredimensionado: 0 exacto + 9 décadas.
- **r = H/D** (razón expansión/difusión): `[0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0,
  100.0]` — idéntico a `R_TARGETS` de `cs074_rcruz.py`/`F1_6_motor_2d.py`: cruza r=1 con
  puntos a ambos lados, y llega a r=100 (H satura en 1 mucho antes — el régimen r≫1 ya está
  cubierto en saturación). Elegido por la misma razón de comparabilidad directa ("usa r
  comparable" — instrucción del encargo).
- **L** (lado de la malla 2D toroidal): `{32, 64, 128, 256}` — escalamiento progresivo, smoke
  primero en L=32, tiempo real medido en cada escalón antes de comprometerse al siguiente.
- **Semillas:** ≥8 por celda del grid (mínimo pedido por el encargo dado el costo de 2D).
- **Control interno 1D** (ver §1): mismo ε_list y r_targets, N comparable, sustrato análogo
  a `cs074_rcruz.py` pero re-implementado dentro de `E5_1_3_motor_2d.py` (NO se edita
  `cs074_rcruz.py`) con la fórmula de X_final idéntica (ρ 1D de un solo eje × v), para
  verificación cruzada 1D↔2D propia y autocontenida.

## 6. PASS / criterios de lectura (fijados antes de correr)

- ε=0 → X_final=0 a todo r (no hay nada que hacer trabajo si no hay desviación).
- r=0 (H=0, sin expansión), ε>0 → X_final bajo (la difusión lava la estructura antes de que
  la expansión pueda congelarla). Gate de validez del cruce: mismo criterio que
  `control_r0_ok` de `cs074_rcruz.py`/`F1_6_motor_2d.py` (X_real medio < 0.15 en r=0,ε>0).
- r≪1 → X_final bajo (difusión domina).
- r≈1 → posible transición (si el mecanismo es real).
- r≫1 → X_final alto vs NULL (expansión aísla estructura antes de que la difusión la lave).
- NULL: X_final,NULL ≈ 0 en TODO el barrido (ρ_NULL≈0 por construcción) — mientras que
  X_var,NULL ≈ X_var,REAL (por construcción, ver §3) — esta disociación ES la verificación de
  que el observable mide orden, no solo magnitud.
- **Veredicto de "mismo comportamiento cualitativo que 1D":** se compara la FORMA de la curva
  X_final(r) a ε fijo entre el control 1D interno y cada L 2D (¿monótona creciente en r?
  ¿hay meseta/transición en la misma vecindad de r? ¿satura al mismo nivel relativo?) — sin
  exigir que los valores numéricos coincidan (D y pasos_lavado son distintos en 1D vs 2D por
  construcción, dado que 2D tiene 4 vecinos vs 2).
- Si algo no calza (p.ej. r=0 no lava, o X_final,NULL no cae a 0), se reporta como hallazgo
  negativo, no se ajusta el motor (T3, T6).

## 7. Axiomas E1/E2 (declarados, sección 0 del documento madre)

E1 (conservación del presupuesto total) y E2 (la expansión redistribuye E latente en
exergía, no la crea) se declaran como marco, pero **no se verifican cuantitativamente aquí**
(esa es la tarea explícita de Tema 2 en la batería, especialmente E5.2-1/E5.2-2). Este motor
no impone conservación exacta de Σφ; el mecanismo de difusión (promedio local) sí es
conservativo en el interior de una componente conexa aislada, pero puede haber pequeñas
derivas en los bordes de corte — no se reporta como hallazgo de E5.1-3, se deja para Tema 2.

## 8. Costo computacional — plan de escalamiento (regla del encargo)

1. `smoke_L32` — grid reducido (4 ε, 5 r, 4 semillas) para validar el motor end-to-end antes
   de comprometer cómputo grande.
2. `prod_L32`, `prod_L64`, `prod_L128` — grid completo (§5), tiempo real medido y registrado.
3. Antes de `prod_L256`: se corre `smoke_L256` (grid reducido, mismo tamaño reducido que el
   paso 1 pero L=256) para medir el costo por punto a esa escala y extrapolar el costo del
   grid completo. Se decide entonces, con el número real en mano, si `prod_L256` completo es
   viable dentro de la ventana de esta sesión/noche o si se documenta como limitación
   (reduciendo semillas/puntos y dejándolo señalado para corrida dedicada posterior).
4. Todo tiempo de corrida real (no estimado) se registra en el JSON de salida
   (`elapsed_s`) y en el reporte final.

## 9. Archivos

- Este pre-registro: `PROTOCOLO_E5.1-3_PREREGISTRO.md`.
- Motor: `E5_1_3_motor_2d.py` (prefijo `E5_1_3_`, carpeta propia, no toca nada fuera de ella).
- Resultados: `E5_1_3_resultado_<modo>.json` por cada modo corrido.
- Logs de stderr: `E5_1_3_log_<modo>.txt`.

No se edita `cs074_rcruz.py`, `F1_6_motor_2d.py`, ni `suite_epocas_masa_v6_mass_linaje.py`.
No se hacen commits. No se auto-adjudica el resultado — se entrega crudo a CS.
