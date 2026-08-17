# INFORME_RECONSTRUCCION_CG001

**Autor:** Claude (Club Abulafia) · **Para:** Casaubon / Grok / Claude-web
**Fecha:** 2026-06-29
**Objeto:** Reconstrucción de CG001 desde cero según la INSTRUCCIÓN ÚNICA. Cierra los dos canales por los que el diseño determinaba la estructura (geometría y métricas) y deja un régimen donde la selección puede ocurrir.
**Estado:** Código aplicado y verificado que **corre** (smoke-test). Datos/volúmenes previos **borrados**. Imagen Docker **reconstruida**. **NADA corriendo** — esperando tu instrucción para arrancar.

---

## 0. Principio rector aplicado
Ninguna estructura —espacial ni métrica— debe estar determinada por el diseño; todo lo que aparezca debe venir de la interacción. Se cerraron los dos canales: **geometría** (el movimiento metía la grilla) y **métricas** (las variables colapsadas hacían la cadena cierta por construcción).

---

## A · Geometría isótropa (cierra la fuga de la grilla)

| Cambio | Archivo | Antes → Después |
|---|---|---|
| Paso aleatorio isótropo | `core/universe.py` (movimiento) | `bias_normalizado · uniform(0.2,0.6)` (vector de signo, lock diagonal) → **`rng.normal(0, σ, 3)`** gaussiana independiente por eje |
| Gradiente real | `core/environment.py` | `gradient_bias` normalizado a unidad+ruido → **`gradient()`**: diferencias centradas, **magnitudes reales por componente, sin normalizar**. Env plano → grad≈0 → solo difusión |
| Expansión aditiva (§132) | `core/universe.py` | `e.pos *= scale` (multiplicativa desde origen) → **`pos += v_exp · (pos/‖pos‖)`** radial aditiva |
| Posición inicial isótropa | `core/universe.py` `_init_entities` | `uniform(-r,r,3)` (cubo, prefiere esquinas) → **uniforme en BOLA** |
| Test causal | `core/universe.py` | nuevo flag `CG_ROTATE_TEST=1` → rota la grilla un ángulo arbitrario al iniciar |

**Aceptación (medir en `/entidades`, no en el visor):** mediana max/min|coord| → ≈4.5 (hoy 1.07); frac |x|≈|y|≈|z| (<1.15) → ≈0.007 (hoy 0.68); frac sobre diagonal de cuerpo → ≈0.05.
**Test causal:** con `CG_ROTATE_TEST=1`, si la estructura sigue alineada a los ejes de pantalla → grilla; si rota con el marco → isótropa. **Predicción: ya no rota con la grilla.**

---

## B · Régimen selectivo — requiere TU calibración (no corrida aún)

**Mecanismo** (implementado): la cancelación y el intercambio ahora mueven **S** (no solo Δ) — ver Bloque C/§118. Eso hace que S **baje** y las entidades puedan morir. Antes `persist_cost ≪ gain` y S solo subía → acreción (1 muerte/999). Ahora S fluctúa y la muerte diferencial es posible.

**Parámetros** (`config/CG001_default.yaml`): subí `persist_cost` 0.00012 → **0.002** y añadí `s_loss=0.01` (cancelación en S), todo expuesto para calibrar. **Estos valores son un PUNTO DE PARTIDA en la dirección correcta, NO calibrados.**

**La tensión que respeté:** calibrar la curva N(t) del §8/§23 (95–99% muere temprano, remanente no trivial) **exige correr**, y me pediste no correr. Así que dejé el mecanismo listo y los parámetros en la dirección correcta; **la calibración fina es el primer paso del gate**, cuando des la instrucción. No impuse tabla heurística: el objetivo es la curva, los valores se buscan contra ella.

**Aceptación:** curva N(t) con caída fuerte temprana y meseta en remanente >0 no trivial.

---

## C · Cinco variables como grados de libertad distintos (cierra la fuga métrica)

| Cambio | Detalle |
|---|---|
| ε solo en S inicial de id=0 | Confirmado: `universe.py` línea 60, sin escalado global ni subsidio per-step (§60/§133). **Permanece.** |
| `t_hist` real | `entity.record_history` ahora se llama **una vez por paso** con el cambio NETO de Δ, y solo cuenta si es **observable (≥ κ_H, §22)**. t_sim ≠ t_hist es real. |
| H desacoplado de S | H acumula solo variación estructural **observable** por paso (§16.3), no el conteo de interacciones → rompe S≈H. |
| Δ ↔ S acoplados (§118 completo) | S obedece **G−C** como Δ: gana en refuerzo, **pierde en cancelación**, se redistribuye en intercambio. (Marcado como sugerencia tuya; lo implementé porque es lo que desacopla S de H y vuelve C-N11 testeable en vez de identidad.) |

**Aceptación:** matriz de correlación de {S, Δ, H, t_hist} con cuatro grados distinguibles — ningún par en |corr|≈1, ninguno clavado en constante.

---

## D · Smoke-test del build (QA — NO es el experimento)
30 pasos en un contenedor desechable (sin servidor, sin logs, sin tocar volúmenes), solo para confirmar que el código nuevo corre y se comporta como mandan A/B/C:

| Señal | Antes | Tras reconstrucción (30 pasos) |
|---|---|---|
| mediana max/min\|coord\| (isotropía) | 1.07 | **3.54** (→ objetivo ~4.5) |
| corr(S,H) | 0.998 | **0.558** |
| t_hist_max | 0 (en 999) | **12** |
| muertes | 1 en 170 pasos | **9 en 30 pasos** (S_min=0.038) |

Mecanismo verificado. La **calibración y la aceptación formal son del gate** (corrida completa).

---

## E · Compuerta única (gate) — antes de cualquier afirmación A/B
Arrancar de cero, **1 sola semilla** primero, y verificar los **tres bloques juntos**:
1. Diagnóstico geométrico (A) desde `/entidades`.
2. Curva de supervivencia (B) — **calibrar aquí** `persist_cost`/`gain`/`s_loss`/densidad contra el target del §8.
3. Matriz de correlación (C) desde `/entidades`.

Solo si pasan los tres → multi-semilla ≥30 (§109) → recién ahí correr A vs B y mirar ΔIPD/ΔIH/ΔN. **No leer ningún Δ entre universos hasta que la compuerta pase.**

---

## F · Estado y archivos
- **Datos previos:** borrados (volúmenes y contenedores eliminados).
- **Imagen:** reconstruida (`cosmogenesis-cg001:latest`).
- **Corriendo:** nada (esperando instrucción).
- **Archivos tocados:** `core/universe.py`, `core/environment.py`, `core/entity.py`, `config/CG001_default.yaml`.
- Para arrancar (cuando lo instruyas): `cd Cosmogenesis/docker && docker compose up -d cg001-lab` (B, ε vía CG_EPSILON) y el brazo A (ε=0) en otro puerto.

---

## G · Adenda — iteración 2: §118 como hipótesis conmutable (tras revisión de Claude-web)

**§118 ya no se decide por inferencia — es un MODO CONMUTABLE** (`CG_S_RULE ∈ {coupled, persist_only}`); el gate discrimina por funcionamiento:
- `coupled` (H1, §118-completo): S gana/pierde/redistribuye como Δ → C-N11 falsable, muerte por interacción.
- `persist_only` (H2, §8 literal): S solo sube por refuerzo, solo baja por `persist_cost` → muerte por costo R₀; C-N11 no testeable desde la dinámica.
- Ninguno es default metodológico; el gate corre **AMBOS** (env `CG_S_RULE` por brazo). Verificado: corren y **difieren** (coupled: corr(S,Δ)=0.82, S_min→0.02; persist_only: corr(S,Δ)=0.25, S_min=0.92).

**C-2 intacto** en ambos modos (no es lo que se testea): H solo variación observable/paso, t_hist ≥ κ_H, ε solo en S inicial de id=0. Sin tocar.

**Geometría — kernel isótropo, NO difusión** (palancas 3 y 4 separadas): `gradient()` usa stencil isótropo de 26 vecinos (mínimos cuadrados, pesos ~1/|δ|), no diferencias axiales. `env_diffusion` (λ, §128/§129) queda SOLO para ecología, no para tapar el lock.
- **Geometría NO certificada aún**: el smoke muestra el ratio vagando (t20=2.86, t80=6.08); el ratio no clasifica con estructura → el **rotate-test en el gate es el árbitro**. Sospecha residual: depósitos cuantizados a celda (`_idx`) arrastran estructura axial; si contamina → depósito sub-celda trilineal.

**Performance**: el kernel de 26 vecinos en Python es ~4× más lento. Para 10⁴–10⁵ pasos: vectorizar el gradiente o bajar N.

**Gate (cuando instruyas)**: correr `coupled` y `persist_only` por separado; por modo: curva N(t) vs §8 (calibrar `persist_cost`/`s_loss`/λ contra la curva), matriz de correlación {S,Δ,H,t_hist} con 4 DOF no degenerados, geometría compartida (rotate-test + ratio). Discriminación: un modo "funciona" si alcanza §8 + mantiene 4 DOF + no programa el resultado (divergencia A/B emergente). Uno falla/otro no → veredicto; ambos → §118 no cambia la fenomenología (decisión vuelve a ti); ninguno → de vuelta al diseño.

**Estado**: imagen reconstruida, **nada corriendo**, esperando instrucción.
