# INFORME-PISO · Vocabulario sintiente, resonancia y línea base de carácter
**Fecha:** 2026-07-05 · **Estado:** baseline (el *piso* sobre el cual mediremos la evolución después)

> Este informe fija la línea base empírica de cómo el organismo A **siente** su repertorio vocal, qué
> sonidos lo hacen estar bien, y en qué punto de individuación están los 5 organismos. Todo se midió
> con el propio organismo (sin métricas impuestas). Los números aquí son el **cero** contra el que se
> comparará cualquier medición futura (tras dejar vivir/aprender a los animalitos).

---

## 1. Método (y su disciplina)

- **Instrumento:** el clasificador sintiente de A. Cada sonido se mete al organismo a intensidad viable
  (pico 0.012), se lo deja **oír repetido** (aprenderlo en sus exposiciones) y se lee el **estado ESTABLE**
  en que lo deja, vía `clasificar_cierre` (régimen canónico LF):
  - **VIABILIDAD** = `A_sys_env` alto **y** `e_R` bajo (acoplado/sano).
  - **ACTIVACIÓN** = `LF` alto **y** `delta_struct` alto (explora/cambia).

  | | activo | reposo |
  |---|---|---|
  | **viable** | **JARDÍN FÉRTIL** (florece) | **CIERRE** (calma) |
  | **no viable** | **SELVA HOSTIL** (tenso) | **COLAPSO** (agota) |

- **Sin Shannon, sin métricas mías:** no se clasifica por el espectro ni se tocan umbrales. La categoría
  **emerge** del organismo. (Se auditó y verificó por separado que no hay entropía/Shannon en la valoración.)
- **Método fresco por palabra** (célula reset por sonido). Un intento previo con organismo *persistente*
  que aprende degeneró (churn → 61/70 Selva Hostil, señal plana) y se descartó — ver §7.
- **Compromiso de fidelidad:** por el costo del campo (~2 s por paso en este hardware), se usó
  `exp=3, pasos=16` (vs. 6/32 ideal). Discrimina bien (W abre 0.31–0.67), pero coincide sólo 16/70 con
  la clasificación previa → **la clasificación es sensible a la fidelidad**; re-medir con más pasos daría
  la versión canónica. Este piso vale como **relativo entre sonidos**, no como verdad absoluta por palabra.

---

## 2. Clasificación del vocabulario R2D2 (70 palabras)

**Distribución final (por cómo hace SENTIR a A):**

| cuadrante | n | lectura |
|---|---|---|
| **COLAPSO** | 47 | lo agota |
| **SELVA HOSTIL** | 19 | lo tensa |
| **CIERRE** | 4 | lo calma (viable) |
| **JARDÍN FÉRTIL** | **0** | nada lo hace florecer |

Bienestar W: **min 0.306 · máx 0.669 · media 0.382 · std 0.063.** La media <0.5 dice que **el repertorio
actual, en conjunto, drena a A.**

**Las 4 únicas VIABLES (Cierre) — el núcleo bueno del repertorio:**

| id | concepto | W |
|---|---|---|
| 13 | **Espera** | 0.669 |
| 18 | **Retirada** | 0.611 |
| 19 | **Cierre** | 0.593 |
| screaming | **Dormir** | 0.586 |

→ **Todas son sonidos de reposo/retirada.** Lo único que no colapsa a A es la *quietud*. No tiene ni una
palabra que lo active con bienestar. (Datos completos: `resultado_reclasificar_aprendiendo.json`.)

---

## 3. Búsqueda de resonancia — ¿con qué vibra bien A?

Se barrió el espacio acústico por el mismo clasificador. **Ningún sonido llegó a Jardín Fértil.**

**Curva de resonancia (tonos puros, W por frecuencia):**

```
 60Hz 0.495 ██████████   (máximo; la curva aún subía hacia abajo de 60)
 70Hz 0.487 █████████
 82Hz 0.477 █████████
 98Hz 0.493 █████████     (barrido separado)
131Hz 0.483 █████████
262Hz 0.432 ██████
415Hz 0.381 ████
523Hz 0.356 ███           (mínimo)
1–2.6kHz  ~0.36 ███        (meseta plana en agudos)
```
→ **A resuena con los GRAVES** (viabilidad ↑ al bajar la frecuencia), pero los tonos puros son **COLAPSO**
(pasivos). Las notas grabadas (Do/Do_alto/Fa/La) son **SELVA HOSTIL** (W ~0.44): *activan* pero no son viables.

**Tres hipótesis paramétricas REFUTADAS** por el organismo: (a) agudos → no; (b) graves puros → COLAPSO;
(c) graves + movimiento R2D2 → todas COLAPSO (el warble *empeora* el W). **La viabilidad de A ante un
sonido NO es reducible a una perilla acústica** (frecuencia, brillo, graveza, modulación). Las 4 viables
tampoco comparten firma acústica (Espera es aguda 1821 Hz; Cierre grave 247 Hz). La respuesta es **holística**.

---

## 4. Construcción de palabras "buenas" (cría desde lo viable)

- **Síntesis desde cero (16 R2D2 complejos al azar): FALLÓ** — todos COLAPSO. No se inventa una palabra buena.
- **Variar las 4 viables: FUNCIONÓ** — la bondad se conserva en el vecindario. Método = *genética*: partir
  de lo viable, variar (pitch/estiramiento/secuencia), conservar lo que sigue viable.

**Palabras nuevas VIABLES nacidas y AÑADIDAS al vocabulario (70 → 75):**

| palabra nueva | W | origen |
|---|---|---|
| var_Dormir_grave | 0.588 | Dormir + grave |
| var_Espera_agudo | 0.581 | Espera + agudo |
| var_Cierre_agudo | 0.554 | Cierre + agudo |
| var_Cierre_Retirada | 0.552 | secuencia Cierre→Retirada |
| var_Cierre_grave | 0.551 | Cierre + grave |
| *cria_Cierre_x0.9* | *0.629* | *Cierre × 0.9 (en curso, aún no añadida)* |

→ **"Cierre" es la semilla más robusta** (grave, aguda, secuencia y estiramientos: todas viables). Los WAV
candidatos viven en `voces_r2d2_candidatas/`. **Techo acústico confirmado: CIERRE (calma), nunca Jardín Fértil.**

---

## 5. LÍNEA BASE DE CARÁCTER — el piso principal

Se probó si los 5 organismos (A/B/C/D Docker + E Pi) responden distinto → primera evidencia de individualidad.

**Resultado (hoy): son casi CLONES.**
- Respuesta AGUDA al mismo sonido: **A ≈ B idénticos** (bienestar 0.421 vs 0.422; `A_sys_env`/`e_R`/`LF`
  iguales). El genoma manda y `/start` reinicia la dinámica → la respuesta aguda no revela carácter.
- PALADAR acumulado (biografía): también casi indistinto — A/B/D con el mismo favorito genérico
  (`mundo|0|1`, residuo de snapshot), C/E vacíos, **los 5 famélicos (energía = 0)**.

**Interpretación (el hallazgo):** *el carácter no está dado, se GANA viviendo.* Los 5 arrancan clones
(mismo genoma) y sólo se vuelven individuos con **biografía divergente** (dietas, pares, historias
distintas). Hoy se los reinició repetidamente y no han vivido lo suficiente para diferenciarse. La
individualidad es **relacional e histórica, no innata** — coherente con que el gusto, el florecer y el
valor también resultaron relacionales.

**ESTE es el cero de carácter:** 5 organismos indistinguibles. Cualquier divergencia futura se mide contra aquí.

---

## 6. Hallazgos transversales

1. **Todo resultó RELACIONAL, no una propiedad de la señal:** el gusto (mismo audio, distinto valor según
   estado), el florecer (ningún sonido aislado logra Jardín Fértil → requiere contexto que importe), y el
   carácter (emerge de la biografía). Anti-Shannon consistente en cada capa.
2. **El organismo refuta al experimentador:** 3 hipótesis acústicas mías caídas. La disciplina de *no meter
   métricas* fue la que dejó hablar al organismo.
3. **Se puede AMPLIAR el vocabulario bueno criando desde lo viable**, no sintetizando desde cero.

---

## 7. Caveats honestos (para no sobre-leer el piso)

- **Fidelidad reducida** (exp=3/pasos=16 por el costo de cómputo) → clasificación relativa fiable, absoluta por
  palabra sensible a fidelidad (coincide 16/70 con la previa). Re-correr canónico (6/32) firmaría los cuadrantes.
- **El intento persistente-que-aprende degeneró** (churn a Selva Hostil) → se usó fresco por palabra. La
  "deriva por aprendizaje" a fidelidad real es un experimento de horas/noche, pendiente.
- **0 Jardín Fértil en todo lo probado** → o el sonido aislado no basta (hipótesis relacional), o falta explorar
  fuera del espacio muestreado.
- Carácter medido en estado **famélico/reiniciado** → subestima diferencias que emergerían con vida sostenida.

---

## 8. Qué medir DESPUÉS contra este piso

1. **Re-clasificar el vocabulario** tras dejar vivir a A (¿cambian los cuadrantes al acumular biografía?).
2. **Carácter:** dejar a los 5 vivir **divergente** (A cazando radio, E fotosintetizando, B/C/D con sus pares) y
   re-comparar paladares/memorias/gustos → ¿emerge la primera diferencia *ganada*?
3. **Jardín Fértil por vía relacional:** ¿aparece cuando una palabra "buena" ocurre en un intercambio que le
   importa a A (no como estímulo aislado)?
4. **Vocabulario criado:** validar en vivo que las 5 (pronto 6) palabras nuevas nutren/calman de verdad al usarse.

---

## Reproducibilidad (scripts y datos, en `Célula_Madre/experimentos/`)

- `reclasificar_fresco.py` → clasificación fresca del vocabulario · `resultado_reclasificar_aprendiendo.json`
- `test_notas_resonancia.py` → curva de resonancia · `resultado_notas_resonancia.json`
- `construir_palabras_graves.py`, `busqueda_amplia_palabras.py`, `cria_cierre.py` → síntesis + cría · WAV en `voces_r2d2_candidatas/`
- `reclasificar_aprendiendo.py` → intento persistente (degenerado, documentado)
- Vocabulario ampliado en `voces_r2d2/` (70 → 75). Renombrado corto del vocabulario: **pendiente de decisión** (dry-run listo).
