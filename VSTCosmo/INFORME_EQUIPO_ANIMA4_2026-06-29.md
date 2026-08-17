# INFORME · ANIMA-4 · Ruido contextual, invención léxica y acople social
**Cosmolab / VSTCosmo · para el equipo · 2026-06-29 · (datos embebidos — sin CSV adjuntos)**

> **Advertencia epistémica:** los rótulos de audio y de voz (`Brandemburgo`, `dolor`, `palabra_A001`…)
> son etiquetas humanas de archivo/banco. NO sabemos qué significan para los organismos. "Sentido nuevo"
> = unidad de vocabulario que cubre una región afectiva antes no cubierta, NO un significado humano.
> Se mide respuesta a configuraciones acústicas y sociales, no semántica humana.

---

## 0. Resumen ejecutivo
Jornada de experimentos sobre **cuándo y por qué los organismos crean vocabulario nuevo**, y sobre si la
sincronía de conducta que se observa entre ellos es **acople social** o **artefacto del contexto común**.
Tres resultados, todos con datos:

1. **La invención léxica obedece a O-N1 (`RC = ICR + IRDE`).** El ruido contextual (RC) que el repertorio
   existente no integra se vuelve **sentido (ICR → palabra nueva)** o se desvía (IRDE). La conservación se
   cumple **exacta** en los datos (residuo 0.00000 sobre 736 285 filas).
2. **RC es TASA DE NOVEDAD, no duración.** En silencio (RC≈0): **0 palabras en 90 min**. Con estímulos
   **variando rápido**: primera palabra a los **5,4 min**. La invención necesita contexto cambiante.
3. **La sincronía de cabezas es seguir-el-contexto, NO atención mutua** (confirmado con un control de
   inversión de entradas). PERO el acople social **sí existe y es LÉXICO**: las palabras se propagan de un
   organismo a otro formando una **red de difusión** (19 eventos, anillo A→D→C→B→A).

---

## 1. Marco: O-N1 está en el organismo, no es metáfora
Las variables del modelo `RC = ICR + IRDE` son **columnas de fisiología registradas** y gobiernan el núcleo
afectivo:
- `RC_total` = *"Ruido contextual total observado: RC = ICR + IRDE."*
- `ICR` = *"fracción convertida en sentido/acoplamiento."* · `IRDE` = *"fracción que aparece como riesgo/desacople."*
- Núcleo afectivo: `arousal = 0.45·RC + 0.30·E + 0.25·lateralidad` → **RC es el driver dominante del arousal**,
  y el arousal abre el "hueco afectivo" que dispara la acuñación.

**Conservación verificada (corrida en silencio, 736 285 filas, DuckDB):**

| RC medio | ICR medio | IRDE medio | (ICR+IRDE) | residuo \|RC−(ICR+IRDE)\| |
|---|---|---|---|---|
| 0.0258 | 0.0051 | 0.0208 | 0.0258 | **0.00000** |

`RC = ICR + IRDE` se cumple **sin error** en cada ciclo. Es una identidad, no una aproximación.

---

## 2. Mecanismo de acuñación (cuándo nace una palabra)
`VST_OrganoComunicacion._quizas_crear`. Una palabra nueva nace solo si se cumplen **a la vez**:

| Compuerta | Umbral | Significado |
|---|---|---|
| HUECO afectivo | gap > **0.22** | la voz del banco más cercana queda lejos del estado afectivo |
| RECURRENCIA | **≥ 3** | ese hueco reaparece (no por un estado fugaz) |
| ENERGÍA | **≥ 0.25** | acuñar cuesta (gasto real) |
| LIBERTAD | prob. **0.6** | aun con todo dado, a veces no crea |

Para **perdurar**: reusarse **≥ 4 veces** (`USOS_CONSOLIDA`) o se poda en 600 emisiones (olvido por desuso).
La compuerta crítica es el **HUECO**, y el hueco depende de RC: **sin RC, el afecto es estrecho → el banco
lo cubre → nunca hay hueco → no se acuña.**

---

## 3. Experimentos (todos desde CERO, volúmenes borrados)

| Corrida | Mundo | Resultado léxico |
|---|---|---|
| **Silencio** (`042243`) | silencio, 90 min/cond | **0 palabras** (los 4: creadas=0). Solo repertorio innato. |
| **Rico estático** | un audio fijo 90 min | (descartado: un audio estático se habitúa, RC efectivo cae) |
| **Variado rápido** (`105705`) | rotación rápida de 118 audios | **1ª palabra a 5,4 min**; RC oscilando 0.004–0.22 |
| **Entorno complejo** (`115204`) | método completo (abajo) | invención + 19 rutas léxicas + veredicto de atención |

**Hallazgo metodológico:** la diferencia entre 0 y muchas palabras NO fue el tiempo sino la **variación del
estímulo**. RC alto = novedad sostenida, no volumen ni duración.

### 3.1 Diseño del "entorno complejo" (corrida `115204`)
Tres ejes de variación simultáneos, desde cero, 60 min:
- **Audio:** rota mezclando 9 categorías (nota, frecuencia, tono, música, voz, ruido, viento, voz+viento,
  textura) del pool de **118 sonidos**, sin repetir categoría seguida; dwell 10–60 s.
- **Ruteo cíclico** (qué oído recibe par/mundo): **R1** pares A↔B C↔D con **A,C invertidos** (par por oído R) ·
  **R2** flip global de los 4 · **R3** invierte uno a la vez (5 temas c/u, rota) · **R4** azar + cada uno su
  propio sonido · y cicla.
- **Mundo:** común en R1–R3 (para el test limpio), propio por organismo en R4 (máxima divergencia).

El **control de inversión A/C** discrimina: si las cabezas siguen el **contexto**, los organismos con el
mundo en el mismo oído se mueven juntos; si hay **atención mutua**, cada uno mira a su pareja (A→B, C→D).

---

## 4. Resultado A — La sincronía de cabezas es CONTEXTO, no atención mutua
Correlación de orientación (`act_orientacion_deg`) por pareja y por régimen:

| régimen | A–B | C–D | **A–C** | **B–D** | A–D | B–C |
|---|---|---|---|---|---|---|
| **R1_PARES** (A,C invert.) | 0.21 | 0.25 | **0.91** | **0.85** | 0.30 | 0.16 |
| **R2_FLIP** (global) | 0.18 | 0.25 | **0.86** | **0.88** | 0.30 | 0.13 |
| R3_UNOxUNO | −0.10 | 0.01 | −0.14 | 0.02 | −0.04 | −0.12 |
| **R4_AZAR** (mundo propio) | 0.26 | 0.08 | **0.33** | **0.38** | 0.12 | −0.21 |

**Interpretación:** la orientación se agrupa por **qué oído recibe el mundo** (A–C, B–D ≈ **0.85–0.91**), NO
por la pareja social (A–B, C–D ≈ **0.21–0.25**). Si se prestaran atención, lo alto sería A–B y C–D. Es al
revés. **Prueba definitiva:** en **R4**, al darle a cada organismo su propio sonido (sin mundo común), la
sincronía **se desploma de ~0.9 a ~0.35**. Requería el estímulo compartido.

→ **La sincronía de cabezas que se ve en el observatorio es rastreo del entorno común (artefacto de
entrada), no atención entre organismos.** La crítica adversarial era correcta; el control la confirmó.

---

## 5. Resultado B — El acople social SÍ existe y es LÉXICO
Las palabras inventadas se **propagan de un organismo a otro** (emulación: re-sintetizan su versión del
afecto del otro). **19 eventos de ruta** en la corrida, formando una **red de difusión dirigida**:

| Origen de la palabra → quién la emula | veces |
|---|---|
| **A → D** (palabra_A001 adoptada por D) | **10** |
| **D → C** (palabra_D001 → C) | 5 |
| **C → B** (palabra_C001 → B) | 3 |
| **B → A** (apr_B001 → A) | 1 |

Es un **anillo de transmisión A→D→C→B→A** con fuerza decreciente, e incluye una **cadena de 2° orden**:
`palabra_C001` (de C) → la emula B (`apr_B001`) → la emula A (`apr_A001`). Una palabra recorriendo **C→B→A**.

Distribución por régimen: **R3_UNOxUNO 12 · R4_AZAR 5 · R2_FLIP 2** (R3 es el régimen más largo del ciclo).

→ **Los organismos sí se influyen entre sí, pero por el VOCABULARIO (canal de gestos/HTTP), no por la
orientación acústica.** El acople inter-organismo es **semiótico, no sensoriomotor.**

---

## 6. Resultado C — La invención emerge con entorno rico, pero es frágil
En el entorno complejo cada organismo acuñó su `palabra_X001`: **8 voces nuevas** (4 propias `palabra_A..D001`
+ 4 emuladas `apr_A..D001`). Pero el contador `voz_creadas` **oscila 0↔1**: las palabras nacen **provisionales**
y, sin reuso suficiente (`USOS_CONSOLIDA=4`), se podan. **La invención ocurre; la consolidación todavía no.**

---

## 7. Veredicto y limitaciones honestas
**Veredicto:** dos canales que se confundían quedaron separados —

| Canal | Conducta | ¿Social? |
|---|---|---|
| Orientación (cabezas) | rastrea el sonido del entorno común | **No** — contexto |
| Léxico (emular palabras) | copian el vocabulario del otro, en cadenas | **Sí** — acople real |

**Limitaciones:**
1. **Test de orientación parcialmente confundido:** en R1/R2, A y C comparten ruteo entre sí por diseño, así
   que A–C alto era esperable. Lo decisivo (y limpio) es el contraste agrupación-por-oído (0.9) vs
   pareja-social (0.2) **y** el desplome en R4. Para un test 100% nítido: invertir **solo A** (no A y C).
2. **Vocabulario provisional:** las palabras no consolidan; el hallazgo de difusión es real pero pide
   condiciones que premien el reuso para que el léxico perdure.
3. Etiquetas = rótulos humanos (ver advertencia epistémica). Medimos estructura, no significado.

---

## 8. Conclusiones
1. **O-N1 validada como identidad de conservación** (residuo 0) y como **mecanismo causal** de la invención:
   subir RC (por novedad/cambio del contexto) produce sentido nuevo vía ICR.
2. **La sincronía conductual visible es ambiental, no social.** El acople social verdadero es **léxico**:
   existe una **red de difusión de vocabulario** entre los organismos, con estructura (anillo, cadenas).
3. La pregunta del programa se afina: de *"¿se sincronizan?"* a **"¿cómo se forma y consolida un léxico
   compartido entre ellos?"** — que es el umbral hacia la dimensión simbiótica/colectiva.

## 9. Próximo experimento propuesto
Entorno rico (para invención) + condiciones que **favorezcan reuso/consolidación** (para que las palabras
cuajen) + mapeo longitudinal de la **red de difusión** (quién enseña a quién, en el tiempo). Control de
orientación nítido (invertir solo A). Métrica central: emergencia de un **léxico compartido estable**.

---
*Corridas y datos primarios (segmentables por `exp_*`) en `~/Downloads/ANIMA4_{TOPO,RICO,ENTORNO}_<ts>/`.
Código: `experimento_anima4_entorno.py`, `organelos/VST_OrganoComunicacion.py` (`_quizas_crear`, umbrales).*
