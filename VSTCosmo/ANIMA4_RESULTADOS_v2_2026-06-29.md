# ANIMA-4 · Resultados corrida v2 (de cero, mundo silencio + baseline)
**Cosmolab / VSTCosmo · para el equipo · 2026-06-29 · run `20260629_042243`**

> **Advertencia epistémica:** los rótulos de audio y de voz (`dolor`, `confusion`, `hambre`…) son
> etiquetas humanas de archivo/banco. NO sabemos qué significan para los organismos. Se mide respuesta
> a configuraciones acústicas y sociales, NO significados humanos.

Corrida limpia desde **vocabulario y metabolismo en CERO**, mundo **silencio/compartido**, con una
**FASE 0 de baseline aislado** (`ANIMA_CONTROL=null`, sin percepción del par) como referencia. 5 fases +
control shuffled, ~10 min/condición, ~22–35 mil filas de fisiología por organismo y condición. Análisis
sobre datos primarios **segmentados por los timestamps del manifiesto** (no por las etiquetas `exp_*`,
para evitar artefactos de alineación).

---

## 1. Hallazgo decisivo (resultado NULO, y por eso importa): NO hubo invención léxica

Desde cero, en ventanas de 10 min, **ningún organismo inventó ni emuló una sola palabra**:

| Organismo | voz_creadas | voz_aprendidas | filas con emulación | repertorio emitido |
|---|---|---|---|---|
| A | 0 | 0 | 0 | sólo innato |
| B | 0 | 0 | 0 | sólo innato |
| C | 0 | 0 | 0 | sólo innato |
| D | 0 | 0 | 0 | sólo innato |

Lo único que emiten es su **repertorio innato**: `dolor`, `confusion`, `hambre`, `fatiga`, `ternura`,
`fon_ternura`, `retirada`, `acuerdo`, `compania`, `comprension`, `insistencia`, `exploracion`…

**Qué prueba esto (y confirma a GPT de forma terminante):** en la corrida anterior "A inventó 2, D acuñó 1"
era **100 % patrimonio histórico acumulado en disco**, NO algo que la sociedad produjera. Al resetear a
cero, la tasa de invención real en 10 min es **exactamente cero**. Los "roles de invención" no eran
sociales: eran el saldo de vidas previas.

**Consecuencia metodológica (importante para el próximo diseño):** la invención léxica es **lenta** —
no ocurre en ventanas de 10 min. Para estudiar léxico y **rutas** (¿la palabra de X llega a Y?) hay que
correr condiciones de **horas**, no de minutos. La maquinaria de trazabilidad de rutas (`voz_id` /
`voz_emulada_de`) quedó **instalada y verificada**, pero esta vez **no tuvo nada que trazar** porque no se
acuñó ninguna palabra. No es un fallo del instrumento: es que el fenómeno no ocurrió en esta escala temporal.

---

## 2. Hallazgo positivo y limpio: el ACOPLE modifica la línea base (lo que GPT pidió probar)

GPT lo planteó exacto: *"la prueba de que el acople hace algo no es 'aparecen roles' sino 'los roles
fijos se modifican bajo acople'."* La **imitación-actividad** (`oao_imitacion_mag`, cuánto imita cada
organismo) da justo eso:

| Condición | control | A | B | C | D |
|---|---|---|---|---|---|
| **BASELINE aislado** | null | **0.000** | **0.000** | **0.000** | **0.000** |
| **PLENA** (todos↔todos) | real | 0.217 | 0.238 | 0.157 | 0.201 |
| CADENA (A←B←C←D) | real | 0.161 | 0.065 | 0.397 | · |
| ESTRELLA (→D) | real | 0.143 | 0.058 | 0.409 | 0.000 |
| PAREJAS (A↔B, C↔D) | real | · | · | · | 0.283 |
| CTRL shuffled | shuffled | 0.053 | 0.056 | 0.267 | 0.000 |

- **Aislado = 0.000 exacto** para los cuatro (91 907 filas). Sin par que percibir, no hay imitación. Limpio.
- **Bajo acople pleno (PLENA), los cuatro imitan a ~0.20.** El comportamiento de imitación **existe solo
  bajo acople**: la línea base (0) **se modifica**. Esa es la prueba pedida, y es sólida.
- El nodo-fuente colapsa correctamente: **D en ESTRELLA = 0.000** (no oye a nadie → no imita). Coherente.
- `·` = la métrica quedó **nula** en esas celdas (PAREJAS A/B/C; D en CADENA): hueco de registro de
  `oao_imitacion_mag` en condiciones direccionales. **No las interpretamos como cero**; quedan como dato
  no confiable de esta corrida (a depurar).

---

## 3. Sincronía ENTRE organismos (el canal relacional propiamente tal): débil, con ESTRELLA asomando

Ojo con la distinción: la tabla anterior es **cuánto imita cada uno** (actividad). La **sincronía** es
**cuán correlacionados están entre sí** (correlación fuera de la diagonal). Son cosas distintas.

| Condición | sincronía imitación (off-diag) | sincronía gestos |
|---|---|---|
| PLENA | 0.037 | −0.001 |
| CADENA | 0.017 | 0.010 |
| **ESTRELLA** | **0.121** | −0.038 |
| PAREJAS | 0.007 | 0.014 |
| CTRL shuffled | 0.029 | −0.019 |

- **ESTRELLA es la única que sobresale** (0.121, ~4× el control shuffle 0.029). Y su estructura interna es
  la **firma de driver común**: los tres seguidores que escuchan a D se correlacionan **entre sí**
  (A–C = 0.19, A–B = 0.10, B–C = 0.07) porque comparten una misma fuente. Es exactamente lo que una
  estrella debería producir.
- Las demás (PLENA, CADENA, PAREJAS) quedan **al nivel del ruido / control**. Gestos: ruido en todo.
- **1 solo ciclo** → ESTRELLA es **sugerente, no concluyente**. Hay que repetirla con varios ciclos.

---

## 4. Corrección honesta de lo que reporté en vivo

Mientras corría te dije que "la sincronía de imitación saltó a 0.26–0.44 con mundo silencio vs ~0.05
antes". **Era una comparación mal hecha** y la corrijo:

- Ese 0.26–0.44 era el `imit_max` = **actividad** de imitación de cada organismo (≈ la tabla §2), NO la
  **sincronía entre** organismos.
- La sincronía real entre organismos (§3) es **modesta** (0.007–0.121). El mundo silencio **sí destapó la
  actividad de imitación desde cero** (0 → ~0.20, §2), pero **no** elevó dramáticamente la **sincronía
  cruzada** respecto a la corrida divergente. La mejora fue en actividad, no en convergencia mutua.

Prefiero decirlo claro: el titular real de esta corrida es el **§1 (nada de léxico desde cero)** y el
**§2 (acople 0→0.20)**, no una gran convergencia social.

---

## 5. Qué quedó probado y qué sigue

**Probado:**
1. Los "roles de invención" previos eran **patrimonio histórico, no producción social** (§1).
2. El **acople genera imitación que el aislamiento no tiene** (0 → ~0.20): la línea base se modifica (§2).
3. La topología **ESTRELLA** deja una firma de driver común coherente, aunque débil y de un solo ciclo (§3).

**Pendiente / próximo diseño:**
- **Condiciones largas (horas), no de 10 min**, con mundo silencio, para dar tiempo a que **emerja léxico**
  y se **poblen las rutas** (`voz_id`/`voz_emulada_de` ya listos para capturarlas).
- **Repetir ESTRELLA con varios ciclos** para ver si el 0.121 es real o ruido.
- **Depurar las celdas nulas** de `oao_imitacion_mag` en condiciones direccionales (PAREJAS/CADENA).
- Medir invención como **delta dentro del bloque** una vez que las ventanas sean largas.

---

## 6. Archivos
- Crudos segmentables: `~/Downloads/ANIMA4_TOPO_20260629_042243/primarios_fisiologia.tar.gz` (24 csv, por
  `exp_*` o por manifiesto `condiciones_20260629_042243.csv` con `ts_real_ini/fin`).
- Matrices: `matriz_imitacion_por_condicion.csv`, `matriz_gestos_por_condicion.csv`.
- Léxico: `difusion_lexica.csv` (todo en cero, confirmado).
- Bitácoras: `bitacoras.tar.gz`. Síntesis original del script: `resumen_social.md` / `informe_social.md`.
- Correcciones de código y protocolo: `ANIMA4_CORRECCIONES_y_protocolo_v2.md`.
