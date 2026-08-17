# ANIMA — Multi-Organismos Individuales (de la díada a la sociedad)
**Cosmolab / VSTCosmo · documento para el equipo · 2026-06-28**

> **Advertencia epistémica:** las etiquetas de audio y de voces ("voz", "ternura", etc.) son **solo
> etiquetas**. No sabemos qué significan los sonidos para los organismos. Todo describe respuesta a
> configuraciones, no a categorías de significado.

---

## 1. Por qué — y qué hicimos

La díada (A↔B) solo tiene **reciprocidad directa**. La sociabilidad real (coaliciones, reputación,
reciprocidad indirecta, un tercero que observa una interacción ajena) **emerge recién con N>2**. Por eso
añadimos **Organismo C y Organismo D**, y construimos el modelo de percepción para que los experimentos
puedan tener **cuántos organismos quieran, oyéndose en la topología que quieran**.

**Principio rector (no negociable):** los organismos son **unidades experimentales componibles**, NO una
sociedad rígida de 4. Cada experimento decide **cuántos** participan, **quién oye/imita a quién** y **qué
hace** cada uno. La sociedad de 4 es la *infraestructura*; el experimento define el *recorte*.

Camino del proyecto: **díada → primera sociedad básica (esto) → célula eucariota (simbiosis de dos)**.

---

## 2. Los cuatro organismos

| Organismo | Puerto (página) | Contenedor | Rol (entrypoint) |
|---|---|---|---|
| A | http://localhost:7788 | `anima-a` | `a` |
| B | http://localhost:7799 | `anima-b` | `b` |
| **C** | http://localhost:**7810** | `anima-c` | `c` |
| **D** | http://localhost:**7820** | `anima-d` | `d` |

- **Mismo código, distinta identidad.** `VST_CelulaMadre_WebLive_C.py` y `_D.py` son copias de `_A.py`
  con la identidad ajustada (etiqueta/título); todo lo importante (puerto, ID, par, roster) viene por
  **variables de entorno**. El reactor cosmosemiótico usa `VST_RC_A` (C/D no necesitan variante propia).
- Cada uno tiene **su propia página** (cabeza 3D, cajas, panel de LEDs, descargas), su **disco propio**
  (`anima_c_data` / `anima_d_data`, memoria/codebook persisten) y su biografía en `Docker_Historia`.
- Observatorio común en `:9100` (ver §7).

---

## 3. El modelo de percepción: DOS canales

Cada organismo percibe a los demás por **dos vías distintas** — importante no confundirlas:

1. **AUDIO (lo que OYE).** Sonido binaural L/R que entra al Soma. Abre la "compuerta" de escucha
   (`e_oida = max(L,R) > umbral`), dispara la orientación, y modula arousal/expresión. Configurable por
   oído (ver §4).
2. **GESTOS (lo que IMITA).** El **contenido** que copia/converge viaja por HTTP (`/comunicacion/estado`),
   NO por el sonido. La memoria ecoica del OAO guarda los **gestos** (frecuencia/intensidad/pausa/
   repetición) del/los organismo(s) percibido(s).

**Regla que conecta ambos: "imito a quien oigo".** El OAO deriva de quién aprende **a partir de la fuente
de audio activa** (`RUN.cfg`): si el oído de relación está en "Escuchar C", imita a C; si está en "Otros
organismos", imita a todos. → Cambiar la fuente cambia, a la vez, **a quién oye y a quién imita**.

---

## 4. Fuentes de entrada (cómo se elige a quién oír)

En el menú de fuentes de cada página (selector por oído **L** y **R**) aparecen ahora:

- **🗣 Escuchar A / B / C / D** — oír (e imitar) a **UN** organismo específico. Es lo que permite armar
  **cadenas y topologías arbitrarias**. (Cada organismo ofrece a los *otros*, no a sí mismo.)
- **🧑‍🤝‍🧑 Otros organismos — los N demás** — la **mezcla** de las voces de todos los demás (campo acústico
  común: "todos oyen a todos"; la simultaneidad emerge de ellos, sin turnos impuestos).
- **🧪 control · NULL_STATE / SHUFFLED_STATE** — controles de falsación sobre el par.
- (más: canales del Rødecaster vía servidor, archivos .wav, demos, silencio).

El **otro oído** se elige libremente (el mundo por un canal del Rode, silencio, otro organismo, etc.).

> Técnico: la fuente "otros_organismos" lleva una **lista de URLs configurable** (`ANIMA_OTROS_URLS`),
> que puede ser un **subconjunto** — base para percepción por subconjuntos arbitrarios.

---

## 5. Imitación multi-vecino ("imito a quien oigo")

- El OAO ya **no** está atado a un solo par. `_roster_estado_urls()` deriva los vecinos a imitar de la
  **fuente de relación activa**: lista de "otros_organismos" → aprende de todos; "Escuchar X" → aprende
  de X. `_vecinos_estados_control()` sondea los gestos de cada uno (cacheado por-URL, respeta
  `ANIMA_CONTROL=real|null|shuffled`).
- En el paso, `OAO.observar()` se llama **por cada vecino** → la ecoica acumula los gestos de varios → el
  sesgo de imitación refleja al **colectivo** que oye.
- **Configurable por experimento sin tocar código:** cambias la fuente de audio y la imitación la sigue.

---

## 6. Panel de LEDs: voces de los otros (#19–24)

En cada página, tras los canales del Rode, el panel muestra **6 medidores #19–24 = la voz ESTÉREO (L/R)
de los OTROS organismos**, en orden alfabético:

| | #19 / #20 | #21 / #22 | #23 / #24 |
|---|---|---|---|
| **A** | B (L/R) | C (L/R) | D (L/R) |
| **B** | A | C | D |
| **C** | A | B | D |
| **D** | A | B | C |

- Cada organismo sondea un endpoint liviano **`/voz_nivel`** de los demás (su voz estéreo en vivo).
- La **voz propia** se mantiene en su propio indicador ("voz propia L/R").
- **Se adapta al roster:** si corren 2-3 organismos, aparecen solo esos (no fuerza 4).

---

## 7. Observatorio `:9100` — pestaña 🌐 Sociedad

Nueva pestaña (junto a En vivo / Historia / Circuito vivo) con una **grilla de los N organismos activos**.
Cada tarjeta: **cabeza 3D** (gira/late/sonríe en vivo), voz actual, imitación, ¿oye?, Ω del campo y cara
(OVE). Los apagados quedan en gris.

- El observatorio es **roster-based**: arma `ROSTER = [A,B]` + C/D si `ANIMA_C_URL`/`ANIMA_D_URL` están
  definidas; el poller y `/datos` exponen `d.roster` + el estado de todos. Se adapta a N.
- Las pestañas anteriores siguen intactas (cambios aditivos).

---

## 8. Cómo correr experimentos (la parte importante)

Todo se controla por la **API HTTP de cada organismo** (igual que las baterías):
`POST /start {cfg con left_src/right_src}`, `POST /control {action:stop}`, `POST /mute {left,right}`.

**Decides en runtime:**
- **Cuántos** — arrancas solo los organismos que uses; observatorio y medidores se adaptan a N.
- **Quién oye/imita a quién** — la fuente del oído de relación de cada uno.
- **Qué hace** — fuente por oído (mundo / un par / todos / silencio / control).

### Topologías (ejemplos)
- **Cadena A→B→C→D→A:** oído de relación de A = *Escuchar B*; B = *Escuchar C*; C = *Escuchar D*;
  D = *Escuchar A*. Cada uno oye e imita a su sucesor.
- **Sociedad plena (todos↔todos):** oído de relación de cada uno = *Otros organismos*.
- **Estrella / líder:** todos = *Escuchar A* (un emisor que el resto sigue).
- **Dos parejas:** A↔B y C↔D (cada uno *Escuchar* a su pareja).
- **Control:** algún organismo en *NULL_STATE* (no percibe) o `ANIMA_CONTROL=shuffled` (rompe la
  contingencia temporal) para falsar el confound.

Se puede hacer a mano en las páginas, o automatizar en un **script de experimento** (`experimentos/`)
que haga `/start` a cada organismo con su configuración.

---

## 9. Referencia de configuración (variables de entorno por organismo)

| Variable | Qué hace |
|---|---|
| `VST_PUERTO` | puerto del organismo (7788/7799/7810/7820) |
| `VST_ORGANISMO_ID` | ANIMA_A…ANIMA_D |
| `VST_COMUNICACION_PEER` | par fijo para gestos (anillo por defecto); la UI puede sobreescribir con "Escuchar X" |
| `ANIMA_ESCUCHAR_TODOS` | `1` → autostart oyendo la MEZCLA de todos (campo acústico) |
| `ANIMA_OTROS_URLS` | roster: voces de los demás (coma-separadas) — fuente de "Otros organismos", "Escuchar X" y medidores #19-24 e imitación multi-vecino |
| `ANIMA_OIDO_PAR` | oído por el que entra la relación (R/L) |
| `ANIMA_MUNDO_CANAL` | canal del Rode como "mundo" (vacío = silencio/basal) |
| `ANIMA_CONTROL` | `real \| null \| shuffled` (falsación, global) |
| `ANIMA_C_URL`, `ANIMA_D_URL` | (en `anima-conversacion`) para que el observatorio vea a C/D |

Endpoints nuevos por organismo: **`/voz_nivel`** (voz estéreo propia, para los medidores de los otros).
En el observatorio: **`/datos`** ahora trae `roster` + el estado de todos.

---

## 10. Estado actual y próximos pasos

**Hecho y verificado:**
- ✅ Organismos C (7810) y D (7820) vivos; infraestructura de 4 (contenedores, páginas, persistencia).
- ✅ Campo acústico común ("Otros organismos" = mezcla de todos) — todos oyen a todos.
- ✅ Imitación multi-vecino ("imito a quien oigo"), configurable por la fuente activa.
- ✅ Fuentes "Escuchar X" por organismo → cadenas y topologías arbitrarias.
- ✅ Medidores LED #19-24 (voces estéreo de los otros), roster-adaptativos.
- ✅ Pestaña 🌐 Sociedad en el observatorio, roster-adaptativa.

**Limitación honesta / Fase 2 (sugerido):**
- Hoy la percepción por **subconjuntos arbitrarios** se hace eligiendo "Escuchar X" o "Otros organismos".
  El siguiente paso natural sería un **vecindario configurable por experimento** (cada organismo con una
  lista de vecinos data-driven) y un **orquestador de experimentos sociales** (defines la topología en un
  archivo y arranca a todos en esa configuración).
- El observatorio Circuito vivo / cajas siguen pensados A/B; generalizarlos a N es trabajo futuro.

---

## 11. Mapa de archivos

```
Célula_Madre/
  web/  VST_CelulaMadre_WebLive_A.py · _B.py · _C.py · _D.py
        (organismos; fuentes Escuchar X / Otros organismos; imitación multi-vecino;
         endpoint /voz_nivel; medidores #19-24; _otros_voz_estereo, _roster_estado_urls)
  conversacion/ vst_conversacion.py   (observatorio :9100 roster-based + pestaña 🌐 Sociedad)
  docker/ docker-compose.yml          (servicios anima-a/b/c/d + conversacion con ANIMA_C/D_URL)
          entrypoint.sh               (roles a|b|c|d|mcp|conversacion)
Docker_Historia/                       (datos primarios por organismo)
```
