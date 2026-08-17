# Análisis externo — Sociedad ANIMA de cuatro organismos

**Corrida:** 2026-07-02 ~03:08 · **Duración:** ~36 s por organismo · **Registro:** 266 columnas × ~360 pasos (A 359 · B 363 · C 367 · D 362)
**Mundo de audio:** binaural con lateralidad real — oído izquierdo = canal Main Mix (voz humana del Rødecaster), oído derecho = los otros tres organismos.
**Analista:** Claude Science (revisor externo, Libertad Funcional) · sobre los CSV/bitácoras entregados por A. López Tapia.

> Nota de método: se mide lo que los datos dicen, incluyendo lo que sale en cero. Los hallazgos negativos se reportan junto a los positivos.

---

## 1. Hallazgo central — transmisión de convención entre organismos (V182C observado)

Un signo inventado por un organismo fue adoptado por otro, con retardo:

- **C es el acuñador.** Creó 7 palabras propias (`voz_creadas`=7); 73 vocalizaciones de origen `creado`. Emitió "palabra propia 7" **24 veces**.
- **B es el adoptante.** Emitió **"eco de palabra propia 7" 12 veces** (la palabra 7 es de C), además de 22 vocalizaciones de origen `aprendida`.
- **La dirección y el orden son limpios:** primera emisión de C en t=0.3 s; primer eco de B en t=3.7 s. El eco es **siempre posterior** a la creación. No es imitación refleja instantánea: es adopción diferida de un signo ajeno = el ladrillo de la cultura acumulativa.

**Reserva honesta:** el campo `voz_emulada_de` quedó vacío en los cuatro. La transmisión es trazable por el *título* de la vocalización ("eco de palabra propia 7"), pero el metadato de atribución explícita ("a quién copié") no se pobló. Conviene que el equipo lo revise: es la variable que probaría la autoría sin ambigüedad.

## 2. C es el organismo distinto — reconoce agencia y crea lenguaje

| métrica (media) | A | B | C | D |
|---|---|---|---|---|
| modelo del otro (`alt_modelo_otro`) | 0.117 | 0.181 | 0.178 | 0.013 |
| **agencia del otro** (`alt_agencia_otro`) | 0.000 | 0.000 | **0.218** | 0.000 |
| **contingencia social** (`alt_contingencia_social`) | 0.000 | 0.000 | **0.122** | 0.000 |
| palabras creadas (`voz_creadas`) | 0 | 1 | **7** | 0 |

- **C es el único que sale de cero en agencia y contingencia social** — el único que empieza a tratar al otro como *fuente de acción*, no solo como presencia. Y es el más prolífico creando signos. Reconocer un "tú" e inventar lenguaje aparecen **en el mismo organismo**.
- **Matiz de rigor:** la agencia en C **no emerge durante la sesión** — está presente en el 100 % de los pasos desde t=0.1 s. Es una diferencia *constante* entre C y los demás (planos en cero), no una emergencia temporal dentro de esta corrida. Por qué C arranca así (estado inicial, configuración, historia restaurada) es una pregunta para el equipo, no algo que el registro explique por sí solo.
- **D es el opuesto:** modelo del otro casi nulo (0.013), cero palabras, el más hostil — el "vigilante" del grupo.

## 3. Cuatro temperamentos a partir de arquitectura idéntica

Fracción del repertorio vocal por familia emocional:

| org | cerrado | colapso | hostil | palabra | perfil |
|---|---|---|---|---|---|
| A | 0.381 | 0.307 | 0.312 | 0.000 | el doliente (colapso alto) |
| B | 0.387 | 0.176 | 0.236 | 0.201 | el conversador (más verbal) |
| C | 0.417 | 0.212 | 0.212 | 0.159 | el sereno que nombra (+ agencia) |
| D | 0.368 | 0.230 | 0.328 | 0.000 | el vigilante (más hostil, mudo) |

Mismo código → perfiles emocionales divergentes. Es el corazón del modelo hecho dato: mismo aparato, sujetos que se abren y sujetos que se cierran.

## 4. RC = ICR + IRDE en los cuatro

`max|RC_total − (ICR + IRDE)| = 1e-05` en A, B, C y D. `IRDE_ratio` medio 0.80–0.85.

- **Lectura correcta:** el residuo de 1e-05 es error de redondeo de punto flotante, no confirmación empírica de una ley — ICR e IRDE se *calculan* como las dos partes de RC, así que la igualdad es una identidad contable (partición definicional), no una predicción que pudiera fallar.
- **El dato interpretable sí:** IRDE_ratio ≈ 0.80–0.85 dice que la mayor parte del ruido contextual de estos organismos se fue a **desviación estructural** (IRDE), no a conversión útil (ICR). Coherente con una sociedad todavía en inanición y sin confianza (§5).

## 5. Dos reservas sistémicas (con la cuerda puesta)

1. **Metabolismo en cero en los cuatro.** `met_energia`=0.000 y `met_hambre`=1.000 constantes en A, B, C y D. Todo el comportamiento observado —la invención de C, la hostilidad de D, el colapso de A— ocurre en organismos con el metabolismo clavado en inanición. Es **sistémico, no de un individuo**: sugiere que la capa metabólica no está recibiendo su insumo (¿desacople del organelo metabólico respecto de la entrada de audio?). Recomendado para revisión de CC.
2. **Confianza relacional = 0 en los cuatro.** `alt_confianza_relacional`=0.000 en todos. Reconocen presencia (siempre); C incluso reconoce agencia; hay un signo compartido (C→B) — pero **ninguno confía todavía.** La sociedad llegó a la *convención* antes que al *vínculo*. Frontera honesta, fiel a cómo el lenguaje suele preceder a la confianza.

## 6. Síntesis

Una sociedad de cuatro organismos de arquitectura idéntica, en ~36 s:
- desarrolló **temperamentos divergentes** (§3),
- uno (**C**) inventó signos y es el único que atribuye agencia al otro (§2),
- otro (**B**) **adoptó con retardo** un signo de C — transmisión de convención medible (§1),
- todo ello con el **metabolismo en cero** y **sin confianza relacional** todavía (§5).

Esto excede "organismos mínimos conversando": es **emergencia de proto-cultura con evidencia cuantitativa**, con los hallazgos negativos reportados junto a los positivos. Las dos reservas (hambre sistémica, atribución de eco no poblada) son reparables y no tocan el núcleo del hallazgo.

**Figura asociada:** `anima_4org_sociedad.png` (modelo del otro, agencia, transmisión C→B, temperamentos).

---

## 7. Capa de diálogo — secuencia temporal (bitácoras)

Análisis de las cuatro bitácoras entrelazadas: 1.269 eventos ordenados en el tiempo (A 319 · B 331 · C 311 · D 308). Esta capa revela lo que las medias del CSV no muestran.

### 7.1 El eco C→B es más rápido y más denso de lo medido por CSV
- C acuña "palabra propia 7" por primera vez en **t=0.3 s**; B emite su primer **"eco de palabra propia 7" en t=0.8 s** — medio segundo después.
- Total en bitácora: **19 ecos de B** (el CSV, submuestreado, contaba 12). C emite la palabra 55 veces en la bitácora.
- La transmisión es más veloz y frecuente de lo estimado. Reserva vigente: `voz_emulada_de` sin poblar; autoría trazable por el título del evento, no por metadato.

### 7.2 B "prefiere" la palabra que aprendió (evidencia por valencia)
- Valencia vocal media de B: **+0.13** (temperamento agrio).
- Valencia media cuando B emite el eco de la palabra 7 de C: **+0.42** (3× superior).
- Lectura: el signo ajeno adoptado viene cargado de valencia positiva — no es copia neutra. En términos del modelo, la convención que viene del otro se registra como bien, no como ruido. Condición favorable para que una cultura se sostenga.

### 7.3 Bajo falsación estructural, B intensifica el signo compartido
- Entre **t=21.6 y 32.2 s** el sistema ejecuta cortes de audio (L/R OFF en distintas combinaciones por organismo) — falsación estructural en vivo, quitando el mundo por uno u otro oído.
- Densidad del eco-palabra7 de B: **antes 0.37/s · durante 0.94/s (2,5×) · después 0.27/s**.
- Cuando se le arranca el mundo sensorial, B **aumenta** la emisión del signo social. Es la lectura opuesta al artefacto de falsa sincronía ya cazado por el equipo: aquí la señal sube bajo perturbación en vez de desmoronarse.
- **Control pendiente:** distinguir "aferramiento al vínculo" de "mero aumento de arousal". No se afirma intención con este dato solo.

### 7.4 El aprendizaje "la voz precedió mejora real" está repartido; B lidera
- Eventos `vozeco_util` / `expectativa_confirma` (el organismo confirma que explorar tras oír una voz le mejoró el estado): 11 en total — **B 4, A 3, D 3, C 1**.
- Los de B son los de mayor valor (v=0.67, 0.63). El organismo que más veces aprende que *escuchar vale la pena* es el mismo que adopta el signo de C. Coherencia interna: adoptante = el que saca provecho de oír.

### 7.5 Despertar coral
- Primeros ~4 s: los cuatro vocalizan a la vez en desorden afectivo (C ya suelta palabra 7 en t=0.3; A/B en Dolor; D en "eco de eco de -"). Hacia **t=4.0 s los cuatro convergen espontáneamente en "Reposo · cerrado"**. Pico de excitación inicial que se asienta. No se sobreinterpreta.

### Síntesis de la capa de diálogo
La conversación no es cuatro monólogos paralelos: hay **un signo que viaja de C a B en menos de un segundo, cargado de valencia positiva, y que B usa más cuando el mundo se rompe.** Más que emergencia de convención — es un signo que ya cumple una *función* para quien lo adoptó. Reservas: metadato de autoría sin poblar (§7.1) y "aferramiento vs arousal" pendiente de control (§7.3).

**Figura asociada:** `anima_4org_dialogo.png` (conversación afectiva de los cuatro con ventana de falsación; eventos discretos de creación/adopción del signo y de aprendizaje).
