# INFORME — BATERÍA DE TESTS DE LA CÉLULA MADRE FUNCIONAL

**Para:** Equipo transinteligente RMD 2.0 · **Fecha:** 2026-06-23
**Qué se probó:** cargador WAV universal, entrada binaural, dos fuentes independientes,
ablación, determinismo. **Cómo:** `bateria_test.py` (6 suites) sobre el corpus real
`audio_binaural/`. Datos crudos: `CelulaMadre_logs/bateria_*.csv` + `bateria_resumen_20260623.json`.
Simulación 6 s por corrida (salvo donde se indica). Reproducible: `venv/bin/python3 bateria_test.py`.

---

## Resumen ejecutivo (lo que encontramos)

1. **El cargador universal funciona sobre todo el corpus** — PCM 16 y **float IEEE** cargan sin fallar (Brandemburgo incluido). 0 errores.
2. **La diferenciación entre estímulos a IGUAL energía es real (estructural)** — XE, R₂ y OI varían con la energía controlada. Pero la estructura que el organismo capta es **dinámica temporal de energía + complejidad**, NO el timbre: **Ω sigue plano** (~0.46–0.49 en todos).
3. **La lateralidad binaural llega al campo pero no cambia el estado integrado (OI)** en la arquitectura actual (mono ≈ binaural). **NO podemos hablar de "conducta": el organismo todavía NO TIENE ACTUADORES** — falta el motor de orientación a la fuente que sí tenían los experimentos iniciales. La pregunta V117–V122 (¿la lateralidad se expresa?) queda **ABIERTA y no testeable hasta incorporar actuadores.**
4. **Sí importa (poco) qué oído recibe qué** — al intercambiar L/R, el balance invierte signo y el OI se desplaza ~0.008.
5. **Anatomía funcional confirmada** — Homeostasis y Consciencia/LF son load-bearing; C_m es observador (no mueve el OI).
6. **Determinismo perfecto** — misma corrida ×2 = idéntica bit a bit (reproducibilidad, como exige el ADDENDUM).

---

## Suite A — Cobertura de formatos (cargador universal)

Los 8 archivos cargan, campo Φ finito, OI computado:

| archivo | formato | muestras | OI | finito |
|---|---|---|---|---|
| BigBang_pos60deg | soundfile:PCM_16 | 11.2M | 0.338 | ✅ |
| **Brandemburgo** | **soundfile:FLOAT** | 7.4M | 0.307 | ✅ |
| Ondas mixtas | soundfile:PCM_16 | 1.68M | 0.468 | ✅ |
| Voz_Estudio_pos60deg | soundfile:PCM_16 | 1.68M | 0.347 | ✅ |
| musica_pos60deg | soundfile:PCM_16 | 1.68M | 0.311 | ✅ |
| Ruido blanco_neg60deg | soundfile:PCM_16 | 1.68M | 0.483 | ✅ |
| La_pos60deg | soundfile:PCM_16 | 33.6k | 0.298 | ✅ |
| freq_439_pos60deg_largo | soundfile:PCM_16 | 1.44M | 0.507 | ✅ |

**Hallazgo:** el float IEEE que antes rompía ahora entra transparentemente. La ruta usada es
**soundfile** en todos (el fallback wave queda como respaldo).

---

## Suite B — Estructura vs energía (normalizados a misma RMS=0.12, todos ON)

| estímulo | OI | XE | C_m_pico | R₂ | Λ_Cos | e_R | Ω |
|---|---|---|---|---|---|---|---|
| tono (demo) | 0.394 | 0.283 | 0.0 | **0.381** | 0.0012 | 1.30 | 0.488 |
| ruido blanco | 0.366 | 0.301 | 0.0 | 0.617 | 0.0089 | 3.99 | 0.464 |
| **voz** | 0.390 | **0.458** | 0.0 | 0.617 | 0.0091 | 4.05 | 0.463 |
| música | 0.315 | **0.079** | 0.0 | 0.617 | 0.0092 | 4.05 | 0.463 |
| clásica (Brand.) | 0.315 | 0.083 | 0.0 | 0.617 | 0.0092 | 4.05 | 0.463 |

**Hallazgo clave:** con la energía **controlada** (e_R ≈ 4.0 idéntico en ruido/voz/música/clásica),
los estímulos **siguen diferenciándose** → la diferenciación es **estructural, no trivialmente
energética**. La **voz** dispara la mayor exaptación (XE 0.46), música/clásica la menor (0.08);
el **tono puro** tiene R₂ bajo (0.38 vs 0.62) — un seno es "más simple de representar".

**Matiz honesto (no sobre-afirmar):**
- La estructura que el organismo capta es la **dinámica temporal de la energía** (voz = ráfagas/silencios → demanda variable → exaptación; música continua → menos) y la **complejidad** (tono simple → R₂ bajo). **No** es identificación de timbre.
- **Ω no separa** (todos ~0.46–0.49): la diferenciación vive en XE/R₂/OI, no en Ω.
- **C_m = 0 en todos**: con 6 s no hay tiempo para que la metacognición emerja (necesita fracaso sostenido). En corridas largas sí aparece — aquí no es discriminante.

---

## Suite C — Binaural (canales propios) vs Mono (mezcla)

| archivo | OI mono | OI binaural | lateralidad mono/bin | lat_real |
|---|---|---|---|---|
| BigBang_pos60deg | 0.3292 | 0.3297 | 0.6964 / 0.6965 | True |
| Voz_Estudio_pos60deg | 0.3528 | 0.3550 | 0.698 / 0.699 | True |

**Hallazgo (importante y honesto):** abrir el canal binaural **no cambia el estado integrado**
del organismo — OI y lateralidad son casi idénticos en mono y binaural (Δ ≈ 0.0005–0.002). La
lateralidad **llega al campo** (`lat_real=True`).

> ⚠️ **NO se puede concluir "no se expresa en conducta": el organismo AÚN NO TIENE ACTUADORES.**
> Los experimentos iniciales (V117–V122) tenían un motor de **orientación a la fuente** (mover la
> "cabeza" hacia el sonido); aquí ese canal de salida **no existe todavía** — solo medimos el estado
> interno (OI, etc.). Por tanto sería injusto decir que la lateralidad "no se expresa". Lo correcto:
> **en la arquitectura actual la lateralidad no modifica el OI, y su posible expresión conductual es
> una pregunta ABIERTA que NO se puede testear sin antes incorporar actuadores.**

Además: la métrica `lateralidad` (~0.70) está **dominada por la asimetría intrínseca de los
hemisferios** (τ_L=30 vs τ_R=300), que tapa la diferencia de canal del audio (0.6964 vs 0.6965).

Además: la métrica `lateralidad` (~0.70) está **dominada por la asimetría intrínseca de los
hemisferios** (τ_L=30 vs τ_R=300), que tapa la diferencia de canal del audio (0.6964 vs 0.6965).
Para medir la lateralidad *del audio* habría que aislarla de la asimetría hemisférica — pendiente.

---

## Suite D — Asimetría L/R (swap de oídos)

| asignación | OI | balance | lateralidad | coher |
|---|---|---|---|---|
| voz→L, ruido→R | 0.3546 | **−0.327** | 0.7002 | −0.5975 |
| ruido→L, voz→R | 0.3633 | **+0.327** | 0.7045 | −0.6206 |

**Hallazgo:** **sí importa qué oído recibe qué.** El `balance` invierte signo exactamente
(como debe), y el OI se desplaza ~0.008 al intercambiar. Es un efecto **pequeño pero real**:
los hemisferios no son simétricos (distinto τ), así que la asignación L/R cambia algo. No es
ruido — es asimetría estructural.

---

## Suite E — Anatomía funcional (ablación, Voz_Estudio binaural)

| config | OI | ΔOI | qué muere |
|---|---|---|---|
| TODO ON | 0.355 | — | (baseline) |
| sin R₂ | 0.222 | **−0.133** | R₂→0, cae LF |
| sin LF | 0.222 | **−0.133** | LF→0 |
| sin Exaptación | 0.304 | −0.051 | XE→0 |
| sin C_m | 0.357 | **+0.002** | nada (C_m es observador) |
| sin Homeostasis | 0.188 | **−0.167** | H→0 |

**Hallazgo:** jerarquía de carga del OI confirmada en un estímulo real: **Homeostasis (−0.167)
> Consciencia/LF (−0.133) > Exaptación (−0.051)**; **C_m no mueve el OI (+0.002)** → es un
observador metacognitivo, no un driver. Coincide con la ablación previa (Blue Monday). La
ablación por interruptor es causalidad **medida**, no afirmada.

---

## Suite F — Determinismo

`La_pos60deg` ×2 → **CSV idéntico bit a bit: True.** El sistema es determinista (semillas
fijas): misma entrada ⇒ misma corrida. Reproducibilidad total (lo que pedía el ADDENDUM).

---

## Conclusiones para el equipo

1. **Ingeniería sólida:** carga universal de WAV, binaural real, dos fuentes independientes,
   ablación y determinismo — todo verificado.
2. **Diferenciación estructural confirmada** a igual energía, pero **vive en el estado
   organísmico (XE/R₂/OI), no en Ω** — y la "estructura" es dinámica-temporal/complejidad, no timbre.
3. **La lateralidad: pregunta ABIERTA, no hallazgo negativo.** La info lateral llega al campo
   pero no cambia el OI en la arquitectura actual. **No podemos juzgar su expresión conductual
   porque el organismo NO TIENE ACTUADORES** (falta el motor de orientación a la fuente de los
   experimentos iniciales). Sería injusto concluir "no se expresa". La pregunta V117–V122 queda
   abierta y **solo será testeable tras incorporar actuadores**.
4. **Limitaciones registradas (no esconder):** **no hay actuadores** (no se puede medir conducta
   orientada); Ω coarse (no separa timbres); C_m necesita corridas largas (>6 s) para emerger;
   la métrica de lateralidad está contaminada por la asimetría hemisférica intrínseca; ritual
   sin calibrar al rango del audio.

## Próximos experimentos sugeridos
- **Incorporar ACTUADORES** (orientación a la fuente, como V117–V122) — *prerrequisito* para poder
  preguntar si la lateralidad se expresa en conducta. Hasta entonces, la pregunta no es testeable.
- **Corridas largas** (60–200 s) de la Suite B para ver emerger C_m y el transitorio.
- **Aislar la lateralidad del audio** de la asimetría hemisférica (medir Δ binaural−mono por organelo).
- **Atacar Ω**: si la diferenciación no vive en Ω, redefinir Ω (leer más del campo, no la media).
- **Calibrar el ritual** al rango de señales derivadas del audio.

*Archivos: `bateria_test.py`, `CelulaMadre_logs/bateria_*.csv`, `bateria_resumen_20260623.json`.*
