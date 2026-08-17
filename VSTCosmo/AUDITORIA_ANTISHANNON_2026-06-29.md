# AUDITORÍA ANTI-SHANNON — Organismo ANIMA / Célula Madre
**Cosmolab / VSTCosmo · 2026-06-29 · barrido completo "de punta a rabo"**

> **Principio auditado.** En Cosmosemiótica el organismo NO puede usar el paradigma de Shannon
> —información-en-la-señal, canal, código, bandas de frecuencia, entropía de Shannon— DENTRO de su
> cognición / representación / afecto / acople. *El significado no está en la señal.* La representación
> es **diferencia estructural (Δ_struct) que sostiene acoplamiento (A_sys-env)** (O-N3.1, O-N3.2).
> Lo que sí es legítimo: **captar** el mundo (RMS, presencia de canal) y **sintetizar** audio de salida
> (voz). El delito es **analizar en bandas / extraer "información" espectral para representar o sentir**.

## Método
Barrido de los 76 .py del organismo vivo (`Célula_Madre/`) + censo mecánico de los ~274 `v*.py`
históricos de la raíz. 5 auditores en paralelo, cada hit leído en contexto y clasificado en
PROHIBIDO / DUDOSO / ACEPTABLE. Patrones: filtros de banda (butter/filtfilt/lfilter/highpass/lowpass…),
Fourier (fft/rfft/spectrogram/stft), rasgos espectrales (centroide/planitud/mfcc/mel/cepstrum),
entropía de Shannon (−Σp·log p / log2 / mutual-info), y el caso sutil del **un-polo `env+=`/`x−env`
usado como par highpass/lowpass perceptual**.

---

## TITULAR
- **El archivo histórico raíz (274 `v*.py`) es deuda INERTE.** Verificado: el organismo vivo **no
  importa ningún `v*.py` raíz** (ni `import`, ni `importlib`, ni `exec`). Sólo importa los `VST_*`
  (organelos) y carga `campo/Célula_Madre_Funcional_001.py`. Las citas a `vNN` en el genoma son
  genealogía en comentarios, no imports. → Limpiar la raíz es **opcional/cosmético**.
- **La deuda Shannon ACTIVA son 3 focos en código vivo** (+ 1 batería contaminada + 1 función en el
  monolito archivado). Todo lo demás está **limpio** y, en varios casos, documenta su anti-Shannon.

---

## DEUDA ACTIVA — 3 focos, por gravedad

### 🔴 FOCO 1 — Percepción por bandas espectrales + codebook aprendido  *(el más grave: es el corazón)*
**`campo/Célula_Madre_Funcional_001.py` · `_estr_rec` (L346-360) → `estructura` / `estructura_L/R`**
- L351: `X = |rfft(win·hanning)|` — FFT de la ventana de audio.
- L352: `flat = exp(mean(log X)) / mean(X)` — **planitud espectral = entropía de Wiener**; `tonalidad = 1−flat`.
- L354-358: `Xlow` partido en **32 bandas** → matching coseno contra un **codebook de prototipos espectrales**.
- L303-305: el codebook (32 bandas × K) es **estado aprendido y persistido** (snapshot/restore).
- L404-413: el organismo **APRENDE el codebook** (`codebook += β·(bins − codebook)`) → un **código espectral aprendido**.
- **Por qué es lo peor:** esto es la **membrana sensorial** que produce el SENTIDO (`estructura`) y la
  **brújula del forrajeo** (`estructura_L/R`, L414-440). El significado se lee del contenido espectral
  y de un *código* aprendido — la encarnación máxima de Shannon (canal + código) dentro del acople.
  El comentario L406 *"Sin Shannon: aprende de lo que vive"* **niega lo que el código implementa.**

### 🔴 FOCO 2 — Afecto derivado del espectro de cada voz
**`organelos/VST_OrganoComunicacion.py` · `_afecto_acustico` (L504-522)**
- L506-520: `rfft` → **centroide espectral** ("brillo") → arousal; **planitud Wiener** → valence;
  segundo par de `rfft` para "contorno de brillo" → valence.
- L521-522: el espectro se convierte en `arousal`/`valence` — la **representación afectiva** de toda
  voz nueva subida al repertorio (R2D2), que luego dirige la síntesis emitida.
- **Por qué:** localiza el afecto en el **contenido espectral de la señal** — contradice directamente
  *"el significado no está en el tono"*. El comentario L491 *"Anti-Shannon… se mide"* es irónico.
- **Tiene reemplazo limpio ya presente:** `_afecto(fila)` (L701-709) deriva arousal/valence desde la
  **fisiología** (RC, E, lateralidad, OI, H, necesidad) — vía sin espectro. La reparación es enrutar
  el afecto del sonido por consecuencia corporal / acople, no por FFT.

### 🟠 FOCO 3 — Hemisferios como filtro un-polo HP/LP  *(el disparador de esta auditoría)*
**`organelos/VST_OrganoHemisferios.py` · `_drive` (L66-73)**
- L69-70: `alpha = 1/τ`; `env += alpha·(energia − env)` — **lowpass un-polo** canónico.
- L71-73: modo `lento` → `env` (sostenido/LP); modo `rapido` → `energia − env` (**highpass**, complemento exacto).
- L84/95: el `drive` filtrado forza el campo Φ → de ahí salen `hemi_R2` (predicción) y `hemi_lateralidad_func`.
- L20-27 (docstring): vocabulario *"highpass/lowpass temporal… equivale al highpass/lowpass espectral de v121"*.
- **Diagnóstico ya establecido:** la distinción rápido/lento debe EMERGER de las dos escalas de tiempo
  de acople (τ), no de un par de filtros de banda. Además R₂ está muerto (forzamiento dipolar media-cero
  + campo saturado). Rediseño: sólo-τ, sin `env`/`x−env`, sin clip que congela, readout = Δ_struct.

### 🟡 Colateral — Batería que valida con vocabulario Shannon
**`experimentos/bateria_hemisferios.py`** — criterio **C2 "diferenciación HP/LP"** (L21-22, 50, 96-102, 145)
formula la validación de la percepción en el paradigma de banda. Hay que reescribirla en términos de
τ y Δ_struct (no "highpass/lowpass"). *(Escrita en esta misma sesión; deuda propia.)*

### ⚪ Archivado — Entropía de Shannon en deliberación (monolito viejo, NO activo)
**`campo/VST_Celula_Madre_001.py` · `calcular_D_conflicto` (L379-384)** computa `−Σp·log(p)` literal
sobre softmax de valencias → `D_conflicto` en la deliberación. Es el **monolito antiguo**; el campo
Funcional vivo **no porta** esta función. Excisar es higiene, no urgencia.

---

## LIMPIO (verificado, explícito)
- **Organelos:** Alteridad, Expectativa, Homeostasis, HomeostasisEmergente, Memoria, Metabolismo,
  ValorEcologicoVoz — sus `EMA` suavizan **escalares ya derivados** (efecto/intención/error, A_sys-env
  y su derivada/varianza, utilidad), **no** separan bandas de un percepto. `*_entropia` en Homeostasis
  es el **nombre** de una margarita Daisyworld (población en [0,1]), no −Σp·log p. `novedad/sorpresa`
  en Memoria salen de familiaridad estructural y |ΔA|, no de −log p.
- **Fonador:** `iirpeak/lfilter` es un **VCF de síntesis de salida** (formante ARP 2600), no análisis.
- **Genoma** (Genoma, Bloque05, Bloque07, Bloque08), **díada, MCP, conversación**: sin DSP en cognición;
  sólo comentarios que niegan Shannon y labels de UI.
- **Web** (WebLive A/B/C/D): I/O + orquestación; pasa los buffers crudos `(L,R)` al soma y sólo computa
  **RMS + medidores VU**. B/C/D heredan la limpieza de A.
- **Audio** (AudioServer, Transcriptor, ReconocedorMusica, monitor): captura RMS / compuertas de energía
  / resample hacia Whisper / huella a Shazam — **I/O legítimo**, fuera del lazo afectivo.
- **Síntesis de mundo:** el FFT de ruido rosa (`_pink_noise`) **genera estímulo**, no filtra percepción.

---

## RUTA DE REPARACIÓN (propuesta, por gravedad × dependencia)
1. **FOCO 2** primero — reemplazo limpio ya existe (`_afecto` fisiológico); alto valor, bajo riesgo.
2. **FOCO 3** — contenido y diseño ya listos (hemisferios sólo-τ + Δ_struct) + reescribir batería C2.
3. **FOCO 1** — el grande: rediseñar la membrana perceptual para que `estructura` emerja de Δ_struct
   y acople, sin FFT/bandas/codebook. Es trabajo teórico de fondo (qué *es* percibir sin espectro).
4. **Higiene** — excisar `calcular_D_conflicto` del monolito; limpieza opcional del archivo raíz.

*Auditoría read-only; ningún archivo modificado. Detalle por archivo en los transcripts de los 5 auditores.*
