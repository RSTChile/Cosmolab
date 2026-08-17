# INFORME — LABORATORIO EN VIVO DE LA CÉLULA MADRE (v2 de la interfaz)

**Para:** Equipo transinteligente RMD 2.0 · **Fecha:** 2026-06-23
**Criterio de éxito (cumplido):** la Célula Madre se observa **en vivo**, con entrada
**biaural real o simulada**, seleccionando audios desde `audio_binaural`, mostrando la
fisiología de sus organelos en **múltiples ventanas temporales mientras corre**.

---

## 1. Archivos creados / modificados

| Archivo | Acción | SHA1 (12) | Líneas |
|---|---|---|---|
| `VST_CelulaMadre_WebLive.py` | **NUEVO** — laboratorio en vivo | `6aea2092b2e8` | 681 |
| `Célula_Madre_Funcional_001.py` | modificado — soma con señales binaurales v2 | `0e624e1dbafd` | 413 |
| `pruebas_biaural.py` | **NUEVO** — suite de pruebas Req9 | `742901229065` | 40 |
| `VST_CelulaMadre_Web.py` | **intacto** (interfaz anterior, compatible) | `e94353fd7adc` | 359 |
| `_backups/*.20260623_*.bak.py` | copias de seguridad previas a editar | — | — |

> El motor **validado** y la interfaz anterior **no se rompieron**: el modo archivo mono
> sigue idéntico (invariante de ingeniería verificado bit a bit en el turno previo).

---

## 2. Cómo ejecutar

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo
venv/bin/python3 VST_CelulaMadre_WebLive.py     # → http://localhost:7788  (laboratorio en vivo)
venv/bin/python3 VST_CelulaMadre_Web.py         # → http://localhost:7777  (interfaz anterior, sigue OK)
venv/bin/python3 pruebas_biaural.py             # corre la suite Req9 → CSVs en CelulaMadre_logs/
```

En el laboratorio: elige **fuente** (archivos del proyecto / demo / subir / en vivo),
canal **izquierdo** y **derecho**, interruptores por organelo, **Iniciar** → los gráficos
se llenan **en tiempo real** (SSE). Botones: Pausar/Reanudar, Detener, Limpiar, ⬇CSV.

---

## 3. Dependencias

- **Sin dependencias nuevas obligatorias** (stdlib + numpy + chart.js por CDN).
- **OPCIONAL** (solo para audio en vivo): `sounddevice`.
  ```bash
  brew install portaudio
  venv/bin/pip install sounddevice
  ```

---

## 4. Audio del sistema en macOS — ⚠️ requiere loopback

macOS **NO** entrega el audio del sistema directamente a un programa. Para capturar lo que
suena en el iMac necesitas un **dispositivo virtual de loopback**:

```bash
brew install blackhole-2ch
```
Luego, en *Preferencias › Sonido* (o un Multi-Output Device en *Configuración de Audio MIDI*),
enruta la salida a **BlackHole 2ch**, y en el laboratorio elige **"BlackHole 2ch"** como
dispositivo de entrada. Sin BlackHole/Loopback, el modo "en vivo" solo capturará micrófono /
entrada de línea, **no** el audio del sistema. (La interfaz lo explica si no hay dispositivo.)

---

## 5. Entrada biaural — comportamiento (Req 1)

| Caso | Qué hace | Lateralidad real |
|---|---|---|
| **Un archivo estéreo** (p.ej. `BigBang_pos60deg.wav`) | usa sus **propios canales L/R** | **sí** (si L≠R) |
| **Dos archivos distintos** (A→L, B→R) | monoiza cada uno y enruta a un oído | sí, si A≠B |
| **Archivo mono** | duplica a L/R | no |
| **Demo** | mono, o duplicado si marcas binaural | no |
| **Biauralizar mono** | SIMULA L/R con delay (0.3 ms) + gain (R=0.95) | no (marcado **simulación**) |

Duraciones distintas (dos archivos): por defecto **truncado a la menor**; el criterio
queda explícito en el resumen y en el encabezado del CSV.

**Columnas nuevas en el CSV** (se conservan las 22 antiguas para compatibilidad):
`omega_L, omega_R, omega_A_L, omega_A_R, energia_L, energia_R, balance_LR, lateralidad,
coherencia_biaural` + observación (`LF_struct, self_coherencia, x_interna, en_rango,
mutacion, adaptacion_activa, exaptacion_activa, activacion_latente, invariantes_ok`).

> **Hallazgo honesto:** `BigBang_neg60deg` y `BigBang_pos60deg` tienen **mezcla mono
> idéntica** (corr 1.0). Por eso usarlos como *dos archivos* (uno por oído) NO produce
> lateralidad: la espacialización vive en los **canales internos** de cada archivo, y
> monoizar la destruye. Para lateralidad con esos archivos, **úsalos de a uno** (modo
> "mismo en ambos") → la célula recibe sus canales L/R reales (`lat_real=True`).

---

## 6. Ventanas de observación (Req 5)

Compacto (4): **Campo Φ**, **Consciencia**, **Libertad**, **Salud**.
Completo (+4): **Entrada biaural**, **Evolución**, **Homeostasis**, **Eventos** (timeline
textual: inicio de juego/ritual/negación, exaptación activa, C_m sobre umbral, H/Λ_Cos bajo
umbral, cambio de nivel LF — cada uno con t y valor).

---

## 7. Resumen de pruebas (Req 9) — `CelulaMadre_logs/biaural_*.csv`

| Caso | binaural | lat_real | lateralidad | coherencia | balance | OI | apagados |
|---|---|---|---|---|---|---|---|
| 01 demo mono dup L/R | sí | no | 0.199 | −0.488 | 0.0 | 0.537 | — |
| 02 BigBang neg→L, pos→R | sí | **no** | 0.500 | −0.536 | 0.0 | 0.299 | — |
| 03 BigBang pos→L, neg→R | sí | **no** | 0.500 | −0.536 | 0.0 | 0.299 | — |
| 04 **BigBang_pos estéreo (canales)** | sí | **sí** | 0.500 | −0.537 | −0.0004 | 0.299 | — |
| 05 **dos distintos (La/Mi)** | sí | **sí** | 0.090 | −0.397 | **−0.051** | 0.293 | — |
| 06 ablación **sin R2** | sí | sí | 0.500 | −0.537 | −0.0004 | **0.217** | meta_representacion |
| 07 ablación **sin LF** | sí | sí | 0.500 | −0.537 | −0.0004 | **0.217** | LF |

- **Binaural real verificado** (04 y 05: `lat_real=True`; 05 con `balance≠0` = energías L/R distintas).
- **Ablación real verificada**: apagar R2 o LF baja OI de 0.299 a 0.217 (sale del ciclo).
- **7) Audio en vivo: OMITIDO** — `sounddevice` no instalado en este entorno (no testeable aquí).

---

## 8. Limitaciones detectadas (honestas)

1. **Audio en vivo no probado aquí** (sin `sounddevice` ni dispositivos). El código está
   implementado y gateado con instrucciones; queda por validar en el iMac (+ BlackHole para sistema).
2. **Captura en vivo = "grabar N s y procesar"** (no streaming continuo de captura). Es una
   primera versión razonable; el streaming de los **gráficos** sí es en vivo.
3. **Ω sigue coarse** (~0.49): la diferenciación entre estímulos vive en R₂/C_m/XE/OI, no en Ω.
4. **Ritual** puede no dispararse (umbrales heredados de CM001; calibración pendiente).
5. **WAV float IEEE** (formato 3) no soportado por el cargador (`Brandemburgo.wav` falla);
   los `*_60deg` probados son PCM 16-bit y cargan bien.
6. **Dos archivos espacializados del mismo origen** no dan lateralidad al monoizar (ver §5).

---

## 9. Compatibilidad y trazabilidad (Req 7, 8)

- Interruptores por organelo = `expresar=False` → **fuera del ciclo metabólico** (ablación real,
  reflejada en el streaming y el CSV). El CSV registra los **apagados** en un encabezado comentado (`#`).
- CSV conserva las **22 columnas antiguas** (compatibilidad con análisis previos).
- Copias de seguridad en `_backups/` antes de modificar. Cambios comentados en el código
  (qué es REAL vs SIMULACIÓN/andamiaje).

---

## 9b. Dos entradas INDEPENDIENTES por oído (ampliación 2026-06-23)

Cada oído tiene su propio selector y se asigna por separado: **🟦 Entrada izquierda → hemisferio L**,
**🟥 Entrada derecha → hemisferio R**. Cada selector se llena desde **`GET /fuentes`** (lista unificada):

- **Cada CANAL de cada dispositivo de entrada** (`max_input_channels>0`), expuesto como
  `"{nombre} — canal n"` con `device_index` + `channel_index`. La **Rødecaster Pro II (16ch)**
  aparece como **16 fuentes** seleccionables.
- **Archivos** `.wav` de `audio_binaural/` + opción **subir** + **demos**.

Conexión al cuerpo: L y R pueden ser **dos dispositivos distintos, dos archivos, o uno de cada**.
Si la fuente es un canal de dispositivo, se graba ese device y se **extrae SOLO esa columna**
(`_extraer_canal`, no mezcla canales). Si ambos oídos son canales del **mismo** dispositivo, se usa
**un solo stream sincronizado** y se extraen las dos columnas (clave para la Rødecaster).

**Preservado:** `ω_B` y el gradiente audio-vs-referencia intactos; modelo de entrada previo
(fuente/left/right) sigue funcionando; **invariante mono idéntico** (el motor no se tocó);
ablación por organelo intacta. Backups en `_backups/`.

**Prueba (Req 5, `pruebas_dos_fuentes.py`):** sin Rødecaster en este entorno, se verificó:
(a) `/fuentes` enumera 117 archivos (0 dispositivos: sin `sounddevice`; con la Rødecaster saldrían
los 16 canales); (b) extracción de un canal de un **buffer 16ch sintético** (proxy Rødecaster) →
cada canal llega al cuerpo **por separado** (`ω_L≠ω_R`, `ω_B`/gradiente intactos); (c) por-oído con
dos archivos distintos → `lat_real=True`; (d) invariante mono idéntico. **Falta validar en el iMac
con la Rødecaster real** (instalar `sounddevice`).

Archivos de esta ampliación: `VST_CelulaMadre_WebLive.py` (sha1 `46c92291f170`),
`pruebas_dos_fuentes.py` (sha1 `328c51cbee2a`).

## 10. Cierre

La interfaz pasó de "ver el resultado al final" a **observar la fisiología de la célula en
vivo, por organelo, con entrada biaural real** y selección directa de audios del proyecto.
El audio en vivo está listo para activarse en el iMac (instalar `sounddevice`; BlackHole para
audio del sistema). Todo lo testeable aquí quedó verificado; lo no testeable está claramente
marcado.
