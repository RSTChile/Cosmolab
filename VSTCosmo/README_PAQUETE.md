# PAQUETE PARA EL EQUIPO — Batería de tests de la Célula Madre Funcional

**Fecha:** 2026-06-23 · **Autoría:** Alexis López Tapia + equipo transinteligente RMD 2.0

Este paquete es **autocontenido para LEER y AUDITAR**. Para **re-correr** la batería hace
falta, además, la carpeta `audio_binaural/` (2.4 GB, se entrega aparte) y el `venv`.

---

## Por dónde empezar
1. **`INFORME_BATERIA_Test.md`** — el informe con los hallazgos.
2. **`bateria_test.py`** — el script que los generó (6 suites).
3. **`CelulaMadre_logs/bateria_*.csv`** + `bateria_resumen_20260623.json` — los datos crudos.

## Contenido y SHA1 (verificable)
```
INFORME (principal)
  INFORME_BATERIA_Test.md                 sha1 65b38cf23980
SCRIPT batería
  bateria_test.py                         sha1 fd1fb919a194
CÓDIGO bajo prueba
  Célula_Madre_Funcional_001.py           sha1 da7fb7aae404   (motor + cargador WAV universal)
  VST_CelulaMadre_WebLive.py              sha1 46c92291f170   (laboratorio en vivo / dos fuentes)
MOTOR / genoma (dependencias)
  VST_Genoma.py                           sha1 0b890c6d3b1e
  VST_Bloque05_ConscienciaFuncional.py    sha1 e703bf7fabb7
  VST_Bloque07_LibertadFuncional.py       sha1 d9444fd6d7be
  VST_Bloque08_DinamicaEvolutiva.py       sha1 788bc5968ea9
  VST_Homeostasis.py                      sha1 83a5745df822
  VST_Celula_Madre_001.py                 sha1 21eee09ddb7e   (campo Φ / Hemisferio)
INFORMES de contexto
  INFORME_CELULA_MADRE_Cosmosemiotica.md  sha1 db5d716d622d   (arquitectura completa)
  INFORME_LIVE_CelulaMadre.md             sha1 2ddfb43e458f   (laboratorio en vivo)
DATOS
  CelulaMadre_logs/bateria_resumen_20260623.json
  CelulaMadre_logs/bateria_*.csv          (17 corridas: B×5, C×4, D×2, E×6)
```
(Verificar: `shasum -a 1 <archivo>`.)

## Reproducir la batería
Requisitos:
- Python 3.13 en un `venv` con `numpy` y **`soundfile`** (carga universal de WAV);
  opcional `sounddevice` (audio en vivo). Instalar: `pip install soundfile sounddevice`.
- La carpeta `audio_binaural/` con los `.wav` (NO incluida aquí por tamaño, 2.4 GB).

Ejecutar:
```
venv/bin/python3 bateria_test.py
```
Es **determinista** (semillas fijas): deben obtener los CSV **idénticos bit a bit**. Si difieren,
hay algo distinto en el entorno (versión de numpy/soundfile, audio, etc.).

## Las 6 suites (resumen)
- **A** Cobertura de formatos (PCM 8/16/24/32 + float IEEE + extensible).
- **B** Estructura vs energía (estímulos a igual RMS).
- **C** Binaural vs mono (mismo archivo).
- **D** Asimetría L/R (swap de oídos).
- **E** Anatomía funcional (ablación por organelo).
- **F** Determinismo.

## Nota honesta clave (leer en el informe)
La lateralidad llega al campo pero **no se puede juzgar su expresión conductual: el organismo
AÚN NO TIENE ACTUADORES** (falta el motor de orientación a la fuente de los experimentos
iniciales V117–V122). Es una **pregunta abierta**, no un hallazgo negativo.
