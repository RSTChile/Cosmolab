# Barrer → seleccionar → ENGANCHAR con histéresis (RSPduo) — 2026-07-05 14:14

## 1) Barrido (mapa por ventana)
| centro | freq dom | estructura | saliencia | potencia |
|---|---|---|---|---|
| 88.0 | 87.06 | 0.114 | 0.060 | 0.848 |
| 89.6 | 88.66 | 0.074 | 0.052 | 0.914 |
| 91.2 | 90.26 | 0.140 | 0.083 | 0.886 |
| 92.8 | 91.86 | 0.094 | 0.051 | 0.880 |
| 94.4 | 93.46 | 0.137 | 0.081 | 0.825 |
| 96.0 | 96.94 | 0.076 | 0.049 | 0.716 |
| 97.6 | 96.66 | 0.159 | 0.087 | 0.856 |
| 99.2 | 98.26 | 0.110 | 0.062 | 0.900 |
| 100.8 | 99.86 | 0.099 | 0.055 | 0.844 |
| 102.4 | 101.46 | 0.080 | 0.045 | 0.839 |
| 104.0 | 103.06 | 0.075 | 0.040 | 0.854 |
| 105.6 | 104.66 | 0.011 | 0.020 | 0.851 |
| 107.2 | 106.26 | 0.083 | 0.062 | 0.854 |

## 2) Selección → emisión con más ESTRUCTURA: **96.66 MHz** (est=0.159, sal=0.087)

## 3) Enganche con histéresis — arranco DESINTONIZADO en 96.36 MHz (objetivo 96.66)
| t(s) | centro (MHz) | freq dom (MHz) | saliencia | acción |
|---|---|---|---|---|
|    0 | 96.362 | 97.300 | 0.093 | candidata(1/3) |
|    4 | 97.300 | 96.362 | 0.036 | sostener |
|    8 | 97.300 | 98.237 | 0.026 | sostener |
|   12 | 97.300 | 98.237 | 0.035 | sostener |
|   16 | 97.300 | 98.237 | 0.032 | sostener |
|   20 | 97.300 | 98.237 | 0.028 | sostener |
|   24 | 97.300 | 98.237 | 0.025 | sostener |
|   28 | 97.300 | 96.362 | 0.030 | sostener |
|   32 | 97.300 | 98.237 | 0.039 | sostener |
|   36 | 97.300 | 98.237 | 0.028 | sostener |
|   40 | 97.300 | 98.237 | 0.029 | sostener |
|   44 | 97.300 | 96.362 | 0.031 | sostener |

## Veredicto
- Centro final: **97.300 MHz** (objetivo 96.66; error 638 kHz).
- Dispersión del centro en el sostén: **0.938 MHz** (96 muestras).
- ⚠️ Aún deriva (disp 0.94): el espectro normalizado-a-visible es demasiado plano; haría falta usar la telemetría signal_snr (dinámica real) en vez de los bins comprimidos.

**Lectura:** con histéresis el actuador deja de perseguir ruido: adquiere la emisión y la
SOSTIENE. Es el paso que faltaba entre 'barrer/seleccionar' (que ya funcionaba) y 'engancharse'.
