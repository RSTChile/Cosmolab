# ¿El organismo BARRE el espectro hasta SELECCIONAR una emisión? — 2026-07-05 14:06

Órgano de radio con **actuador ACTIVO** (sintonia_activa=True), política real = argmax de
SALIENCIA (estructura+novedad), sobre el RSPduo. Banda 88.0–108.0 MHz, paso 1.6 MHz.

Lector vivo. Espectro fluyendo. Comienzo el BARRIDO.

## 1) Barrido del espectro (mapa: qué encuentra en cada ventana)
| centro (MHz) | freq dom (MHz) | estructura | saliencia | potencia | bins señal |
|---|---|---|---|---|---|
| 88.0 | 87.06 | 0.073 | 0.044 | 0.878 | 475 |
| 89.6 | 88.66 | 0.107 | 0.059 | 0.906 | 472 |
| 91.2 | 90.26 | 0.091 | 0.049 | 0.883 | 471 |
| 92.8 | 91.86 | 0.110 | 0.059 | 0.890 | 473 |
| 94.4 | 93.46 | 0.110 | 0.057 | 0.837 | 472 |
| 96.0 | 96.94 | 0.081 | 0.059 | 0.711 | 475 |
| 97.6 | 96.66 | 0.150 | 0.081 | 0.864 | 475 |
| 99.2 | 98.26 | 0.093 | 0.052 | 0.892 | 477 |
| 100.8 | 99.86 | 0.111 | 0.065 | 0.852 | 474 |
| 102.4 | 101.46 | 0.095 | 0.050 | 0.833 | 482 |
| 104.0 | 103.06 | 0.084 | 0.047 | 0.864 | 489 |
| 105.6 | 106.54 | 0.019 | 0.019 | 0.842 | 500 |
| 107.2 | 106.26 | 0.078 | 0.047 | 0.862 | 483 |

## 2) Selección de emisión
- **Elegida por el órgano (saliencia=0.081, estructura=0.150)** → ventana 97.6 MHz, emisión en **96.66 MHz**
- Habría ganado por POTENCIA cruda → ventana 89.6 MHz (pot=0.906, saliencia sólo 0.059)
- ⇒ **el órgano NO elige por energía**: selecciona ESTRUCTURA (una portadora/emisión real), no el pico de potencia.

## 3) Enganche — sintonizo 96.66 MHz y sostengo 45 s (¿se queda clavado?)
| t (s) | freq dom (MHz) | Δ vs objetivo (kHz) | estructura | saliencia |
|---|---|---|---|---|
|    0 | 106.262 | +9600 | 0.076 | 0.047 |
|    5 | 104.387 | +7725 | 0.072 | 0.051 |
|   10 | 102.512 | +5850 | 0.088 | 0.052 |
|   15 | 96.887 | +225 | 0.111 | 0.066 |
|   20 | 95.012 | -1650 | 0.165 | 0.110 |
|   25 | 85.637 | -11025 | 0.081 | 0.068 |
|   30 | 76.262 | -20400 | 0.051 | 0.030 |
|   35 | 74.387 | -22275 | 0.040 | 0.029 |
|   40 | 74.387 | -22275 | 0.035 | 0.025 |

## Veredicto
- Dispersión de freq_dom durante el enganche: **33.750 MHz** (90 muestras).
- ⚠️ freq_dom sigue moviéndose (disp 33.75 MHz): o la ventana tiene varias emisiones que compiten, o no hay una portadora dominante clara.

**Lectura:** Fase 1 = el organismo BARRE (mueve el LO por la banda). Fase 2 = SELECCIONA por
estructura, no por potencia (agencia perceptiva, no medidor de energía). Fase 3 = se ENGANCHA
y sostiene. Eso es 'oír activo': percibir haciendo algo, el linaje del oído pasivo → buscar.
