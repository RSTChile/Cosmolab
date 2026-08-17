# Caza de emisión con el actuador nuevo (histéresis + SNR) — 2026-07-05 14:24

El órgano barre SOLO (el test no lo conduce). histeresis=True, usa signal_snr real.

| t(s) | centro (MHz) | SNR dB | saliencia | enganchado | acción |
|---|---|---|---|---|---|
|    0 | 93.10 | 1.7 | 0.063 | no | barre→93.5 |
|    5 | 94.30 | 0.9 | 0.059 | no | barre→94.7 |
|   10 | 95.50 | -0.1 | 0.082 | no | — |
|   15 | 96.30 | 8.9 | 0.055 | SÍ | SOSTIENE |
|   20 | 96.70 | 5.9 | 0.044 | no | barre→97.1 |
|   25 | 96.16 | 9.1 | 0.053 | no | — |
|   30 | 97.36 | 0.2 | 0.071 | no | — |
|   36 | 98.56 | 0.0 | 0.095 | no | — |
|   40 | 99.36 | 1.3 | 0.075 | no | barre→99.8 |
|   45 | 100.56 | 0.4 | 0.072 | no | — |
|   50 | 101.76 | 0.1 | 0.087 | no | — |
|   55 | 102.96 | -0.1 | 0.084 | no | — |
|   60 | 103.76 | 6.9 | 0.049 | no | barre→104.2 |
|   65 | 104.96 | -0.1 | 0.066 | no | — |
|   71 | 106.16 | 1.8 | 0.058 | no | — |
|   75 | 107.36 | 1.0 | 0.036 | no | — |
|   80 | 88.00 | 1.8 | 0.045 | no | barre→88.4 |
|   86 | 89.20 | 1.1 | 0.086 | no | barre→89.6 |
|   90 | 90.40 | 1.3 | 0.068 | no | — |
|   95 | 91.60 | 0.9 | 0.060 | no | — |
|  100 | 91.06 | 0.8 | 0.084 | no | — |
|  105 | 90.92 | 0.8 | 0.075 | no | — |
|  110 | 91.72 | 1.3 | 0.064 | no | barre→92.1 |
|  115 | 92.92 | 0.5 | 0.100 | no | — |
|  121 | 94.12 | 0.5 | 0.104 | no | — |
|  125 | 95.33 | 0.2 | 0.105 | no | — |
|  130 | 94.79 | 0.5 | 0.067 | no | — |
|  135 | 94.65 | 0.8 | 0.073 | no | — |
|  140 | 95.45 | 0.1 | 0.063 | no | barre→95.8 |
|  145 | 96.25 | 16.3 | 0.074 | SÍ | SOSTIENE |
|  150 | 96.25 | 16.0 | 0.099 | SÍ | SOSTIENE |
|  156 | 96.25 | 16.5 | 0.055 | SÍ | SOSTIENE |
|  160 | 96.25 | 15.4 | 0.059 | SÍ | SOSTIENE |
|  165 | 96.25 | 15.0 | 0.053 | SÍ | SOSTIENE |
|  171 | 96.25 | 15.9 | 0.061 | SÍ | SOSTIENE |
|  175 | 96.25 | 16.3 | 0.073 | SÍ | SOSTIENE |

## Veredicto
- ✅ **ENGANCHÓ** una emisión tras barrer: primera vez a los **13 s**, ~96.3 MHz.
- SNR medio mientras enganchado: **13.9 dB** (75 muestras) → señal real, no ruido.
- Saltos de barrido antes/entre enganches: ~74.

**Lectura:** el órgano, solo, BARRIÓ el espectro, ADQUIRIÓ una emisión (SNR alto) y la
SOSTUVO — la conducta de 'oír activo' (barrer hasta seleccionar) ahora EMERGE del órgano.
