# ¿El órgano de radio caza ESTRUCTURA o se posa donde sea? — batería de falsación

Percepción PURA (sin SNR). Arms: REAL / NULL (ruido plano, misma energía) / SHUFFLED (bins
permutados: misma energía, sin estructura espacial). Órgano fresco por espectro. seed=42.

## Fuente A (RSPduo/WS — comprimido)

| arm | n | tasa enganche | estructura media | saliencia media |
|---|---|---|---|---|
| REAL | 20 | 0% | 0.036 | 0.018 |
| NULL | 20 | 0% | 0.015 | 0.007 |
| SHUFFLED | 20 | 0% | 0.018 | 0.009 |

→ **A NO discrimina** claramente (REAL−NULL=0.022, REAL−SHUFFLED=0.018): sobre este canal el
  órgano se posa parecido con o sin estructura. Duda CONFIRMADA para esta fuente.

## Fuente E (RSP1/SoapySDR — dinámica real)

| arm | n | tasa enganche | estructura media | saliencia media |
|---|---|---|---|---|
| REAL | 20 | 80% | 0.138 | 0.069 |
| NULL | 20 | 0% | 0.014 | 0.007 |
| SHUFFLED | 20 | 20% | 0.064 | 0.032 |

→ **E DISCRIMINA estructura**: REAL supera a NULL (+0.124) y a SHUFFLED (+0.074) en estructura.
  (misma energía en los tres; sólo cambia la estructura espacial → percibe forma, no potencia).

## SYNTH-conflicto: meseta de POTENCIA vs pico de ESTRUCTURA

- Meseta (bins 40–140, val 0.65): banda ~2 · potencia media 0.65 · estructura BAJA (plana)
- Pico (bins 300–308, val 0.98): banda ~9 · potencia media 0.36 · estructura ALTA (peaky)
- El órgano eligió banda **9.0** (estructura=0.415, saliencia=0.207, enganchó=True)
- ✅ **Fue a la ESTRUCTURA (pico), NO a la potencia (meseta)** — anti-Shannon confirmado.

## Veredicto global (¿resuelve las dudas?)
- **Duda #3 (¿estructura o cualquier pico?):** E=discrimina, A=no discrimina. SYNTH→estructura.
- **Duda #1 (muleta SNR de A):** sin SNR, E=discrimina y A=no discrimina → si E discrimina y A no, la muleta de A es su CANAL comprimido, no el órgano.
