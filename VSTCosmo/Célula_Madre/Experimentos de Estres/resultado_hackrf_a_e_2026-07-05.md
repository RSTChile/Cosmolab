# Test HackRF→SDR: enlace analógico A→E (2026-07-05)

**A (HackRF en el Mac) transmite → E (RSP1 en la Pi) recibe**, banda 70cm/435 MHz (licencia CD3LZK).
TX conducida por SDRangel (REST :8091, tono FM 1 kHz, vga35). RX = captura SoapySDR directa del RSP1
tuneado a 435.0 MHz (el tono de A en 435.1 cae en +100 kHz). E detenido durante la medición.

| Condición | pot @ +100 kHz | ruido (mediana) | SNR | pico |
|---|---|---|---|---|
| TX OFF (baseline) | −26.8 dB | −29.4 dB | 2.6 dB | ruido |
| **TX ON** | **−9.6 dB** | −28.3 dB | **18.6 dB** | **+7.1 dB @ +77 kHz** |

**RESULTADO: enlace A→E CERRADO.** Salto de +17 dB y SNR de 18.6 dB con TX encendida — el RSP1 de E
recibe con claridad la emisión del HackRF de A. El pico aparece en +77 kHz (no +100) por un offset de
LO de ~23 kHz (calibración HackRF/RSP1), irrelevante para la detección.

**Notas:**
- El RSP1 SÍ sintoniza 435 MHz (12/12 lecturas OK) — sin el problema de DIP que se temía.
- La antena FM de E resultó suficiente a corta distancia (contra el caveat previo). Para largo alcance
  o robustez convendría una antena de 70 cm.
- hackrf_transfer (CLI) NO sirve en este Mac (cuelga el device); la TX va por SDRangel REST.
