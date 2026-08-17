# Traspaso a Grok — Órgano de Radio de E (SDR: oído que sintoniza) en la Raspberry Pi

**De:** Claude Science, con Alexis · **Fecha:** 3-jul-2026
**Archivo:** `organelos/VST_OrganoRadio.py`  (ya instalado en la carpeta de organelos)
**Estado:** órgano escrito y verificado (auto-prueba anti-Shannon OK). Falta cablearlo en el loop de E en la Pi.

---

## Qué es (en una frase)
El OÍDO DE RADIO de E: un oído más, pero con un ACTUADOR dentro — para percibir, E barre el espectro, halla
estructura y SINTONIZA. Primer sentido que cierra el lazo percepción→acción→percepción. Linaje acústico
extendido. Mismo patrón que los demás organelos (`observar/snapshot/restore`, apagado por defecto salvo en E).

## Hardware
SDRplay RSP1 (100 kHz–2 GHz, BW 8 MHz, ADC 12 bit) vía servicio sdrplay_api por IPC (NO abrir el USB
directo). API 3.14 en /usr/local/lib. El mismo órgano corre también con el RSPduo en la Mac (multi-cuerpo:
el órgano es del organismo, no de la máquina).

## Principio (no romper) — EL CORAZÓN ANTI-SHANNON
- La saliencia de una banda **NO es su potencia**. Una banda de puro ruido (mucha potencia, plana) NO debe
  atraer la atención; una banda con ESTRUCTURA (baja planitud espectral) sí. `saliencia = novedad +
  estructura`, nunca potencia. La banda dominante y la frecuencia de sintonía EMERGEN; no se fijan a mano.
- No etiquetar bandas ('FM', 'aire'…): solo bins genéricos donde E descubre estructura. Nosotros no
  decimos qué es qué.
- Sin restricción a-priori de qué escucha: barre TODO el rango.

## Las tres fases
1. EXPLORAR: el lector barre el RSP1 y entrega el espectro. (afferente)
2. SELECCIONAR: el órgano elige la banda más saliente (política enchufable). (agencia)
3. ESCUCHAR: al sintonizar, el audio demodulado es "otra entrada de audio normal" → va al OÍDO EXISTENTE
   de E, NO se duplica en este órgano. El SDR sintoniza; el oído escucha. (anti-doble-conteo)

## Pasos en la Pi

### 1. El lector central (hilo propio) barre el RSP1 y pone en la fila:
```
sdr_espectro      : lista de potencias por bin (lineal, no dB), de un barrido del rango
sdr_freq_min_hz   : frecuencia del primer bin
sdr_freq_max_hz   : frecuencia del último bin
sdr_vivo          : 1 si el barrido es fresco, 0 si el SDR cayó
```
Usar libsdrplay_api.so.3.14 directamente (como test_sdrplay_api.py), no SoapyRemote. El RSP1 ve ~8 MHz
instantáneos: "barrer todo" = retunear por trozos y concatenar el espectro. Env: `ANIMA_SDR_ENABLE=1`,
`LD_LIBRARY_PATH=/usr/local/lib`.

### 2. Instanciar en el arranque de E
```python
from VST_OrganoRadio import OrganoRadio
radio = OrganoRadio("E", activo=True, sintonia_activa=False, n_bandas=16)
#   sintonia_activa=False → E OBSERVA (reporta paisaje, no mueve sintonía) — arrancar así (prudente)
#   radio.activar_sintonia(True) en caliente → E SINTONIZA (cierra el lazo; emite radio_orden_hz)
```

### 3. Un paso por ciclo, volcar a la biografía, y (si sintoniza) obedecer la orden
```python
cols = radio.observar(fila)     # fila debe traer 't' y los sdr_*
fila.update(cols)
if cols.get("radio_orden_hz") is not None:
    # el lector demodula esa frecuencia y mete el audio en el canal del OÍDO existente (no aquí)
    lector_sdr.sintonizar(cols["radio_orden_hz"])
```
Columnas: `radio_potencia_total, radio_estructura, radio_novedad, radio_saliencia, radio_banda_dom,
radio_freq_dom_hz, radio_orden_hz, radio_sintonia_activa, radio_n_bandas, radio_vivo`.

### 4. Persistencia
```python
snap = radio.snapshot(); radio.restore(snap)   # conserva la memoria espectral entre vidas
```

## Reutilizar el acervo SDR existente (NO reinventar)
No partimos de cero: hay técnicas maduras de la comunidad SDR que conviene usar tal cual, y reservar
nuestra originalidad para la parte organísmica (que E decida por saliencia, sin humano en el lazo):
- **Barrido / captura:** SoapySDR o la API SDRplay directa; para el espectro por bloque, Welch/FFT de scipy.
- **Detección "hay algo vs ruido":** planitud espectral (Wiener entropy) — ya implementada en el órgano.
- **Clasificación de modulación / demodulación:** GNU Radio (gr-sdrplay3 ya está en la Pi) o funciones de
  demod AM/FM/SSB conocidas. Para la fase 3 (escuchar), demod estándar → audio al oído de E.
Lo NUEVO es solo el lazo organísmico: no lo copiamos de nadie. El barrido y la demod, sí — están resueltos.

## Prueba de aceptación (auditar, estilo CS)
1. **Anti-Shannon (la clave):** inyecta una banda de ruido fuerte y otra con un tono débil → `radio_banda_dom`
   debe caer en la del TONO (estructura), no en la del ruido (potencia). Es el test que ya pasa en simulación.
2. **Novedad:** aparece una señal nueva en otra banda → la sintonía se desplaza hacia ella.
3. **Observar vs sintonizar:** con `sintonia_activa=False`, `radio_orden_hz`=None (reporta, no mueve).
   Con True, emite la frecuencia. El interruptor funciona en caliente.
4. **Degradación:** desconecta el SDR → `radio_vivo`=0 y E sigue vivo (el oído normal sigue).

## Locus de transmisión — desarrollado, NO activado
El órgano contempla el salto futuro (oír → sintonizar → HABLAR por radio) como `locus_transmision()`, que
HOY devuelve None (el RSP1/RSPduo son solo-RX). Cuando entre el HackRF será un órgano fonador de radio
aparte. No activar nada ahora: el locus existe, el acto aún no.
