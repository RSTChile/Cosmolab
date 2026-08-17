# Traspaso a CC — Sentido de radio en el organismo A (Mac + RSPduo)

**De:** Claude Science, con Alexis · **Fecha:** 3-jul-2026
**Estado:** SOFTWARE listo y verificado (compila, degrada elegante). Falta SOLO la pieza de HARDWARE (tuya).

## Lo que ya quedó hecho (software, sin tocar hardware)
- `organelos/VST_IntegracionRadioA.py` — integración mínima SOLO-radio para A. Mismo contrato que
  IntegracionE (arrancar_lectores/observar_paso/cerrar/columnas_csv/organelos_persistibles/.activo),
  pero sin ATmega/GPS/cámara. Lector ENCHUFABLE (parámetro `lector=`).
- `web/VST_CelulaMadre_WebLive_A.py` — injertado en 5 puntos, TODO en paralelo a la ruta de E, sin tocarla:
  import, instancia, persistencia, observar_paso (loop), arranque/cierre de lectores.
- Activación explícita: **`ANIMA_RADIO_A=1`**. Por defecto (sin esa env) A queda IDÉNTICO a antes.
- Verificado: A/B/C/D compilan; con ANIMA_RADIO_A=1 pero sin backend SDR, A NO revienta — degrada
  (sdr_vivo=0, radio_vivo=0.0), la caja `radio_sdr.js` queda latente en "—".
- La caja web ya existe y espera: sdr_espectro, sdr_vivo, radio_vivo, radio_freq_dom_hz, radio_saliencia,
  radio_estructura, radio_novedad, radio_potencia_total, radio_n_bandas.

## Lo que falta — y es TUYO (necesita el hardware en mano; CS no tiene acceso al USB/SDR)
En la Mac NO hay SoapySDR, así que el `LectorSDR` actual (backend SoapySDR/sdrplay directo) no abre el RSPduo.
Alexis decidió ir por el **servidor SDRconnect** (dejarlo corriendo al inicio). Falta el lector-cliente:

1. **Levantar el servidor** (Alexis ya probó que responde). Comando base con su RSPduo:
   ```
   cd /Applications/SDRconnect.app/Contents/MacOS
   ./SDRconnect --server --hwser=2007054F32 --port=50000 --samplerate=2000000 \
     --centerfrequency=100000000 --antenna=0 --ifagc=1 --lnastate=0 --ifgr=59
   ```
   (antena 0 = Tuner 1 / 50 Ω. Verificar el protocolo real del servidor: SDRconnect Networking PDF.)

2. **Escribir un LectorSDRServidor** (cliente del servidor) con la MISMA interfaz que LectorSDR:
   `arrancar()->bool`, `cerrar()`, `sintonizar(freq_hz)`, `inyectar(fila)` — donde inyectar pone en la fila:
   `sdr_espectro` (lista de potencias por bin), `sdr_freq_min_hz`, `sdr_freq_max_hz`, `sdr_vivo`.
   Ese es el ÚNICO contrato. El órgano y la caja ya consumen esas columnas.

3. **Enchufarlo** sin tocar nada más:
   ```python
   IntegracionRadioA(ORGANISMO_ID, lector=LectorSDRServidor(host="127.0.0.1", port=50000))
   ```
   O registrar la fábrica y que IntegracionRadioA la tome por env. La instancia ya prioriza `lector=`.

## Prueba de aceptación (cuando el lector-servidor entregue datos)
- `ANIMA_RADIO_A=1` + servidor arriba → `/ultima_fila` de A debe traer sdr_vivo=1 y radio_vivo=1.0.
- Con FM sintonizada (Alexis lo tenía en 94.5 y 100 MHz): radio_freq_dom_hz cae en una banda con estructura,
  NO en la de más potencia (test anti-Shannon del órgano — ya pasa en simulación).
- Abrir la página de A → activar la caja "Radio / SDR" → debe pintar el espectro y las gauges.
- B/C/D con ANIMA_RADIO_A sin setear → siguen idénticos, caja latente.

## Sintonía fija en 94.5 MHz (campo, sin estaciones que barrer)
Alexis vive en el campo: hay ~1 emisora. E NO debe arrancar en modo barrido o se "vuelve loco" saltando por
ruido. Dos niveles, los dos fijados a 94.5:
- **Servidor** centrado en 94.5 MHz: `--centerfrequency=94500000` (ver mac_sdr_server/, autoarranque abajo).
- **Órgano** en modo observar, NO explorar:
  ```
  ANIMA_RADIO_A=1            # activa el sentido de radio en A
  ANIMA_RADIO_SINTONIA=0     # 0 = escucha quieto donde lo centra el servidor (NO re-sintoniza)
  ```
  Con SINTONIA=0 el órgano observa 94.5 sin barrer. Cuando ya funcione y se le quiera dar curiosidad, subir a 1.

## Autoarranque del servidor SDR con la Mac
Carpeta `Célula_Madre/mac_sdr_server/` (hecha por CS):
- `sdr_server_94500.sh` — el comando, centrado en 94.5, ejecutable.
- `cl.cosmolab.sdrserver.plist` — LaunchAgent (RunAtLoad + KeepAlive, reinicia si cae).
- `README.md` — instalar con `cp … ~/Library/LaunchAgents/ && launchctl load …`; logs en ~/Library/Logs/.
- OJO conflicto USB: la GUI de SDRconnect abre el equipo en EXCLUSIVO. Debe estar CERRADA cuando el servidor
  arranca, o el KeepAlive entra en ciclo de reintento (visible en sdr_server.err.log).

## Nota de frontera
CS escribió y probó todo lo que no toca el SDR (degradación con sandbox sin SoapySDR = misma situación que
la Mac). Abrir el RSPduo / hablar con el servidor SDRconnect / cargar el LaunchAgent es hardware y sistema:
va aquí, contigo. CS no puede ejecutar launchctl ni abrir el USB.
