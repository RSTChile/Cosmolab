# ÍNDICE para CC — sentido de radio en A + cajas de sentidos (3-jul-2026)

Todo verificado en disco. Orden de lectura: primero las notas, luego el código.
Rutas absolutas (base = /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre).

## 1. LEER PRIMERO — las dos notas de traspaso
    traspasos/traspaso_radioA_CC.md          ← qué hacer, qué falta (lector-cliente del servidor SDR), env vars
    traspasos/traspaso_cajas_columnas_CC.md  ← qué columnas debe traer /ultima_fila para que pinten las cajas

## 2. CÓDIGO — órgano + integración (lo que corre en A / la Pi)
    organelos/VST_OrganoRadio.py         ← el órgano (ya listo; consume sdr_*, produce radio_*)
    organelos/VST_IntegracionRadioA.py   ← integración SOLO-radio para A (lector enchufable). NUEVO
    organelos/VST_LectorSensores.py      ← LectorSDR actual (backend SoapySDR directo). AQUÍ va el lector-servidor
    web/VST_CelulaMadre_WebLive_A.py     ← injertado en 5 puntos (activar con ANIMA_RADIO_A=1)

## 3. FRONT — cajas del observatorio
    web/Cajas/radio_sdr.js            ← con waterfall (canvas). NUEVO
    web/Cajas/localizacion.js         ← reescrita (sentido loc_* + crudo gps_*). NUEVA versión
    web/Cajas/vision.js               ← reescrita (saliencia/novedad + retina). NUEVA versión
    web/Cajas/cloroplasto_fisico.js   ← ya calzaba, sin cambios
    web/Cajas/manifest.json           ← las 4 registradas

## 4. SERVIDOR SDR EN LA MAC — autoarranque (94.5 MHz)
    mac_sdr_server/sdr_server_94500.sh         ← el comando (centro 94.5 MHz, puerto 50000)
    mac_sdr_server/cl.cosmolab.sdrserver.plist ← LaunchAgent (RunAtLoad + KeepAlive)
    mac_sdr_server/README.md                   ← instalar (launchctl load) + regla del conflicto USB con la GUI

## LO QUE FALTA (tuyo, es hardware/sistema)
1. Levantar el servidor SDR (sección 4) — o instalar el autoarranque. GUI SDRconnect CERRADA.
2. Escribir el lector-cliente del servidor (puerto 50000) con la interfaz de LectorSDR:
   arrancar()->bool · cerrar() · sintonizar(hz) · inyectar(fila){sdr_espectro,sdr_freq_min_hz,sdr_freq_max_hz,sdr_vivo}
3. Enchufarlo:  IntegracionRadioA(ORGANISMO_ID, lector=LectorSDRServidor(host="127.0.0.1", port=50000))
4. Arrancar A con:  ANIMA_RADIO_A=1  ANIMA_RADIO_SINTONIA=0   (escucha quieto 94.5, no barre)
5. Prueba: /ultima_fila de A → sdr_vivo=1, radio_vivo=1.0; abrir caja "Radio / SDR" → waterfall pinta.
