# Servidor SDR autoarranque (RSPduo en la Mac) — 94.5 MHz

Arranca el servidor SDRconnect solo, al iniciar sesión en la Mac, centrado en 94.5 MHz
(la FM del campo — así E no barre el vacío buscando estaciones que no hay).

## Archivos
- `sdr_server_94500.sh`         — el comando (ya ejecutable). Centro 94.5 MHz, puerto 50000.
- `cl.cosmolab.sdrserver.plist` — el LaunchAgent que lo dispara al login.

## Instalar (una sola vez) — correr en Terminal
    cp "/Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre/mac_sdr_server/cl.cosmolab.sdrserver.plist" ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/cl.cosmolab.sdrserver.plist

Eso lo arranca AHORA y en cada login. (launchctl moderno: `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/cl.cosmolab.sdrserver.plist`)

## Comandos útiles
- Ver si corre:     launchctl list | grep sdrserver
- Ver el log:       tail -f ~/Library/Logs/sdr_server.log
- Ver errores:      tail -f ~/Library/Logs/sdr_server.err.log
- Parar/quitar:     launchctl unload ~/Library/LaunchAgents/cl.cosmolab.sdrserver.plist
- Reiniciar:        launchctl unload ... && launchctl load ...   (los dos con la ruta del plist)

## IMPORTANTE — el conflicto del USB
SDRconnect **GUI** abre el RSPduo en modo EXCLUSIVO. Si la app gráfica está abierta, el
servidor no verá el equipo (solo "IQ File"). Reglas:
- Para que el autoarranque funcione, la GUI NO debe estar abierta cuando el servidor arranca.
- Al iniciar sesión limpio la GUI no está abierta → el servidor toma el equipo. OK.
- Si abres la GUI a mano después, MÁTALA antes de que el servidor la necesite, o el KeepAlive
  entrará en un ciclo de reintento cada 15 s (visible en el .err.log). En ese caso: cierra la GUI.
- Quita SDRconnect de "Abrir al iniciar sesión" (Ajustes ▸ General ▸ Ítems de inicio) si estaba,
  para que no pelee con el servidor.

## Cambiar de emisora
Edita `--centerfrequency` en `sdr_server_94500.sh` (en Hz: 94.5 MHz = 94500000) y reinicia el agente.

## Nota
El KeepAlive reinicia el servidor si cae — bueno para que E nunca se quede sin oído. El precio es
el ciclo de reintento si el USB está tomado; por eso la regla de la GUI de arriba.
