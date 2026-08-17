# Informe técnico — SDRplay RSP1 en Raspberry Pi (Organismo E)

**Fecha:** 2026-07-03  
**Equipo:** Organismo E - Planta / Célula Madre  
**Estado:** hardware visible por USB, API operativa (`GetDevices` = 1 dispositivo)  
**Contexto ANIMA:** linaje acústico extendido (SDR) — futuro órgano sensorial, no humano

---

## Resumen ejecutivo

La Pi hospeda un **SDRplay RSP1** conectado por USB en el mismo hub que la ATmega del cloroplasto. Tras corregir la contención con `soapyremote-server`, el dispositivo responde correctamente vía **SDRplay API 3.14** (`sdrplay_api_GetDevices` → 1 RSP, `hwVer=1`, `valid=1`).

Para el organismo E, el SDR es candidato a **oído extendido** (leer ondas de radio como magnitudes crudas; semántica en E, no en el driver — anti-Shannon).

---

## 1. Hardware — SDRplay RSP1

| Parámetro | Valor |
|-----------|--------|
| Modelo | SDRplay **RSP1** (1ª generación) |
| ID API | `hwVer=1` → `SDRPLAY_RSP1_ID` |
| USB Vendor/Product | `1df7:2500` |
| Número de serie (USB) | `0000000001` |
| USB | 2.0 High-Speed, bus-powered (**400 mA**) |
| Host | Raspberry Pi, **aarch64**, Ubuntu 22.04.4 LTS |
| Conexión observada | Bus 001, puerto hub (p. ej. `1-1.3`) |

### Especificaciones RF (datasheet SDRplay RSP1)

| Parámetro | Especificación |
|-----------|----------------|
| Rango de frecuencia | **100 kHz – 2 GHz** |
| Ancho de banda máximo | **8 MHz** |
| ADC | **12 bits** |
| Cadena de ganancia | Procesamiento interno hasta **14 bits** |
| Conector antena | SMA |
| Uso previsto en ANIMA | Extensión del linaje **acústico/vibración** a bandas de radio (SDR) |

---

## 2. Coexistencia USB en el banco de E

| Dispositivo | USB ID | Interfaz / servicio |
|-------------|--------|---------------------|
| SDRplay RSP1 | `1df7:2500` | `sdrplay_apiService` (daemon) |
| ATmega 2560 Pro CH340 (cloroplasto) | `1a86:7523` | `/dev/ttyUSB0` @ 115200 (`LUZ` / `GPS`) |

**Nota:** cerrar Monitor Serie de Arduino IDE antes de arrancar captura SDR si hay contención en el hub USB.

---

## 3. Driver y stack de software (Pi)

### 3.1 Capa principal — SDRplay API Service

| Componente | Detalle |
|------------|---------|
| Servicio systemd | `sdrplay.service` (**enabled**, **active**) |
| Binario | `/opt/sdrplay_api/sdrplay_apiService` |
| Biblioteca | `libsdrplay_api.so.3.14` → API **v3.14** |
| Ruta | `/usr/local/lib/` |
| Instalador | `~/Descargas/SDRplay_RSP_API-Linux-3.14.0.run` |
| Modelo de acceso | El **daemon (root)** posee el USB; las aplicaciones llaman `libsdrplay_api` por IPC |

**Secuencia API validada (2026-07-03):**

```text
sdrplay_api_Open()       → err=0 (Success)
sdrplay_api_GetDevices() → err=0, count=1
  [0] SerNo='0000000001' hwVer=1 valid=1
```

Script de prueba en repo: `docker/test_sdrplay_api.py`

```bash
export LD_LIBRARY_PATH=/usr/local/lib
python3 /home/ubuntu/anima/celula_madre/docker/test_sdrplay_api.py
```

### 3.2 Capa opcional — SoapySDR / GNU Radio / GQRX

| Componente | Versión / nota |
|------------|----------------|
| SoapySDR | v0.8.1 (`libsoapysdr0.8`) |
| Módulo SDRplay | `libsdrPlaySupport.so` → `/usr/local/lib/SoapySDR/modules0.8-3/` |
| Compilación módulo | Contra API **3.15** (warning si API instalada es **3.14**) |
| GNU Radio | `gr-sdrplay3` (20240224) |
| gr-osmosdr | 20220916 |
| GQRX | 20240224 (`/usr/local/bin/gqrx`) |

Para el **órgano ANIMA** se recomienda **`libsdrplay_api` directa**, no SoapyRemote.

### 3.3 Servicio deshabilitado (conflicto resuelto)

| Servicio | Estado | Motivo |
|----------|--------|--------|
| `soapyremote-server.service` | **disabled / inactive** | Competía con `sdrplay_apiService` → `libusb_claim_interface() -6` |

**Arreglo aplicado:** `docker/fix-sdr-hardware.sh` (lanzador escritorio: `Arreglar SDR RSP1.desktop`).

---

## 4. Incidente y resolución (2026-07-03)

### Síntomas

- `lsusb` veía el RSP1, pero `GetDevices` fallaba.
- Logs: `fwDownload failed`, `libusb_claim_interface() -6`.
- `soapyremote-server` activo desde el arranque.

### Causa raíz

Doble cliente USB: **SoapySDRServer** + **sdrplay_apiService** reclamando la misma interfaz.

### Solución

1. `systemctl stop` + `disable soapyremote-server.service`
2. `systemctl restart sdrplay.service`
3. Verificación con `test_sdrplay_api.py`

---

## 5. Reglas de diseño para el futuro órgano SDR (linajes §5)

1. **Lector central** en hilo propio → inyecta magnitudes crudas en la `fila` (no varios organelos abriendo el USB).
2. **Anti-Shannon:** potencia, banda, espectro reducido = números; E decide significado.
3. **Degradación elegante:** si `GetDevices` falla o `valid=0` → aporte 0; E sigue vivo.
4. **No usar SoapyRemote** en producción del organismo (riesgo de contención USB).
5. **Binding cross-modal (futuro):** correlación RF ↔ audio audible ↔ visión (emerge en atención, no asignado a mano).

### Campos sugeridos para bitácora (borrador)

```text
sdr_vivo
sdr_hw_ver
sdr_ser_no
sdr_freq_hz
sdr_sample_rate
sdr_rf_power_dbm
sdr_bandwidth_hz
sdr_agc
```

*(Definitivos cuando CS entregue el diseño del órgano.)*

### Variables de entorno sugeridas

```text
ANIMA_SDR_ENABLE=1
LD_LIBRARY_PATH=/usr/local/lib
```

---

## 6. Archivos de referencia en Célula_Madre

| Archivo | Función |
|---------|---------|
| `docker/test_sdrplay_api.py` | Prueba mínima `Open` + `GetDevices` |
| `docker/fix-sdr-hardware.sh` | Reparación (stop SoapyRemote, restart API) |
| `docker/arreglar-sdr-lanzador.sh` | Lanzador escritorio con sudo |
| `docker/Arreglar SDR.desktop` | Icono Pi «Arreglar SDR RSP1» |
| `linajes_sensoriales_exaptativos_ANIMA.md` | Marco: SDR = extensión del oído |
| `INFORME_Cloroplasto_Fisico_E_2026-07-02.md` | Cloroplasto (mismo hub USB) |

---

## 7. Criterios de éxito (hardware validado)

- [x] `lsusb` muestra `1df7:2500 SDRplay RSP1`
- [x] `sdrplay.service` activo
- [x] `soapyremote-server` deshabilitado
- [x] `GetDevices` → `count=1`, `valid=1`
- [ ] Captura IQ estable ≥ 30 min sin bloquear loop de E (pendiente órgano software)
- [ ] Degradación elegante verificada en corrida con organismo vivo

---

*Registrado: 2026-07-03 · Grok (Diotallevi) / validación en Pi `rpi` (192.168.86.33)*