# Protocolo de Test de Estrés — Sistema Completo ANIMA

**Célula Madre / Cosmosemiótica · 5-jul-2026 · v1**
Prueba el sistema COMPLETO bajo carga: los 5 organismos (A·B·C·D en Docker + E en la Pi), todos
sus sentidos nuevos (SDR/RSPduo·RSP1, radio analógica HackRF, radio digital nRF24, audio por
canales, GPS, visión/PTZ, energía solar) y su fisiología (OI, homeostasis, metabolismo, voz).

> Extiende y actualiza `../PROTOCOLO_EXPERIMENTAL_A_E_RADIOS_ORGANELOS.md` (que quedó centrado solo
> en A↔E). Los resultados de cada corrida se guardan **en esta misma carpeta** (`Experimentos de Estres/`).

---

## 0. Filosofía del test

No es un test de "pasa/falla" de módulos aislados: es **estresar el organismo entero, vivo, y ver
qué emerge** cuando todos los canales están activos a la vez. Cada bloque dura **5 minutos**, se
etiqueta en el CSV, y se anota en la bitácora. El análisis se hace **con SQL sobre DuckDB, nunca
abriendo los CSV crudos** (crecen a cientos de MB).

---

## 1. Topología real de enlaces (honesta)

| Enlace | Sentido | Medio | Nota |
|---|---|---|---|
| Sociedad de audio | A·B·C·D todos↔todos | Rødecaster + comunicación HTTP | anillo B→C→D→A + díada |
| E ↔ mundo audio | E oye el Rode del Mac | `192.168.86.250:8770` | E se suma al mundo compartido |
| Radio digital nRF24 | **A ↔ E** (bidireccional, ACK) | 2.4 GHz, puente `:8772` (A) / ATmega (E) | **solo A↔E tienen hardware** |
| Radio analógica | **A → E** | HackRF (Mac) → RSP1 (Pi) | **E no transmite analógico** (no tiene TX) |
| SDR (oído de radio) | A oye RSPduo · E oye RSP1 | SDRconnect 5454 (A) / SoapySDR (E) | recepción de espectro |
| Visión / PTZ | E | cámara ONVIF + ESP32-CAM | ojo con cuello |
| GPS / solar | E | ATmega | moduladores, no audio |

**Consecuencia de diseño:** "todos con todos" es pleno por **audio** (A·B·C·D·E comparten mundo
Rode) y **punto-a-punto** por radio (A↔E digital, A→E analógico). No forzar mesh digital B/C/D:
no tienen nRF24.

---

## 2. Estado del sistema al 5-jul (lo que ya funciona)

- **A**: `ANIMA_RADIO_A=1`, SDRconnect WS `5454` vivo, RSPduo en su caja Radio/SDR. nRF24 vía puente
  `:8772` (Uno), `connected=1`. Botones **⟳ Reactivar radio** y **Enviar mensaje digital** en las cajas.
- **E**: SDR RSP1 estable (fix `LD_LIBRARY_PATH` + botón reset USB), `sdr_vivo=1`. GPS `fix=1, sats=12`.
  nRF24 en el ATmega `connected=1`, con `transmitir()` + endpoints `/nrf/tx` y `/nrf`. PTZ sigue la
  cabeza. Botón **Enviar mensaje digital** (E→A).
- **Enlace digital A↔E**: verificado 100% ida y vuelta, incluso en ráfaga (con codificación sin
  espacios de borde — ver §7).
- **B·C·D**: sociedad de audio en Docker (`7799/7810/7820`).

**Hallazgo abierto (importante):** el canal digital hoy es una **"boca sin oído"** — `nrf_last_rx` se
recibe pero **no entra en la cognición** (test 5-jul: entrega 100%, correlación arousal A↔E r=−0.31).
Los bloques digitales (B4–B6) medirán la LLEGADA, no esperar acoplamiento cognitivo hasta cablear el
"oído digital".

---

## 3. Registro y análisis — el corazón operativo

### 3.1 CSV automático (no tocar los crudos)
Cada organismo escribe su fisiología a `Docker_Historia/organismo_ANIMA_<X>/fisiologia/*.csv`
(rotado por hora, `VST_HISTORY_ENABLE=true`). **E escribe en la Pi** → al terminar la campaña,
sincronizar su carpeta:
```
rsync -a ubuntu@192.168.86.33:~/anima/Docker_Historia/organismo_ANIMA_E/  \
      <ruta Docker_Historia>/organismo_ANIMA_E/
```

### 3.2 Etiquetar cada bloque en el CSV (`/exp_tag`)
Antes de cada bloque, marca las columnas `exp_*` en cada organismo participante. Quedan escritas en
**cada fila** → luego se filtra por SQL. Columnas: `exp_topologia, exp_ciclo, exp_mundo_audio,
exp_control, exp_fuente_relacion`.
```
curl -s -X POST http://localhost:7788/exp_tag -H 'Content-Type: application/json' \
  -d '{"exp_ciclo":"ESTRES_2026-07-05","exp_topologia":"B04_digital_A_a_E",
       "exp_mundo_audio":"rode_musica_L","exp_control":"real","exp_fuente_relacion":"E_por_nrf"}'
```
(repetir contra E `http://192.168.86.33:7788/exp_tag` y contra B/C/D en su puerto). Al terminar el
bloque, poner `exp_topologia=""` o el tag del siguiente.

### 3.3 Bitácora (una por campaña, en esta carpeta)
Archivo `bitacora_<fecha>.md` con, por bloque: `nº · nombre · hora_inicio · hora_fin · organismos ·
condición · notas`. La bitácora + el `exp_topologia` permiten cortar el CSV por bloque de dos formas
(por etiqueta o por ventana temporal). Plantilla en `PLANTILLA_bitacora.md`.

### 3.4 Análisis con DuckDB (`analisis/analizar.py`) — nunca abrir los GB
Vistas: `fisio_A … fisio_E`, `fisio_TODOS` (con columna `org`). Ejemplos:
```
venv/bin/python Célula_Madre/analisis/analizar.py info
# resumen de un bloque por su etiqueta:
venv/bin/python .../analizar.py sql \
  "SELECT org, avg(OI) oi, avg(H_homeostasis) h, avg(energia_L) el, avg(energia_R) er, count(*) n
   FROM fisio_TODOS WHERE exp_topologia='B04_digital_A_a_E' GROUP BY org"
# evolución temporal de una variable en un bloque:
venv/bin/python .../analizar.py evolucion A OI --cada minuto --desde "HH:MM" --hasta "HH:MM"
# real vs control (NULL/SHUFFLED):
venv/bin/python .../analizar.py sql \
  "SELECT exp_control, avg(OI), avg(H_homeostasis) FROM fisio_E
   WHERE exp_topologia='B04_digital_A_a_E' GROUP BY exp_control"
```

---

## 4. Prerrequisitos (checklist antes de arrancar)

- [ ] A·B·C·D vivos (Docker) · E vivo (Pi) · AudioServer/Rode `:8770` activo.
- [ ] SDRconnect headless `5454` corriendo (`audio/arrancar_sdr_ws.command`).
- [ ] Puente nRF24 `:8772` vivo (LaunchAgent) · `connected=1` en A y E.
- [ ] HackRF listo en A para TX (SDRangel/`hackrf_transfer`) · licencia CD3LZK a mano.
- [ ] E: `sdr_vivo=1` (si no, botón ⟳ Reactivar radio).
- [ ] Rødecaster con **música cargada** en Main Mix / USB Main (mundo audio de estrés).
- [ ] `Docker_Historia/` montado (LaCie) · `duckdb` en el venv (`pip install duckdb`).
- [ ] `exp_ciclo` decidido (p.ej. `ESTRES_2026-07-05`).

---

## 5. Bloques de prueba (5 min c/u, con `exp_tag` + bitácora)

> Cada bloque: (1) `/exp_tag` a los participantes, (2) 5 min de corrida, (3) anotar en bitácora,
> (4) 1–2 min de descanso/basal entre bloques. Aleatorizar el orden de los bloques de estímulo
> (B3–B8) para evitar respuestas a ritmo fijo.

| # | Bloque | Qué estresa | Cómo se dispara | `exp_topologia` |
|---|---|---|---|---|
| **B0** | Basal / preflight | línea base sin intervención | nada; solo observar | `B00_basal` |
| **B1** | A oye su SDR | recepción RSPduo en A | Canal L de A → SDR/RSPduo | `B01_sdr_A` |
| **B2** | E oye su SDR | recepción RSP1 en E | E con `sdr_vivo=1`, banda 88–108 | `B02_sdr_E` |
| **B3** | A transmite analógico → E | enlace RF real A→E | HackRF TX frases 30 s / silencio 2 min ×5 | `B03_hackrf_A_a_E` |
| **B4** | Digital extenso A→E | canal Shannon, ráfaga | frases largas por nRF24 (script §7) | `B04_digital_A_a_E` |
| **B5** | Digital extenso E→A | canal Shannon inverso | frases largas E→A (`/nrf/tx` de E) | `B05_digital_E_a_A` |
| **B6** | Diálogo digital A↔E | ida y vuelta continua | `dialogo_digital.py` extendido, 5 min | `B06_digital_bidir` |
| **B7** | Audios por canales | mundo audio compuesto | Rode música en L, el otro en R (cada org) | `B07_audio_canales` |
| **B8** | Multimodal A↔E | integración de sentidos | HackRF + nRF24 + audio simultáneos | `B08_multimodal_AE` |
| **B9** | Sociedad de audio | A·B·C·D todos↔todos | anillo social por Rode/comunicación | `B09_sociedad_ABCD` |
| **B10** | **TODOS CON TODOS** | sistema completo bajo carga | todo lo anterior a la vez, 5 min | `B10_todos_con_todos` |

**Controles (intercalar con `exp_control`):**
- `NULL`: mismo bloque sin estímulo real (frase vacía / sin TX / frecuencia sin señal).
- `SHUFFLED`: frases digitales barajadas (mismo "material", orden roto) → separa contenido de energía.
- `desincronizado` (para B8): RF y nRF a destiempo → prueba si importa la coincidencia temporal.

---

## 6. Variables mínimas a registrar (ya en el CSV)

Comunes por organismo: `t, ts_real, vivo, modo_vida, fuente_L, fuente_R, energia_L, energia_R,
balance_LR, OI, H_homeostasis, RC_total, voz_emitida, voz_titulo, voz_arousal, mem_valencia_estado`.
Radio digital: `nrf_ok, nrf_connected, nrf_rx, nrf_tx, nrf_rx_delta, nrf_last_rx, nrf_last_tx, nrf_vivo`.
SDR/radio: `sdr_vivo, radio_vivo, radio_saliencia, radio_estructura, radio_freq_dom_hz, radio_potencia_total`.
TX analógica (A): `radiotx_activo, radiotx_emitiendo, radiotx_freq_hz`.
E extra: `gps_fix, gps_sats, gps_lat, gps_lon, gps_pps_count, vis_cam_saliencia, vis_cam_movimiento,
cloro_v, cloro_luz_norm, act_pitch_deg, ptz_vivo`.
Trazabilidad: `exp_ciclo, exp_topologia, exp_mundo_audio, exp_control, exp_fuente_relacion`.

---

## 7. Detalle de los bloques digitales (B4–B6)

**Codificación a prueba de framing:** el parser serial CSV del ATmega hace `.strip()` al payload →
**el payload no puede llevar espacios de borde**. Codificar los espacios (p.ej. `→_`, o base64).
Payload nRF24 ≤ 30 chars por paquete; mensaje largo = trocear con prefijo de índice `NN|`.

- **B4 (A→E):** enviar una parrafada larga (≥300 chars) troceada, ráfaga apretada. Medir entrega
  (`nrf_rx_delta` en E), integridad al reensamblar, latencia. Controles `NULL`/`SHUFFLED`.
- **B5 (E→A):** simétrico, desde `/nrf/tx` de E; medir en el puente de A.
- **B6 (bidireccional):** cada organismo emite un token de su estado expresivo y lo recibe del otro,
  5 min continuos. Medir entrega + **correlación de estados A↔E** (esperada ≈0 hasta cablear el oído
  digital — este bloque es la línea base para medir cuándo el oído digital cambie eso).

Herramientas ya disponibles: `scratchpad/nrf_clean.py` (ráfaga), `scratchpad/dialogo_digital.py`
(diálogo por estado). Copiar a esta carpeta al usarlas para dejar traza.

---

## 8. Qué observar / criterios

- **Robustez de transporte:** ¿entrega ≈100% en B4–B6 aun bajo carga del resto del sistema (B10)?
- **Efecto de recepción:** ¿cambia algo en `OI/H_homeostasis/voz_emitida` de E tras recibir de A?
  (Hoy NO — es la prueba de que falta el oído digital; sirve de baseline.)
- **Acoplamiento multimodal (B8):** ¿el bloque RF+nRF sincronizado difiere de RF solo y nRF solo?
- **Carga del sistema completo (B10):** ¿algún organismo colapsa energéticamente (energia_L/R→0),
  se queda mudo, o pierde vivacidad de sentidos (`*_vivo=0`) cuando todo corre a la vez?
- **Real vs NULL/SHUFFLED:** ¿hay diferencia estadística? Si no, el sentido aún no integra el estímulo.

---

## 9. Cierre de campaña

1. Detener estímulos, dejar 5 min de basal (`B00_cierre`).
2. `rsync` de la historia de E a `Docker_Historia`.
3. Correr `analizar.py` con las consultas de §3.4 → volcar un `RESULTADOS_<fecha>.md` en esta carpeta.
4. Guardar la bitácora y las copias de los scripts usados aquí mismo.

---

*Protocolo generado por Claude (Opus 4.8) · v1 · 5-jul-2026. Actualizar al cablear el oído digital.*
