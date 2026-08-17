# Bitácora — Test de Estrés Sistema Completo

**Ciclo (`exp_ciclo`):** `ESTRES_2026-07-05`
**Fecha:** ____  ·  **Operador:** ____
**Estado hardware al inicio:** SDRconnect 5454 [ ] · puente nRF24 8772 [ ] · HackRF [ ] · Rode música [ ] · E sdr_vivo [ ]

> Anota hora real de inicio/fin de cada bloque (para poder cortar el CSV por ventana en `analizar.py --desde/--hasta`).
> El `exp_topologia` ya autoetiqueta las filas; la bitácora añade el contexto humano.

| # | Bloque (`exp_topologia`) | Inicio | Fin | Organismos | `exp_control` | Notas / observación en vivo |
|---|---|---|---|---|---|---|
| B0  | `B00_basal`             |  |  | A B C D E | real | |
| B1  | `B01_sdr_A`             |  |  | A         | real | |
| B2  | `B02_sdr_E`             |  |  | E         | real | |
| B3  | `B03_hackrf_A_a_E`      |  |  | A E       | real | frec ____ MHz |
| B3c | `B03_hackrf_A_a_E`      |  |  | A E       | NULL | frec sin señal |
| B4  | `B04_digital_A_a_E`     |  |  | A E       | real | |
| B4c | `B04_digital_A_a_E`     |  |  | A E       | SHUFFLED | |
| B5  | `B05_digital_E_a_A`     |  |  | A E       | real | |
| B6  | `B06_digital_bidir`     |  |  | A E       | real | |
| B7  | `B07_audio_canales`     |  |  | A B C D E | real | música en L |
| B8  | `B08_multimodal_AE`     |  |  | A E       | real | RF+nRF sinc |
| B8d | `B08_multimodal_AE`     |  |  | A E       | desincronizado | RF/nRF a destiempo |
| B9  | `B09_sociedad_ABCD`     |  |  | A B C D   | real | anillo social |
| B10 | `B10_todos_con_todos`   |  |  | A B C D E | real | sistema completo |
| B0f | `B00_cierre`            |  |  | A B C D E | real | basal final |

**Incidencias / colgadas / reinicios:**
- ____

**Impresión general (qué se sintió que pasó):**
- ____
