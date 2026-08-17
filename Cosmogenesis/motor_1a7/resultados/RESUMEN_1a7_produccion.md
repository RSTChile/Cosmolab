# Motor unificado 1→7 — producción

**Fecha:** 2026-07-23  
**chain_pass en JSON:** `true` (con etapa 7 = v6)  

---

## ⚠ ESTATUTO (auditoría director): **cadena NO CERRADA**

Ver: `../../HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md`

El `chain_pass=True` del JSON **depende** de sustituir la etapa 7 (pre-registrada como **v5**, que falló al 0.40) por **v6**, cuya “masa” está construida con las mismas variables del linaje.  
**No se presenta este archivo como cierre del enfoque 1→7.**

| etapa | pass operativo | nota de rigor |
|-------|----------------|---------------|
| 1 Campo + ε | ✅ | limpio (r-cruz) |
| 2 Expansión r | ✅ | limpio |
| 3 Estiramiento | ✅ | limpio |
| 4 Densidad | ✅ | limpio |
| 5 Orden sin masa | ✅ | limpio |
| 6 Átomo E3 | ✅ | limpio |
| 7 Masa linaje | ⚠ **abierta** | v6 ≠ prueba de masa; tautología linaje↔mass_obs |

**Primera corrida producción (con v5):** `chain_pass: false` (conservar ese hecho).  
**Segunda corrida (tras swap a v6):** `chain_pass: true` en disco — **no admite** como cierre.

## Lectura

Etapas **1–6** sostienen el relato hasta el átomo.  
Etapa **7** queda como **hallazgo abierto** hasta un observable de masa **independiente** del juez de linaje, o hasta renunciar honestamente a la palabra “masa” y claimar solo linaje.
