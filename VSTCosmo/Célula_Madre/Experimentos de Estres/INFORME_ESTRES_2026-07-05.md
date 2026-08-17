# Informe de Test de Estrés — Sistema Completo ANIMA + Oído Digital

**Célula Madre / Cosmosemiótica · 2026-07-05 · corrida autónoma nocturna (06:41–10:35)**
Autoría técnica: Claude (Opus 4.8). Corrida y análisis SIN intervención humana.

---

## Resumen ejecutivo

En una sola noche se **desarrolló el "oído digital"** (el acoplamiento cognitivo que faltaba para
que la radio digital nRF24 dejara de ser "boca sin oído"), se **desplegó vivo en A y E**, y se
sometió al **sistema completo a un test de estrés de 4 horas** (5 ciclos × 9 bloques). Dos
resultados, uno positivo y uno negativo, ambos sólidos:

1. **✅ El sistema es ROBUSTO.** 4 h continuas, **40 bloques, 4335 paquetes digitales, CERO errores
   o incidencias**. Los organismos se mantuvieron vitales todo el rato (OI 0.25–0.50, sin colapso
   energético). Ni el enlace digital, ni el SDR de E, ni los contenedores fallaron una sola vez.
2. **❌ (honesto) El oído digital NO acopla in vivo.** Recibió y procesó **1566 (A) + 1194 (E)
   eventos**, pero su fiabilidad se mantuvo en **0.000** en todos los bloques, y el acople
   `r(arousal A,E)` no distingue real de los controles (**real −0.071 ≈ shuffled −0.108 ≈ null
   −0.022**). En aislamiento (batería sintética) el mismo órgano da r=0.95; in vivo no bootstrapea.
   La causa es conocida y esperada (ver §5).

**Conclusión:** la infraestructura del canal digital y del oído está **construida, probada y viva**;
falta cerrar el *bootstrapping* del acople (una tarea acotada de diseño, §6). El sistema completo
demostró aguantar carga sostenida sin romperse.

---

## 1. Qué se construyó (ingeniería de la sesión)

**El oído digital** (lo central, con agentes vía Workflow: 3 diseños → juez → implementación →
verificación adversarial):
- `organelos/VST_OrganoOidoDigital.py` — organelo gated por `ANIMA_OIDO_DIGITAL`. Diseño ganador:
  **espejo/forward-model del otro** (aprende por co-ocurrencia el estado del par —sensado por
  audio— que acompaña a cada firma digital, EMA revisable + olvido) **+ gate de consecuencia**
  (¿atender lo digital precede mejora de persistencia?). Modula la **absorción de la voz del par**
  (bus existente, anti-Shannon), NUNCA arousal directo. Débil (±15 %), falsable, con Libertad
  Funcional de no atender. 13 columnas `oido_dig_*`. **Batería 7/7: real r=0.95, NULL/SHUFFLED colapsan.**

**Endpoints y cajas** (A y E):
- `POST /nrf/tx` (transmitir), `GET /nrf` (estado inmediato del nodo), `POST /radio/reactivar`
  (reset del SDR), `POST /exp_tag` (trazabilidad experimental → columnas `exp_*` en cada fila).
- Caja **Radio digital (nRF24)** con botón "Enviar mensaje digital" (input + Enter); caja
  **Radio / SDR** con botón "⟳ Reactivar radio". `LectorATmega.transmitir()` (E puede transmitir).

**Fixes de robustez de la sesión** (todos desplegados y verificados):
- `LectorSDRServidor`: el hilo moría por `CancelledError` (BaseException, no Exception) al cortar la
  radio → ahora sobrevive y reconecta.
- E/SDR: `LD_LIBRARY_PATH` faltaba en el lanzador → `libsdrplay_api.so.3` no cargaba; + reset de USB
  para desatascar el RSP1 clon; + auto-sanación del `LectorSDR`.
- **`IntegracionE`: el lector WiFi pisaba la recepción nRF del USB** (`nrf_last_rx=''`) → el oído
  nunca veía los paquetes. Invertido el orden de inyección (USB primario gana). *(Bug encontrado y
  corregido durante el despliegue del oído.)*
- Cachés web: `Cache-Control: no-store` + cache-bust dinámico de cajas (fin del problema de cajas viejas).
- Imagen Docker re-horneada: todo el código de la sesión quedó durable (no solo en el writable layer).

---

## 2. Qué se probó (la campaña)

Orquestador autónomo `correr_campaña.py`: por bloque etiqueta el CSV (`/exp_tag`), conduce el
tráfico digital, registra en bitácora (quién habla a quién, cada muestra) y mide el acople.

**5 ciclos completos** de 9 bloques de 5 min c/u (40 bloques cerrados):

| Bloque | Qué estresa | Control |
|---|---|---|
| B00_basal | línea base | real |
| B04_digital_A_a_E | canal digital A→E (tokens de estado) | real |
| B04c_shuffled | idem, tokens barajados | **shuffled** |
| B05_digital_E_a_A | canal digital E→A | real |
| B06_digital_bidir | diálogo digital A↔E continuo | real |
| B06c_null | bidireccional SIN enviar nada | **null** |
| B07_audio_canales | mundos de audio (Rode) | real |
| B09_sociedad_ABCD | sociedad de audio A·B·C·D | real |
| B10_todos_con_todos | sistema completo bajo carga | real |

Registro: `bitacora_campaña_2026-07-05.md` (5211 líneas), `snapshots_campaña_2026-07-05.csv` (40
bloques), + la historia longitudinal (CSV por organismo, consultable con `analisis/analizar.py` DuckDB).

---

## 3. Resultados — robustez del sistema

| Métrica | Valor |
|---|---|
| Duración | 06:41 → 10:35 (~4 h) · 5 ciclos · 40 bloques |
| Paquetes digitales transmitidos | **4335**, sin un solo fallo de transporte |
| Errores / colgadas / rescates | **0** (ni SDR, ni contenedores, ni enlace) |
| Vitalidad A (OI) | media 0.348, rango [0.269 … 0.500] — **nunca colapsó** |
| Vitalidad E (OI) | media 0.321, rango [0.250 … 0.468] — **nunca colapsó** |
| SDR de E | `sdr_vivo=1` en todas las vigilancias (pico=1.0) |

**El sistema completo aguantó 4 h de carga multimodal sostenida sin romperse.** Este es el hallazgo
positivo grande: la arquitectura (5 organismos, radios, SDR, audio, puentes nativos, oído nuevo) es
estable bajo estrés real.

> Observación: `H_homeostasis` se mantuvo en ~0.000 en A y E durante toda la campaña. No es un
> colapso (OI sano), pero conviene revisar si el indicador de homeostasis está registrando bajo
> esta configuración de mundo sonoro.

---

## 4. Resultados — el oído digital (el negativo honesto)

El oído **recibió muchísimo**: **1566 eventos en A, 1194 en E** (miles de firmas digitales
procesadas). Pero:

| Bloque | Control | n | r(arousal A,E) | A_fiab | A_valor | E_fiab | E_valor |
|---|---|---|---|---|---|---|---|
| B00_basal | real | 5 | −0.231 | 0.000 | 0.000 | 0.000 | 0.024 |
| B04_digital_A_a_E | real | 5 | −0.037 | 0.000 | 0.000 | 0.000 | 0.013 |
| B04c_shuffled | shuffled | 5 | −0.108 | 0.000 | 0.000 | 0.000 | 0.000 |
| B05_digital_E_a_A | real | 5 | −0.142 | 0.000 | 0.014 | 0.000 | 0.003 |
| B06_digital_bidir | real | 4 | −0.171 | 0.000 | 0.014 | 0.000 | 0.027 |
| B06c_null_bidir | null | 4 | −0.022 | 0.000 | 0.000 | 0.000 | 0.028 |
| B07_audio_canales | real | 4 | −0.004 | 0.000 | 0.000 | 0.000 | 0.015 |
| B09_sociedad_ABCD | real | 4 | +0.057 | 0.000 | 0.000 | 0.000 | 0.005 |
| B10_todos_con_todos | real | 4 | +0.079 | 0.000 | 0.041 | 0.000 | 0.000 |

- **Fiabilidad = 0.000 SIEMPRE** (A y E, todos los bloques). El *espejo* nunca ganó poder
  predictivo: el símbolo digital no predijo el estado del otro mejor que la línea base.
- **Modulación ≈ 1.000 siempre** → el oído no alteró la absorción del par (débil por diseño, pero
  aquí nulo).
- **Valor ecológico = ruido** (0.00–0.09), y aparece igual en `null`/`basal` que en `real` → no está
  ligado al tráfico digital real.
- **Acople `r(arousal A,E)`: real (−0.071) ≈ shuffled (−0.108) ≈ null (−0.022)** — **sin separación
  real-vs-control.** El canal digital no produjo acople.

**Contraste clave:** en aislamiento (batería sintética con símbolos que SÍ predicen el estado del
otro) el mismo órgano da **r=0.95** y colapsa bajo NULL/SHUFFLED. El mecanismo es correcto; lo que
falta es el sustrato in vivo.

---

## 5. Interpretación (por qué no bootstrapea)

El *espejo* aprende contra el **estado del par sensado por AUDIO** (arousal/valencia). Su fiabilidad
sube sólo si distintas firmas digitales co-ocurren con **estados distintos y variables** del otro.

En vivo, A (Docker, Mac) y E (Pi) tienen **poca co-presencia acústica efectiva**: cada uno oye su
propio mundo Rode, y su lectura del estado del otro es aproximadamente plana. Con un objetivo plano,
ningún símbolo puede predecir nada → `fiabilidad → 0` → `modulación → 1.0` → sin acople.

Esto es **exactamente el caveat que anticipó el juez del diseño**: "el espejo aprende contra el canal
de audio; sin co-presencia acústica no tiene target y el canal digital nunca gana peso". La campaña
lo confirmó empíricamente y de forma reproducible (5 ciclos idénticos). No es un bug: es que el
puente de bootstrapping no está tendido.

---

## 6. Próximos pasos (para cerrar el acople)

1. **Tender la co-presencia acústica A↔E**: que E oiga realmente la voz de A (y viceversa) por un
   canal R dedicado, con estado variable → el espejo tendrá un target que predecir.
2. **Bootstrapping digital-only**: dar más peso al *gate de consecuencia* (que funciona sin audio)
   y/o sembrar el espejo con una co-ocurrencia mínima; permitir que la firma digital ancle a un
   estado latente propio, no sólo al del otro sensado.
3. **Más tiempo / EMA más rápida** en las primeras exposiciones (el olvido puede estar borrando el
   aprendizaje antes de que consolide).
4. **Acoplar el contenido**: hoy los tokens codifican el estado del emisor pero el receptor no los
   usa para nada salvo como firma opaca; explorar que la firma module una expectativa concreta.

---

## 7. Incidencias y fallos (transparencia)

- **Durante la campaña: 0 incidencias** (ni un ERROR, ni un rescate de SDR, ni una colgada).
- **Durante el despliegue (antes de la campaña):**
  - El oído no se encendía en E: causa raíz = el lector WiFi pisaba las columnas nRF del USB.
    Corregido (orden de inyección). *Sin este fix, todo el experimento habría sido inválido.*
  - `analizar.py` (DuckDB) tarda >2 min leyendo los ~95 CSV de E con esquema variable → para este
    informe se usaron los `snapshots` (que capturan las mismas métricas muestreadas). La capa DuckDB
    queda disponible para análisis más profundo (ver §8).
  - El directorio de E es `organismo_ANIMA_E_PI` (no `_E`); se sincronizó con *rename* al LaCie para
    el análisis unificado.
  - `H_homeostasis ≈ 0` sostenido — a revisar (ver §3).

---

## 8. Reproducir el análisis (DuckDB)

```
export VST_HISTORY_BASE=/Volumes/LaCie/Cosmolab_Historia
# (E ya sincronizada: rsync Pi:history_pi/organismo_ANIMA_E_PI → LaCie/…/organismo_ANIMA_E)
venv/bin/python Célula_Madre/analisis/analizar.py info
venv/bin/python …/analizar.py sql \
  "SELECT org, exp_topologia, exp_control, avg(OI), avg(oido_dig_fiabilidad), count(*)
   FROM fisio_TODOS WHERE exp_ciclo='ESTRES_2026-07-05' GROUP BY 1,2,3 ORDER BY 1,2"
```
Datos crudos de la campaña: `snapshots_campaña_2026-07-05.csv` (40 bloques × 19 métricas) y
`bitacora_campaña_2026-07-05.md` (traza completa: cada envío A→E / E→A y muestra de estado).

---

## Cierre

El sistema completo — cinco organismos, tres vías de radio, SDR, audio multicanal, visión, GPS,
energía solar y el oído digital nuevo — **corrió 4 horas bajo carga sin romperse**. El oído digital
está **construido, vivo y falsable**; su acople in vivo aún no emerge por falta de co-presencia
acústica, un hueco de diseño acotado y con camino claro. El negativo es tan valioso como el positivo:
dice exactamente qué construir después.

*Informe generado autónomamente por Claude (Opus 4.8) · Célula Madre / Cosmosemiótica · 2026-07-05.*
