# Transparencia Experimental — VSTCosmo / ANIMA Series

**Propósito**: Responder directamente a cuestionamientos de verificabilidad (ej. interacciones con Grok en X, junio 2026) proporcionando:
- Ubicación exacta de mecanismos en código.
- Métricas cuantitativas de ejecuciones reales (de logs y reportes).
- Distinción explícita entre dinámicas programadas y comportamientos/resultados observados.
- Referencias a datos disponibles en el repo (logs/, csv, png, PDFs canónicos).
- Reconocimiento de limitaciones y estatus de las claims.

**Repo**: https://github.com/RSTChile/Cosmolab/tree/main/VSTCosmo (y versión local completa en este workspace).

**Fecha de este documento**: 2026-06 (basado en commits locales y logs hasta V167).

## 1. Principio general de transparencia

Todos los "comportamientos orgánicos" (fatiga, ritual, lateralidad, R relacional, meta-representación, etc.) son **consecuencias de dinámicas explícitas** integradas en arquitecturas de control/campo que se refinaron iterativamente vía falsaciones (ver Addendum CN202, Informe VSTCosmo, Hito v80h, Síntesis V90-V103, Informe ANIMA-1).

No se afirma "biología literal en silicio". Se afirma:
- Implementación de nodos de la Teoría Cosmosemiótica Canónica (C-Nx, O-Nx) en sustrato computacional.
- Que ciertas combinaciones de mecanismos producen regímenes estables con propiedades análogas a las descritas en biología (persistencia de diferencia, exaptación de marcos rígidos bajo presión, costo recuperable, etc.).
- "Organismicidad" aquí = configuración que sostiene S > 0 (persistencia) + las dinámicas específicas de acoplamiento, memoria histórica, costo de persistencia y libertad funcional bajo los constraints cosmosemióticos.

El valor científico está en:
- La cadena de falsaciones que llevó a la arquitectura actual (v55–v72c para modos propios; V122–V150 para ANIMA-1; V153–V167 para ANIMA-2 ritual/R^R).
- Las métricas cuantitativas reproducibles bajo parámetros fijos.
- Los trade-offs medidos (ej. ritual mejora precisión post-desafío pero reduce exploración/juego).

## 2. ANIMA-1 (V122–V150): Cierre — "Primer organismo artificial mínimo (IONB-1)"

**Script principal**: [V150.py](V150.py)

**Mecanismos clave (programados explícitamente)**:

- **Separación historia vs fatiga activa** (líneas ~142-198):
  ```python
  class FatigaMetabolicaV150:
      self.historia = 0.0          # permanente, nunca decae
      self.fatiga_activa = 0.0     # recuperable, decae con tau
      # acumula abs(delta) durante trabajo; *= exp(-dt/tau) en reposo
  ```
  Efectos **solo de fatiga_activa**:
  - `factor_gain = exp(-k_gain * fatiga_activa)` → reduce Kp efectivo (K_GAIN=0.0003)
  - `zona_muerta_efectiva = ZONA_MUERTA_BASE + k_precision * fatiga` (K_PRECISION=0.002, base=2.0° → máx 15°)
  - `temblor = k_temblor * fatiga_activa * randn()` (K_TEMBLOR=0.001) → "Parkinson computacional endógeno"

- Controlador P con plasticidad Kp (habit uación/sensibilización según oscilación vs zona muerta), inercia 0.95, zona muerta base, bihemisferio previo para R + lateralidad.

**Parámetros finales calibrados** (V150:12-18, desde V147 baseline):
- KP_BASE=0.002, INERCIA=0.95, K_GAIN=0.0003, K_PRECISION=0.002, K_TEMBLOR=0.001, TAU_RECUPERACION=180.0s
- ZONA_MUERTA_BASE=2.0°, MAX=15.0°

**Resultados cuantitativos de la corrida V150 (cierre)** (de ejecución reportada en script + logs/v150_logs/*.png):

Fases (protocolo de 3+50+3 ciclos alternantes + reposo 180s + post):
- F1 (fresco): error final ~2.1°, amplitud ~115.6° (o 57.8° por polo), T_settle ~31.0s
- F3 (fatigado tras ~50 ciclos): error ~15.0° (×7.1), amplitud reducida ~45°, fatiga_activa alta, zona muerta expandida
- F5 (post-reposo): recuperación parcial de fatiga (ej. ~% calculado en script)

Degradación demostrada sin colapso total. Recuperación por reposo (fatiga decae, no historia). Temblor aumenta con fatiga.

**Evidencia en repo**:
- v150_logs/v150_cierre_anima1_*.png (gráficos de orient, fatiga, zona muerta, barras error)
- Prints internos en V150.py:158-568 (tablas comparativas, degradacion_error, recuperacion_fatiga)
- Logs previos V147 (baseline sano), V148-V149 (desarrollo fatiga)

**Ablación / control implícito**: Versiones anteriores (V140-146) fallaron en inducir fatiga útil porque "casi no había trabajo real" (pegado a setpoint); se requirió baseline con alternancia real y separación historia/estado para que el costo tuviera sentido. Ver comentarios en V150 y previos.

**Limitación reconocida**: El "temblor endógeno" es ruido gaussiano escalado; el "Parkinson" es analógico. El modelo de fatiga es una implementación directa de "costo acumulable recuperable", no derivado de simulación física de neuronas/músculos.

## 3. ANIMA-2 (V153–V167+): Ritual, Juego, Cb, R^R (meta-representación)

**Scripts clave**:
- V165.py (ritual con persistencia natural, cierre práctico)
- V167.py (Etapa 4: META-REPRESENTACIÓN OBSERVACIONAL Rᴿ — versión corregida)
- V162-V164 (desarrollo detector ritual por cruces de cero + integraciones)
- V157 (juego A/B paralelo, baseline)

**Mecanismos clave (programados)**:

- **Cb (consciencia básica / presión de desacople)**: integral dCb/dt = e_R * (1 - A_sys-env) - Cb/tau (desde V155+). Mide esfuerzo para sostener acoplamiento.

- **Juego (desacople enactuado amortiguado)**: modo que permite más movimiento/costo para "jugar" (explorar?); implementado como influencia separada.

- **RitualV166** (V167.py:297-391, heredado de V162+):
  - `detectar_cruce_por_cero(orientacion)`: cuenta cambios de signo en orientación del motor.
  - Si cruce + Cb > umbral (28.0): busca patrón temporal (dt ≈ 40.0s ±30% tol) en buffer.
  - Si repeticiones_consecutivas >=3: activation += (Cb * reps /100) * dt ; decae con tau=180s.
  - Si activation > 0.4: active=True; persiste mientras ciclos_sin_cruce < persistencia_min (3), sino decaimiento suave.
  - `modular_correccion`: cuando active, mezcla delta_raw con termino ritual (reduce exploración 30%*activation + bias ritual).

- **MetaRepresentacionObservacional** (V167.py:397-426): monitor "desajuste".
  - Acumula error medio en buffer.
  - Si ritual_activo AND error_sostenido >=15°: presion = min(1, error/60); integra hacia desajuste (tau=30s).
  - Output: señal [0,1] que se usa para validar correlación.

**Parámetros ritual/Meta** (V167.py:66-84):
RITUAL_TAU=180.0, REPETICION_MIN=3, PATRON_TEMPORAL=40.0, UMBRAL_ACTIVACION=0.4, UMBRAL_CB=28.0, etc.
META_UMBRAL_ERROR=15.0, VENTANA=200, etc.

**Resultados cuantitativos de corridas recientes (ej. V167 log + tabla cronología compartida en X)**:

De log v167_run_20260603_044554.log (Etapa 4, Rᴿ observacional, 20+20 ciclos F2/F3 + F4 invertido):

F2 Control (sin ritual, 20 ciclos):
- Al final: fatiga=20000°, historia=3803°, Cb=500.0, juego activo mucho tiempo.

F3 Ritual (con detector, 20 ciclos):
- Ritual act fluctúa (0.147 a 0.308), cruces acumulados altos (92 al final).
- Historia acumulada diferente (a veces mayor en ritual).
- Al final F3: fatiga~20000, historia~4692.

F4 Test post (setpoint invertido, prueba de persistencia/observación):
- CONTROL: Cb=487.7, juego=True (1742s), fatiga=20000, historia=3921
- RITUAL: Cb=0.0, juego=False (1491s), ritual activo=True act=0.412, cruces=130, Meta-R desajuste=1.000, fatiga~19601, historia=5336

Métricas Rᴿ:
- Ritual activo en F4: True ✅
- Señal desajuste máx: 1.000
- Pasos ritual + error_alto (≥15°): 8227
- Señal >0.5 en esas cond: True
- Correlación ritual_activo ↔ señal (error sostenido): **0.988 (n=21677)** ✅
- Error RMS F4 ritual: 30.21°

**Criterios Etapa 4 cumplidos** (3/3).

**De tabla cronología (V165 ejemplo previo, ritual estabilizador)**:
- Tiempo ritual activo ~12.7% (203s), persiste en F4=True, act_final~0.491
- RMS post F4: control ~110°, ritual ~30° (mejor precisión a costa de menos variación/juego inhibido -80s)
- Historia ratio ritual/control ~0.92 (compresión moderada)

**Controles/ablaciones presentes**:
- V159: ritual sin entrenamiento/ historia/ variabilidad → detector nunca activa (ritual_act=0, 0/20 episodios). Confirma que requiere presión + historia + patrón.
- V161: ritual sobre "tormenta" (ruido fuerte + fatiga saturada, sin patrones) → permanece 0.
- V157 / V162 etc: ramas paralelas A/B idénticas excepto presencia del módulo ritual/juego → comparaciones directas de historia, Cb, error RMS post, tiempo en modos.
- V164: ritual jerárquico (inhibe juego) → comprime historia (0.74×), empeora error post (62.8° vs 38.3° control) — muestra versión "degradante/rígida".
- V165/V167: versión con persistencia natural (sin forzar off en F4) y correcciones → ritual puede estabilizar o correlacionar con señal bajo desafío.

Estos controles muestran que el surgimiento/efecto del ritual **depende de la integración** con Cb (presión), historia de trabajo, y el detector de patrón temporal bajo las condiciones del aparato motor fatigable.

**Evidencia en repo**:
- v167_logs/*.log (prints detallados + métricas finales + correlación)
- v167_logs/*.png (plots)
- v165_logs/, v164_logs/, v163_logs/, v162_logs/, v159_logs/ (serie completa de etapas)
- V167.py (código completo + prints de validación)
- Similar para V150 y previos.

## 4. Datos raw / logs disponibles (no solo reportes internos)

- **Tiempo series / historiales**: muchos vXXX_logs/ contienen .csv (ej. v100_logs/v100_resultados_*.csv, v72a_transiciones.csv, historial_*.csv en root para viento/voz), .json (v111b_logs etc.), y los .log contienen dumps de arrays clave por fase.
- **Plots**: sistemáticos en cada vXXX_logs/ (orientación, fatiga, Cb, ritual_act, histogramas, etc.).
- **CSVs de transiciones y perfiles espectrales**: de fases tempranas de campo continuo (v70-v80h).
- **PDFs canónicos con datos brutos**:
  - TeoriaCosmosemiotica Addendum CN202 EvidenciaComputacional Abril2026.pdf : datos v72c Fase5 (W_tras_entreno=0.0424, Phi_hist=0.157, gradientes 0.2393/0.5290, ratio espectral diff 9.951 voz/ruido, frecuencias dominantes modo 19→38, etc.).
  - Síntesis de Experimentos V90 a V103.pdf + .docx : Ω por estímulo/dirección, clustering.
  - INFORME CANÓNICO DE CLAUSURA DE ETAPA - VSTCosmo 150.pdf
  - Teoría Cosmosemiótica Aplicada - Informe VSTCosmo.pdf
  - Hito v80h, Fourier CampoContinuo, etc.

**Limitación actual (reconocida)**: No todos los runs guardan arrays completos de series temporales en formato estandarizado .csv por defecto (algunos solo png + prints resumidos). Los experimentos tardíos (ANIMA) priorizaron métricas agregadas y plots. Esto es mejorable (ver sección 5).

**Cómo reproducir**:
- Clonar repo.
- Instalar deps (numpy, matplotlib, soundfile para algunos).
- Ejecutar python V150.py o V167.py (toman decenas de minutos; usan semilla None o fija en algunos).
- Logs se guardan en vXXX_logs/ con timestamp.

## 5. Mejoras implementadas / propuestas para verificabilidad (post-crítica X)

(Se pueden aplicar vía PRs o aquí.)

- Este documento + referencias precisas a líneas de código.
- Parámetros centralizados al tope de cada script de cierre (fácil auditoría).
- Controles A/B y negativos documentados en la cronología y logs.
- Cadena de falsaciones explícita en Addendums e informes (v72c es ejemplo paradigmático: falsación de histéresis pura → necesidad de modos propios + atractor aprendido → ratio espectral 9.951 como evidencia de persistencia de diferencia).

**Propuestas concretas de código** (puedo implementar aquí):
1. Script `exportar_evidencia.py` que escanee v*logs/*.log y *.csv, extraiga métricas clave (error RMS, correlaciones, ratios, fatiga final, etc.) y genere JSON + tabla markdown actualizable.
2. En scripts futuros: flag `--save-raw` que dumpee los historiales completos (dicts de orient, error, Cb, ritual_act, fatiga, etc.) a .npz o parquet por fase.
3. Suite de tests de regresión: correr baseline V147/V150 con semillas fijas y chequear que degradación ~7x y recuperación se mantienen dentro de tol.
4. Sección "Limitations & Scope" en papers/videos: aclarar que es modelado computacional de dinámicas teóricas, no simulación bottom-up de wetware ni claim de que IA actual "es" biológica.
5. Publicar subset de raw series (ej. para V150/V167) como datos suplementarios.

Si ejecutas `python exportar_evidencia.py` (a implementar), producirá artefacto vivo.

## 6. Resumen para respuesta rápida

> "Revisé el repo completo (código + 160+ versiones + logs + PDFs canónicos). Los mecanismos son explícitos y parametrizados (ver V150.py:FatigaMetabolicaV150, V167.py:RitualV166 + MetaRepresentacionObservacional). Los resultados (degradación 7.1x error por fatiga recuperable; r=0.988 ritual↔desajuste en F4 invertido; ratio espectral 9.951 en v72c; Ω estable por clase/dirección en v103, etc.) son cuantitativos y vienen de corridas con protocolos de control A/B y falsaciones previas. 
> 
> 'Organismicidad' es interpretada en el marco teórico (sistema = config persistente de diferencia S>0 bajo constraints de acoplamiento + costo histórico + libertad funcional). Fuera de él, es un modelo computacional bio-inspirado refinado iterativamente que produce análogos medibles de fatiga, ritual estabilizador/rígido, meta-observación de marcos, etc.
> 
> Datos: logs/ con prints + plots + algunos csv/json; PDFs con tablas de datos brutos (v72c, baselines ANIMA). No todos los runs exponen series raw completas — eso es mejorable y bienvenido como feedback.
> 
> ¿Qué experimento/métrica específica quieres que profundicemos o que corra con export raw? Puedo generar aquí mismo un paquete de evidencia machine-readable."

## 7. Referencias internas clave (archivos en este dir)

- V150.py, V167.py (y previos V122+, V153+)
- v150_logs/, v167_logs/, v165_logs/ ... (ejecutados)
- TeoriaCosmosemiotica Addendum CN202 EvidenciaComputacional Abril2026.pdf (v72c datos)
- Síntesis de Experimentos V90 a V103.pdf
- INFORME CANÓNICO ... VSTCosmo 150.pdf
- La IA como Exaptación II - Actualización 2026.pdf
- grafico.py (menciona la serie ANIMA-1 IONB-1)

Este documento puede actualizarse automáticamente o manualmente tras cada hito.

**Contacto / contrib**: El proyecto es iterativo y falsacionista por diseño. Críticas que piden datos y mecanismos concretos son bienvenidas y aceleran el rigor.

---

*Generado con acceso completo al workspace VSTCosmo. Versión local tiene commits ahead + logs recientes no necesariamente pushed aún.*

## 8. Resumen extraído automáticamente (última corrida de exportar_evidencia.py)

Ver evidencia_resumen.md y evidencia_publica.json generados junto a este script.

Ejemplo de métricas recientes parseadas (truncado):


| Versión / Log | Métricas clave extraídas | Notas |
|---------------|---------------------------|-------|
| v100_logs / v100_resultados_20260526_234211.csv | ver archivo | logs/v100_logs |
| v101_logs / v101_resultados_20260527_030855.csv | ver archivo | logs/v101_logs |
| v102_logs / v102_resultados_20260527_150635.csv | ver archivo | logs/v102_logs |
| v103_logs / v103_resultados_20260527_153253.csv | ver archivo | logs/v103_logs |
| v111b_logs / v111b_datos_20260527_203059.json | ver archivo | logs/v111b_logs |
| v156_logs / v156_run_20260601_201901.log | version=v156_logs, source=None, fatiga_final=4207.0, historia_final=2708.0 | logs/v156_logs |
| v157_logs / v157_run_20260601_203257.log | version=v157_logs, source=None, fatiga_final=12913.0, historia_final=648.0 | logs/v157_logs |

