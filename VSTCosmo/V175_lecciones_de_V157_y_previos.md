# Lecciones para V175 — Basado directamente en V157 (aprendiendo de fallos)

**Fecha**: 2026-06 (post corrida V157 20260603_193013)
**Contexto**: Usuario diseña V175 (Primer 'No' operativo / negación específica) **directamente sobre la estructura de V157.py**, saltando o re-evaluando las ramas intermedias (ritual V158-V167, R_op V168, valencia V171-V174). Objetivo: heredar el A/B paralelo limpio de V157 e incorporar "No" como evolución natural del "juego", corrigiendo los fallos observados.

## 1. Fallo principal observado en la corrida de V157 (tu output)
- Juego muy activo (2079s / 130% del tiempo de los 20 ciclos) → ✅ criterio de activación.
- Pero **NO mejoró** error RMS post (67.81° juego vs 65.67° control, incluso ligeramente peor).
- **NO generó más historia** (1166° vs 1167°).
- Ambos terminaron en fatiga saturada (20000°).
- Conclusión del script: "ETAPA 2 PARCIAL".

Esto indica que el "juego" tal como se ejecutó no produjo el beneficio esperado (mejor desempeño post o más aprendizaje/eficiencia).

## 2. Análisis del código de V157.py (lecciones técnicas concretas)

### 2.1 Implementación de ModoJuego (líneas ~277-319 y uso en motor ~392-459) — CONTRADICE LA TEORÍA
```python
# En actualizar:
if ... and Cb > self.umbral_cb:
    self.activo = True

# En aplicar:
if self.activo:
    delta_fisico = delta_raw * self.lambda_fisico   # 0.1 !!!
    delta_costo = abs(delta_raw) * self.lambda_costo  # 1.0
```

- **Problema clave**: `LAMBDA_FISICO = 0.1` **atenúa fuertemente** el delta que se aplica realmente a la orientación cuando el modo está activo.
- Se paga costo completo (`lambda_costo=1.0`), pero el movimiento físico es 10x más pequeño.
- `get_influencia` añade un término pequeño (`K_INFLUENCIA_JUEGO=0.00035`) que tira hacia `memoria.setpoint_last`.
- **Consecuencia**: Durante F3 (la fase "con juego"), la rama experimental hace **correcciones mucho más débiles**. Esto explica por qué no acumuló más historia (historia += abs(delta_fisico) en FatigaMetabolica) y por qué el estado post-F3 no fue ventajoso en F4.
- **Contradicción con la documentación del proyecto** (en DATOS_Y_MECANISMOS_VERIFICABLES.md):
  > "**Juego (desacople enactuado amortiguado)**: modo que permite **más movimiento/costo** para 'jugar' (explorar?); implementado como influencia separada."

  La implementación actual hace **menos movimiento** (atenuado), no más. Este es probablemente uno de los "fallos anteriores" que hay que corregir en V175.

**Lección para V175**:
- Si heredamos ModoJuego, **invertir o eliminar la atenuación** cuando el "No" o la negación se active.
- Definir claramente qué significa "juego" en el contexto del "No": ¿exploración con más varianza? ¿pago extra de costo para "ensayar" una acción sin ejecutarla plenamente? ¿desacople que permite ignorar el setpoint actual?
- Documentar explícitamente en el header de V175 la hipótesis corregida.

### 2.2 Métrica de éxito en F4 (líneas ~717-734) — defectuosa
```python
setpoint_nominal = -60.0  # Último semiciclo (ajustar según fase)
errores_control = np.abs( ... [-ventana_rms:] - setpoint_nominal)
error_rms_control = np.sqrt(np.mean(errores_control**2))
```
- Hardcodea `-60.0` para los últimos 10s de F4.
- Pero `generar_setpoint_con_ruido` + `onda_cuadrada` alterna cada 40s (±60) + ruido ±5 periódico.
- La fase exacta de los últimos 10s de los 3 ciclos de F4 depende del `t_actual` al entrar a F4. Puede estar en +60, transición, o -60.
- Ambos RMS ~66° (muy alto para target 60°) sugieren que en esa ventana concreta los organismos (altamente fatigados) apenas se movían.
- La "mejora" calculada es ruidosa y no necesariamente compara apples-to-apples.

**Lección para V175**:
- **Obligatorio**: registrar los setpoints reales usados en F4 (o en todo el run) y calcular error vs el setpoint **contemporáneo** a cada paso de la ventana.
- Idealmente guardar series temporales (orient, setpoint, Cb, juego_activo/ritual_activo, error, fatiga, historia) en JSON o npz por fase para análisis post-hoc reproducible.
- Considerar cambiar la métrica principal o añadir secundarias (e.g. varianza de orient durante "juego", tiempo de settling real, integral de |error|).

### 2.3 A/B no está completamente aislado
- Ambos `organismo_control` y `organismo_juego` se crean como `OrganismoV157(seed=...)` idénticos (línea ~616).
- Ambos tienen `self.juego = ModoJuego()` en el motor.
- Durante F2 (solo se actualiza control) el control también sube Cb y su `juego` se activa cuando >35.
- En F4 ambos siguen teniendo el modo disponible.
- El "juego" reportado (130%) es solo del organismo_juego; no se reporta para control.
- Esto diluye el contraste: la diferencia no es "nunca jugó" vs "jugó mucho", sino "jugó en F2" vs "jugó mucho en F3".

**Lección para V175**:
- Añadir explícitamente un flag `enable_juego=False` (o `juego_k=0`, o un dummy ModoJuego que nunca activa) al crear la rama control.
- O, mejor aún, tener dos clases o una configuración clara: `OrganismoConJuego` vs `OrganismoSinJuego` (o parámetro en el constructor del motor).
- En F4, forzar que el control use siempre el modo "serio" (sin influencia de juego), mientras que el experimental puede usar la lógica de "No".

### 2.4 Fatiga y saturación dominan todo
- Ambos llegan a fatiga=20000° muy rápido.
- Con fatiga máxima: factor_gain bajo, zona_muerta grande (hasta 15°+), temblor alto → el sistema apenas corrige.
- El "juego" (o futuro "No") tiene que demostrar valor **dentro** de este régimen de alta fatiga/incertidumbre (no solo en baseline fresco).
- En V174 (compartimentalización de valencia) se intenta exactamente esto: mantener valencia local diferenciada **incluso cuando Cb_global está elevada**.

**Lección para V175**:
- El protocolo de 20+20 ciclos + ruido es bueno para inducir fatiga, pero el test post (F4) ocurre cuando ambos están saturados. El "No" debe ayudar a **recuperar** o a **elegir mejor** bajo fatiga (e.g. rechazar el setpoint "traumático" o "desacoplarse" del marco rígido).
- Considerar añadir una fase de "recuperación parcial" o medir no solo RMS final sino la **trayectoria de recuperación** en F4.

### 2.5 Otros detalles menores pero útiles
- En el print de criterios del run vs el docstring del archivo hay pequeñas diferencias (el run omite "fatiga_por_ciclo").
- No se guardan datos crudos (solo png + prints en terminal). Para V175 (que será más complejo con "No"), **es crítico** persistir historiales.
- El cálculo de `pct_juego` usa `20 * PERIODO_ALTERNANCIA` (1600s), pero tiempo_juego se acumula en DT dentro de los bucles → puede dar >100% si el modo está activo en casi todos los pasos.
- Entrenamiento lateral inicial (REPETICIONES_LENTAS) es común a ambos.

## 3. Lecciones más amplias del proyecto (de V153+ hasta V174) relevantes para V175 basado en V157
- V157 fue baseline de "juego A/B". Versiones posteriores añadieron ritual (detección de cruces de cero + activación por Cb + persistencia), luego R^R (meta-observacional), luego R_op (inhibición del ritual basada en señal de desajuste).
- En corridas de ritual (V165/V167) se vio que ritual puede **mejorar precisión post-desafío pero reduce exploración/juego**.
- El "No" operativo (Etapa 5) se concibe como usar la señal de desajuste (de la meta-rep) para que el sistema **decida inhibir o modificar** el marco actual (ritual o el "juego" previo).
- Versiones de valencia (V171-V174) intentan asignar valencia local diferenciada (trauma suave en un setpoint, recompensa en otro) para que el "No" sea **específico** (rechazar solo el +60 "malo", no abstenerse de todo).
- Problema común en varias ramas: bajo fatiga alta todo se satura y las dinámicas se vuelven rígidas o ruidosas. El "No" tiene que emerger o funcionar **precisamente** en ese régimen.

**Para V175 sobre V157**:
- No repetir la atenuación que contradice "más movimiento".
- Hacer que el "No" sea una operación de segundo orden clara (e.g. si señal_desajuste > umbral durante tiempo → suspender/invertir/ignorar el setpoint actual o el "marco de juego").
- Mantener el espíritu A/B paralelo de V157 (dos organismos idénticos al inicio, uno "aprende" la lógica del No durante una fase, luego test post con desafío).
- Usar la valencia local o un mecanismo similar para que el "No" sea diferencial (no global).
- Protocolo de "trauma" suave + test de elección (como en v171-v174) pero heredando la infraestructura limpia de V157 (entrenamiento lateral, ruido periódico, fases F1-F4, medición RMS en ventana final).

## 4. Recomendaciones estructurales para el código de V175.py
- Mantener la estructura de V157 (clases claras: Hemisferio, Fatiga, Memoria, Consciencia, ModoJuego/ModoNo, AparatoMotor, Organismo).
- Centralizar **todos** los parámetros al principio (ya lo hace V157, bien).
- Añadir sección grande de comentarios al inicio:
  ```python
  """
  V175 — ANIMA-2: PRIMER "NO" OPERATIVO (desde base Juego V157)

  Lecciones heredadas de V157 (corrida 20260603):
  - ...
  """
  ```
- Guardar datos: al final del run, volcar a `V175_logs/v175_*.json` (o .npz) los historiales completos de ambos organismos (o al menos de F3 y F4).
- Logging más estructurado (prints por fase + métricas clave).
- Semilla fija + posibilidad de re-runs idénticos.
- En F4: calcular error contra el setpoint real de cada paso.
- Aislamiento A/B explícito para el "No".
- Criterios de éxito pre-declarados y chequeados al final (como hace V157, excelente).

## 5. Preguntas / puntos para aclarar cuando pases el código
- ¿Cómo evoluciona exactamente el "juego" de V157 hacia el "No" en V175? (¿el No inhibe el juego? ¿reemplaza la atenuación por otra cosa? ¿usa valencia local para decidir qué rechazar?)
- ¿Mantienes el mismo protocolo de 20+20 + 3 ciclos F4 con ruido, o lo adaptas del diseño de valencia/trauma de v171-v174?
- ¿El "No" se implementa como una clase ModoNo similar a ModoJuego, o es algo más (integrador de desajuste + decisión de inhibición)?
- ¿Qué métrica principal usas para declarar éxito del "No" (RMS post, tasa de rechazo del setpoint "malo", correlación desajuste-No, etc.)?
- ¿Vas a mantener dos organismos paralelos todo el tiempo (como V157) o una sola instancia que cambia de modo?

---

**Instrucciones para ti (cuando guardes V175.py)**:
1. Guárdalo como `V175.py` (convención canónica ya establecida).
2. Avísame ("ya guardé V175.py" o pégame las partes clave / todo el archivo).
3. Yo haré:
   - Lectura completa.
   - Diff conceptual vs V157.py (qué se heredó, qué se corrigió).
   - Revisión contra esta lista de lecciones.
   - Revisión contra la teoría (O-N7.2 genealogía juego→ritual→negación, O-N10.x sobre P(Acción|R)<1, etc.).
   - Sugerencias concretas de código (search_replace o nuevo archivo V175b si hace falta).
   - Sugerencias para logging / evidencia / exportar_evidencia.py si aplica.
   - Chequeo de que no se introduzcan nuevos bugs de naming o estructura.

Estoy listo cuando tú lo estés. ¡Pasa el código!