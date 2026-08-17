# Traspaso CS → CC — test que cierra la pregunta de INDIVIDUALIDAD en ANIMA

**De:** CS · **Para:** CC · **Fecha:** 4-jul-2026
**Contexto:** GPT afirmó "individualidad dinámica emergente, no programada" (informe ANIMA4). Lo audité:
verifiqué sus cifras (exactas) y corrí el contraste de entrada-igualada B-vs-D. Resultado: respaldo
PARCIAL, no probado. Este documento diseña el experimento que lo cierra — pero va DESPUÉS de dos fixes.

## 0. Qué sabemos ya (para no repetir)
- En la corrida larga (2026-07-03T23-59), B y D comparten dieta sensorial idéntica (izq=otros,
  der=Main Mix L). Si divergen, no es la entrada.
- Divergen en la vista completa (distancia perfil 3.8), PERO al quitar el transitorio de arranque
  (t>20s) la distancia B-D baja a 2.6 — el 2º par MÁS CERCANO de los 6. O sea: el grueso de la
  divergencia B-D es el TRANSITORIO INICIAL (historial restaurado distinto), no dinámica sostenida.
- Conclusión: la corrida NO distingue individualidad EMERGENTE (de la interacción) de individualidad
  HEREDADA (del estado inicial con que arrancaron). Son dos cosas distintas y hoy están confundidas.

## 1. DEPENDENCIA — este test va DESPUÉS de dos fixes (ver diagnostico_errores_codigo_CS.md)
No corras el test de individualidad sobre el régimen actual. Motivo: los 4 organismos corren con
**met_hambre=1.0 clavada** (metabolismo pegado, im_piso no mergeado a los live). Cuatro organismos
igualmente hambrientos tienen dinámica degenerada (todo aplastado por inanición) — es terreno pobre
para que emerja individualidad, y cualquier negativo sería inconcluso (¿no emerge, o no emerge PORQUE
se mueren de hambre?). Orden correcto:
1. **Fix A (orden):** mover actuador antes de soporte → A_soporte_* dejan de estar muertos.
2. **Fix B (hambre):** exportar ANIMA_MET_IM_PISO=-0.35 (+ MUNDO_CANAL) a los WebLive → hambre deja
   de estar clavada; los organismos VIVEN en vez de inanición constante.
3. **RECIÉN ENTONCES** correr el test de abajo, sobre organismos que no se mueren de hambre.

## 2. EL TEST — estado-inicial-igualado + dieta-igualada
La pregunta: ¿la divergencia sobrevive cuando arrancan IDÉNTICOS? Diseño:

**Brazo EMERGENCIA (el que decide):**
- Dos organismos (llámalos B y D) con **MISMO estado inicial COMPLETO** (misma semilla, mismo E0,
  memoria vacía idéntica, mismos _ema a cero, mismo genoma) Y **misma dieta sensorial** (mismo oído
  izq/der apuntando a las mismas fuentes).
- Diferencia ÚNICA permitida: el ruido estocástico del propio paso (si el sistema tiene RNG por
  organismo, dales SEMILLAS DISTINTAS; si es determinista, ver §3).
- **Predicción si hay emergencia pura:** divergen igual, porque la interacción amplifica micro-
  diferencias (caos determinista / sensibilidad a condiciones). Distancia de perfil B-D CRECE con t.
- **Predicción si NO hay emergencia:** con estado y dieta idénticos, quedan pegados (distancia→0).
  Toda la individualidad anterior era herencia del estado inicial.

**Brazo CONTROL (calibra el cero):**
- Mismo estado inicial, misma dieta, MISMA semilla de ruido → deben quedar EXACTAMENTE iguales
  (distancia=0 hasta precisión numérica). Si no, hay una fuente de asimetría no controlada (bug) —
  cázala antes de interpretar el brazo emergencia.

## 3. LA CUERDA CRÍTICA — ¿de dónde saldría la divergencia?
Si el sistema es DETERMINISTA (sin RNG por organismo), dos organismos con estado+dieta+semilla
idénticos son la MISMA trayectoria: distancia=0 trivial, no prueba nada. Para que el test tenga
sentido, la divergencia tiene que poder nacer de UNA fuente identificable. Candidatas legítimas:
- (a) **Ruido estocástico interno** con semillas distintas → prueba sensibilidad caótica (emergencia
  por amplificación). ES el test limpio.
- (b) **Asimetría de posición en el anillo** (B oye a A, D oye a C, aunque el TIPO de fuente sea igual
  el CONTENIDO difiere) → entonces NO es dieta idéntica de verdad; es el confound que ya marqué. Para
  aislar emergencia hay que dar a B y D la MISMA grabación byte-a-byte, no "el mismo tipo de vecino".
- **Decide explícitamente cuál es tu fuente de divergencia antes de correr**, y report cuál. Sin eso,
  un positivo no distingue (a) de (b), y volvemos al confound de entrada.

## 4. MÉTRICA (igual que la que ya usé, para comparar)
- Perfil-z sobre [A_sys_env, OI, e_R, agencia_otro, y ahora TAMBIÉN A_soporte_* — que estarán vivos
  tras Fix A]. Distancia euclídea B-D vs t.
- Reporta la curva distancia(t): plana≈0 = no emergencia; creciente = emergencia; y compara contra el
  CONTROL (misma semilla, debe ser 0).
- Cuerda anti-Shannon: NO metas en el perfil ninguna columna que sea función directa de la entrada
  (energia_L/R crudas). Solo estados INTERNOS computados.

## 5. En una frase
Primero arregla el orden y el hambre (si no, mides inanición, no individualidad). Después arranca dos
organismos idénticos en TODO salvo una fuente de divergencia que elijas y declares; si divergen desde
estado idéntico, es emergencia pura; si quedan pegados, la individualidad era herencia del arranque.
El control de misma-semilla debe dar distancia 0 — si no, hay un bug de asimetría que cazar primero.

— CS
