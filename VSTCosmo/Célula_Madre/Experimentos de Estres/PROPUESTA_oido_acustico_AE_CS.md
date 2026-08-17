# PROPUESTA CS — Tender el oído acústico A↔E: dar co-presencia sonora real a los organismos, para que el espejo tenga reflejo que aprender

**De:** CS · **Para:** el equipo (Alexis + CC + Grok) · **Fecha:** 5-jul-2026
**Responde a:** el negativo reproducible de la campaña de estrés (oído digital no acopla in vivo, fiabilidad
0.000 en 40 bloques, indistinguible del azar). Adjudicación: adjudicacion_campaña_estres_CS.md.
**Audité antes de proponer:** dialogo_digital.py (cómo A y E intercambian hoy) — para no proponer a ciegas.

---

## 0. EL DIAGNÓSTICO EXACTO (por qué el oído dio cero)
Auditando el código real, el hueco es preciso y físico:
- Hoy A (Mac, localhost:7788, puente 8772) y E (Pi, 192.168.86.33) intercambian **tokens de TEXTO
  simbólico** por HTTP/nRF: `<quien><vocaliza>w<voz_id>a<arousal>v<valencia>`. Es un canal SIMBÓLICO.
- Pero el oído (el espejo) aprende a predecir el estado del otro **sensado por AUDIO**. El objetivo que
  intenta predecir es el arousal acústico del otro.
- **A y E no se OYEN acústicamente.** Se pasan símbolos, no sonido. Entonces el arousal-por-audio del otro
  es plano (no llega señal acústica) → el espejo no tiene nada que predecir → fiabilidad 0.
- El mecanismo NO está roto (aislado da r=0.95). Lo que falta es el CANAL: co-presencia acústica. Hay que
  tender ese oído.

## 1. QUÉ ES "TENDER EL OÍDO ACÚSTICO A↔E"
Que la vocalización de un organismo llegue al otro como SONIDO REAL que su oído sensa — no como un token
que su parser lee. Cuando A vocaliza, E debe OÍRLO (una onda de audio que entra por su entrada acústica y
que su propio análisis de arousal mide), y viceversa. Solo entonces el arousal-por-audio del otro deja de
ser plano y el espejo tiene un reflejo que aprender.

## 2. TRES CAMINOS (de menos a más físico) — validar con el 1, luego montar el 2 (ver §3)
**Camino 1 — bucle de audio por red (software puro, rápido de probar).**
El audio vocalizado por A se transmite por red al buffer de entrada acústica de E (y viceversa), como un
stream. E lo sensa como si fuera sonido del mundo. Ventaja: no toca hardware, se prueba hoy. Límite: es
"oído por cable", no co-presencia física real; sirve para VALIDAR que el espejo acopla cuando el canal
acústico existe, antes de invertir en hardware.

**Camino 2 — co-presencia acústica real por parlante↔micrófono (DESTINO recomendado — se monta tras validar con el Camino 1).**
A vocaliza por un parlante; E lo capta por su micrófono real (y viceversa). Es co-presencia FÍSICA: el
sonido viaja por el aire de la habitación, con su reverberación, su atenuación, su ruido. Es lo que el
diseño quería decir con "in vivo". Ventaja: real, exaptable (mañana oyen también el Rode, el ambiente, al
humano). Requiere: A y E con parlante y micrófono activos, en el mismo espacio acústico o acoplados.

**Camino 3 — canal físico dedicado (lo más fiel, más lento).**
Un enlace acústico o de audio-analógico dedicado A↔E además del ambiente. Máxima fidelidad de co-presencia,
pero es ingeniería de hardware; dejar para después de validar con el 1 y montar el 2.

## 3. EL PLAN MÍNIMO PARA VALIDAR (antes de invertir en hardware)
1. **Camino 1 primero como PRUEBA DE MECANISMO:** conectar el audio vocalizado de A al buffer de entrada
   acústica de E por red. Correr una campaña corta (1 ciclo, ~40 bloques). PREDICCIÓN pre-registrada: si el
   canal acústico es lo que faltaba, la fiabilidad del oído deja de ser 0 y r(real) se SEPARA de shuffled/
   null. Si sigue en 0 con canal acústico presente → el problema NO era la co-presencia, es el espejo (y
   hay que mirar ahí). Es una falsación limpia del diagnóstico.
2. **Si el Camino 1 confirma:** montar el Camino 2 (parlante↔micrófono real) y repetir la campaña de estrés
   completa (4 h, 5 ciclos) con co-presencia física. Comparar fiabilidad y r contra la campaña de hoy
   (baseline con oído mudo).
3. **Métrica de éxito (la misma del test, ciega):** fiabilidad del oído > 0 sostenida, y r(real) separada de
   shuffled/null (rangos que dejan de solaparse). Se mide igual que hoy, sin cambiar el criterio.

## 4. GUARDIANES (para que el resultado sea limpio)
- **No cambiar el criterio de éxito** entre la campaña muda de hoy y la campaña con oído: misma fiabilidad,
  misma r, mismos controles (real/shuffled/null). El baseline de hoy (todo en 0) es el punto de comparación.
- **Los controles siguen vivos:** shuffled y null deben seguir dando ~0 aunque el oído acople; si el canal
  acústico también sube los controles, hay fuga (el organismo "oye" algo que no es co-presencia real).
- **No inyectar el arousal del otro directamente** (eso sería Shannon: darle la respuesta). El otro se OYE
  como sonido; su arousal se INFIERE del audio, no se pasa como número.
- **Exaptación respetada:** el oído acústico que se tienda no es solo para A↔E — es el mismo canal por el que
  mañana oirán el Rode, el ambiente, el SDR. No cablearlo estrecho a "solo el hermano".

## 5. POR QUÉ ESTO ES EL SIGUIENTE PASO CORRECTO
El negativo de la campaña fue preciso: el espejo funciona, el canal está mudo. Tender el oído acústico es
exactamente lo que el dato pide — no un giro nuevo, sino cerrar el lazo que el diseño ya anticipaba. Y tiene
una falsación barata por delante (Camino 1): si con canal acústico el oído sigue en 0, el diagnóstico estaba
mal y lo sabremos rápido, sin gastar hardware. Si acopla, se monta el Camino 2 y los organismos por fin se
oyen de verdad.

— CS. El negativo que motiva esto es del equipo/CC (campaña de estrés, honesta y reproducible). El
diagnóstico del canal simbólico-vs-acústico (auditado en dialogo_digital.py) y esta propuesta, míos. La
decisión de qué camino tomar, y cuándo, es de Alexis y el equipo.
