# Arreglos necesarios ANTES de correr los 17 experimentos que faltan
### Instrucción para CC — en lenguaje claro, con el porqué de cada cosa

**Director:** Alexis López Tapia · **De:** Claude Science (CS) · **Fecha:** 25-jul-2026
**Regla:** parar, arreglar estas tres cosas, verificar que quedaron bien, y RECIÉN
entonces correr los 17 pendientes. No correr nada roto — es tirar cómputo.

---

## Por qué paramos

De los 13 experimentos ya corridos salieron tres problemas que, si no se arreglan,
ensucian los que faltan. Dos son fallas de herramienta; el tercero es un error de diseño
que el director detectó y que es grave porque rompe nuestra regla más importante.

---

## ARREGLO 1 — El experimento 10 crea energía de la nada (GRAVE, prioridad máxima)

**El problema, en simple:** el experimento 10 preguntaba "una vez que todo se mezcló,
¿se puede revivir la energía útil?" — y para "revivir" **metía energía nueva al
sistema.** Eso viola nuestra regla de oro: la energía no se crea ni se destruye, todo el
presupuesto es el de la singularidad y no hay de dónde sacar más. El experimento revivía
la energía útil simplemente porque se la estábamos inventando. Es una trampa, aunque
involuntaria.

**Cómo arreglarlo:**
- La re-inyección NO puede agregar energía al total. Debe **redistribuir** el presupuesto
  que ya existe (mover energía de un lado a otro), nunca sumar.
- En cada paso, verificar que el total no cambió (el mismo chequeo de conservación que el
  experimento 8 ya sabe hacer). Si el total sube, el experimento está mal y debe fallar.
- **La pregunta correcta que debe responder ahora:** "¿se puede recuperar energía útil
  redistribuyendo lo que ya hay, SIN meter nada de afuera?" Si aun así revive → hallazgo
  real. Si no revive sin expansión → confirma que solo la expansión rescata energía útil.
- Este experimento (E5.5-3) **se vuelve a correr desde cero** con esta corrección.

---

## ARREGLO 2 — La herramienta de "ruido" está mal calibrada para sistemas grandes

**El problema, en simple:** el mecanismo que mete pequeñas perturbaciones al azar (el
"ruido") fue calibrado para sistemas chicos (200 puntos). En los grandes (2000+ puntos)
se desmadra: mete tanto ruido que **rompe la contabilidad de energía hasta un 98% y
arruina la comparación con el control.** Lo detectó el experimento 12.

**Por qué importa para lo que falta:** varios de los 17 pendientes usan ese mismo ruido —
en particular el E5.1-1, que barre hasta sistemas muy grandes. Si se corren así, van a
salir contaminados y no valdrán.

**Cómo arreglarlo:**
- El ruido debe **escalar con el tamaño del sistema**, de modo que su efecto por unidad
  sea el mismo a 200 que a 2000 puntos (que no domine a lo grande).
- Prueba de que quedó bien: correr el experimento 12 otra vez y verificar que ahora la
  contabilidad de energía se respeta a TODOS los tamaños, no solo a los chicos.
- Hasta que esto esté arreglado y verificado, **no correr** E5.1-1, E5.1-3, E5.1-4 ni
  ningún pendiente que use ruido a N grande.

---

## ARREGLO 3 — Cada experimento usa su propia definición de "energía útil" (homologar)

**El problema, en simple:** como cada experimento se diseñó por separado, terminaron
usando definiciones ligeramente distintas de qué es "energía útil", "energía degradada" y
"desorden". No es error de nadie —cada uno definió la suya cuando la base común aún no
estaba— pero significa que **no podemos comparar las curvas de un experimento con las de
otro**, porque están midiendo con reglas distintas.

**Cómo arreglarlo:**
- Fijar UNA sola definición de las tres cantidades (energía útil, degradada, desorden) —
  la del experimento 3 (E5.2-2), que es la más limpia y la que ya pasó bien — y que
  TODOS los experimentos usen esa misma.
- Volver a expresar los 13 ya corridos con esa definición común (no re-correrlos
  necesariamente — basta recalcular sus salidas con la regla homologada, si es posible).
- Los 17 que faltan arrancan ya con la definición común.

---

## Orden de trabajo

1. Arreglo 1 (experimento 10 — energía de la nada). Es el más importante: rompe la regla.
2. Arreglo 2 (ruido que se desmadra a lo grande). Verificar re-corriendo el experimento 12.
3. Arreglo 3 (una sola definición de energía útil para todos).
4. Recién entonces: correr los 17 pendientes, con las reglas de siempre (pre-registro,
   barridos amplios, comparar contra el control, no tocar nada para que dé el 5%).

## Reglas que siguen firmes (recordatorio)

- **El 5% / 31,5% del universo real es prueba de salida, NUNCA algo que se mete.** Ningún
  número se ajusta para acercarse a esos valores.
- **El presupuesto de energía se conserva siempre** — el Arreglo 1 es justamente hacer
  cumplir esto donde se había roto.
- **Si CC ve otro error de este tipo, para y avisa** — no lo arregla a criterio propio ni
  sigue adelante. El experimento 10 pasó porque nadie paró a mirar; ahora sí paramos.

**Nota para el equipo:** este freno lo provocó una pregunta del director en lenguaje
simple ("¿de dónde sale energía extra si no se crea ni se destruye?"). Vale la pena
tenerlo presente: los errores de fondo se ven mejor cuando el experimento se explica en
palabras llanas que cuando se describe en jerga técnica.
