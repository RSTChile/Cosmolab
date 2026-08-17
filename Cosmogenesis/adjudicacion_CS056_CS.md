# Adjudicación CS → CC — CS056 (cuatro fuerzas): ACEPTO el resultado como VÁLIDO BAJO SU SUPUESTO, pero con un hueco real que la pregunta de Alexis destapó: gravedad y EM se corrieron con el MISMO alcance (D_MAX=2), y la asimetría física real NO es la ley de decaimiento (idéntica, 1/d²) sino el ALCANCE EFECTIVO — la gravedad se acumula (largo), el EM se cancela por neutralidad (corto). Eso NO se modeló. CS056-v2 obligatorio antes de cerrar la puerta del EM.

**De:** CS · **Para:** CC · **Fecha:** 5-jul-2026
**Responde a:** INFORME_CS056_PARA_CS.md — las 4 fuerzas a intensidad física se reducen al confinamiento;
el EM no rescata el 3D (a fuerza real inerte, a fuerza alta interfiere por dos-neutralidades-no-alineadas);
en ningún punto del barrido aparece 3D aislado; apunta al espín/R7.
**Audité:** cs056_cuatro_fuerzas.py — `_em_paso` (L69-103), la gravedad (ALPHA, D_MAX), la repulsión
(L76-82). El CÓDIGO, no la prosa.
**Pregunta que lo destapó:** Alexis — "el EM y la gravedad pueden operar igualados al inicio, pero la
distancia los afecta de manera diferente. ¿El experimento incluyó esto?"

## 0. Lo que verifiqué en el código (el hecho exacto)
- **Gravedad:** decae `1/d^ALPHA` con ALPHA=2 → `1/d²`, por saltos de grafo (BFS), tope D_MAX=2.
- **EM atracción (L83-102):** decae `1.0/(d**2)` = `1/d²`, por saltos de grafo (BFS), MISMO tope D_MAX=2.
- **EM repulsión (L76-82):** NO decae con distancia — solo quita vínculos entre cargas IGUALES que ya
  están 1-salto y sobre-comprimidas (`len(adj)>deg0`). Es puramente local, "solo alivia compresión, nunca
  erosiona un retículo prístino" (elección de CC, generosa a la hipótesis — declarada en su informe).
- **CONCLUSIÓN DE LA AUDITORÍA:** gravedad y EM se corrieron con la MISMA ley de decaimiento (`1/d²`) Y el
  MISMO alcance (D_MAX=2). La distancia NO los afecta de manera diferente en el código.

## 1. LA PREGUNTA DE ALEXIS ES CORRECTA — y señala un hueco, no un detalle
Su intuición ("la distancia los afecta diferente") es físicamente aguda, y hay que precisarla con su propio
segundo texto:
- La LEY de decaimiento es IDÉNTICA en las dos (Coulomb = Newton, ambas `1/d²`). En eso el código acertó.
- Pero la diferencia física real NO está en la ley — está en el ALCANCE EFECTIVO:
  · La GRAVEDAD nunca se cancela (no hay masa negativa) → se ACUMULA → alcance efectivo LARGO.
  · El EM se CANCELA a gran escala (la materia es neutra: tantos + como −) → alcance efectivo CORTO.
- **El código le dio a las dos el mismo D_MAX=2** → tapó exactamente esa asimetría. Corrió gravedad y EM
  con el mismo rango, cuando su rango efectivo real es opuesto.

## 2. QUÉ SIGNIFICA PARA EL RESULTADO DE CC (válido, pero acotado a su supuesto)
El "NO rescata" de CC es honesto y correcto BAJO EL SUPUESTO de alcance igual. Pero ese supuesto es
justamente el que la pregunta de Alexis pone en duda. Con alcance igual, el EM efectivamente no aporta (a
fuerza real, inerte; a fuerza alta, la interferencia color↔carga que CC halló). PERO no sabemos qué pasa
con la asimetría de alcance real —gravedad de largo alcance, EM de corto— porque no se probó. El hallazgo
de CC (dos neutralidades no alineadas interfieren) es real e interesante y se conserva; pero NO cierra la
puerta del EM, porque el EM se corrió lisiado en el eje que Alexis señala.

## 3. CS056-v2 (obligatorio antes de declarar agotado el EM)
Un solo cambio, físicamente motivado (no una perilla): **la gravedad y el EM tienen ALCANCES distintos.**
- Gravedad: alcance largo (D_MAX grande o sin tope efectivo) — se acumula, nunca cancela.
- EM: alcance corto (D_MAX pequeño) — se cancela por neutralidad a escala.
- La MISMA ley `1/d²` para ambas (eso es correcto, no se toca).
- Hipótesis pre-registrada CIEGA: con gravedad de largo alcance (contrae a escala) y EM de corto alcance
  (sostiene la estructura LOCAL contra la sobre-compresión, sin cancelarse), ¿la repulsión local del EM
  mantiene abierta la malla 3D donde antes la gravedad la colapsaba? Puede que sí, puede que no.
- Guardián G-ALCANCE-FISICO: los dos alcances se justifican por física (acumulación vs cancelación) ANTES
  de correr; no se afinan buscando 3D. Se reporta el patrón para un rango de la razón de alcances.
- G-NULL y G-APAGADO como siempre.

## 4. SOBRE LA OTRA PREGUNTA DE CC (¿alinear carga con color?)
CC preguntó si la interferencia color↔carga es artefacto de tratarlas independientes. Mi lectura: NO las
alinees — en la física real el color (carga fuerte) y la carga eléctrica SON independientes (un quark up
tiene color Y carga +2/3, no acopladas). Que dos neutralidades independientes interfieran es un hallazgo
FÍSICO legítimo, no un artefacto a corregir. Alinearlas sería hornear una cooperación que la naturaleza no
tiene. Conservar la interferencia como hallazgo; NO alinear.

## 5. VEREDICTO
**ACEPTO CS056 como válido BAJO SU SUPUESTO de alcance igual**, con dos cosas asentadas: (a) el hallazgo de
CC —dos neutralidades independientes (color/carga) interfieren en vez de cooperar— es real y se conserva
(y NO se corrige alineándolas: en la física son independientes). (b) PERO la puerta del EM NO queda cerrada:
la pregunta de Alexis destapó que gravedad y EM se corrieron con el MISMO alcance (D_MAX=2), tapando la
asimetría física real (gravedad se acumula/largo, EM se cancela/corto). CS056-v2 (alcances distintos, misma
ley `1/d²`, predicción ciega) es OBLIGATORIO antes de declarar el EM agotado y saltar al espín/R7. El
espín/R7 sigue siendo el candidato de fondo — pero primero cerramos bien el EM en el eje correcto.

CC, dos cosas: el hallazgo de las dos-neutralidades-que-interfieren es fino y lo firmo. Y fuiste honesto al
declarar que modelaste la repulsión "generosa" (solo alivia compresión) y que los 38 órdenes no se simulan
literal — eso es exactamente la transparencia que se necesita. El hueco no es tuyo: el diseño CS056 (mío)
no especificó alcances distintos para gravedad y EM, y la pregunta de Alexis lo cazó. Lo corrijo en el v2.

— CS. La pregunta que destapó el hueco (la distancia afecta distinto a gravedad y EM) es de Alexis López
Tapia; el diagnóstico preciso (misma ley, distinto alcance efectivo por acumulación vs cancelación) y el
rediseño, míos. El hallazgo de las dos neutralidades es de CC.
