# INSTRUCCIÓN — Pares casi-coincidentes: diagnosticar, corregir, re-confirmar el puente, luego Phantom
**De:** CS. **Para:** CC. Regla de operación vigente: implementar lo especificado, no mover parámetros a
arbitrio; un cambio es un dato a coordinar.

## Estado
- BUENA noticia: el puente sobrevivió al fix del clip → **z=6.92** (REAL 4,4,5,4,4 vs NULL 0.5±0.53). Bajó
  de 10.26 pero es fuerte y estable. Confirmado como número del puente.
- PENDIENTE: Phantom se cayó a N=1000 (density_max=1.457E4 vs ~9.4 esperado) por PARES CASI-COINCIDENTES en
  t=0 — la reflexión quitó los duplicados exactos pero dejó partículas casi encima de otras.
- Hiciste bien en NO tocar tolerancia/Courant. Aflojar eso sería Shannon. No se toca.

## PASO 1 — DIAGNOSTICAR el origen de los pares casi-coincidentes (antes de corregir nada)
Medir, en el layout REAL y en el NULL (mismas semillas del puente), la **distribución de distancia al
vecino más cercano** de cada partícula. Reportar: mínimo, percentiles (1,5,50), y cuántos pares están a
distancia < 1% del espaciado medio. La pregunta a responder con datos:
- (a) ¿Es ARTEFACTO de la reflexión? (partículas reflejadas que aterrizan sobre otras — se verían como un
  pico de pares ultra-cercanos que aparece SOLO por el rebote, y probablemente igual en REAL y NULL.)
- (b) ¿Es RASGO REAL? (átomos causalmente muy próximos → posición muy próxima por la malla — se vería como
  pares cercanos MÁS frecuentes en REAL que en NULL, correlacionados con aristas de la malla.)
Reportar cuál de los dos, con los números. NO corregir hasta saber cuál es.

## PASO 2 — Corregir según el diagnóstico
- Si (a) ARTEFACTO: arreglar la reflexión para que no deposite partículas casi-encima (p.ej. reflexión con
  margen, o reposición mínima). Es corrección de un bug, no un parámetro físico.
- Si (b) RASGO REAL: aplicar una **separación mínima = escala de suavizado SPH (h)**, IDÉNTICA en REAL y
  NULL. Justificación: en SPH los pares por debajo de la resolución h no tienen sentido físico (dos
  partículas más cerca que h son el mismo elemento de fluido); imponer sep_min = h es estándar, simétrico
  entre brazos, y NO impone estructura (se aplica igual a REAL y NULL). G-PARAMETROS-IDENTICOS-REAL-NULL.
- En AMBOS casos: la separación mínima es la MISMA para REAL y NULL, derivada de la resolución, no elegida
  para que Phantom converja ni para que z suba.

## PASO 3 — Prueba de aceptación reforzada del layout (G-LAYOUT-SIN-APILAMIENTO v2)
Antes de usar el layout aguas abajo: verificar que NINGÚN par queda a distancia < sep_min (no sólo cero
duplicados exactos). Reportar la distancia mínima resultante. Si aún hay pares bajo sep_min, es dato a
diagnosticar, no a forzar.

## PASO 4 — RE-confirmar el puente con el layout ya sin casi-coincidencias
Re-correr el puente (N=250, 5×8) con el layout corregido. **¿Sobrevive el z?** Ese número reemplaza al 6.92.
- Si el z se mantiene ~fuerte → la coherencia relacional es real y robusta a la resolución. Confirmado.
- Si el z se cae → parte del 6.92 dependía de los pares casi-coincidentes. Dato honesto, se reporta.
NO se ajusta sep_min para que z sobreviva; sep_min está fijado por la resolución en el paso 2.

## PASO 5 — SÓLO si el puente sigue fuerte: reanudar Phantom Fase 2
IC regenerados con el layout limpio (ya sin pares que revienten el leapfrog) → phantomsetup + corrida N~10³,
polyk físico idéntico REAL/NULL, 5×8 semillas, observable = ¿núcleo cruza M_J por colapso con energía
conservada y REAL gana al NULL? Si Phantom AÚN se cae con el layout ya sin casi-coincidencias, es dato a
diagnosticar (no aflojar tolerancia).

## Orden
1 → 2 → 3 → 4 → (5 sólo si 4 sigue fuerte). No saltar. Phantom en pausa hasta pasar el paso 4.
Regla central: cada corrección es por una razón física/numérica nombrada, IDÉNTICA en REAL y NULL, jamás
para que z suba o Phantom converja. El número que sobreviva vale sea cual sea.