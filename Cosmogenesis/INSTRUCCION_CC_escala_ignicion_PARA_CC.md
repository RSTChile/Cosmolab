# INSTRUCCIÓN — Subir la escala hasta que la ignición se resuelva (decisión de Alexis: opción 1)
**De:** CS. **Para:** CC. Regla vigente: no mover parámetros a arbitrio; un cambio es un dato a coordinar.

## Decisión
Alexis define el rumbo: subir la escala del experimento hasta que el modelo de estrella funcione. El
experimento base (puente) está confirmado (z=6.92 con layout limpio). El único bloqueo de la ignición fue
resolución (N=250 sub-resuelto), no física ni teoría. La máquina aguanta corridas largas (hemos corrido ~1
día). Adelante — y sin más paradas por problemas de código: todo esto ya está probado (Fase 0: 12/12
gravedad, órbita 2 cuerpos a 2.2e-14). Lo que sigue es correr a la escala correcta.

## PASO 1 — Calcular el N mínimo que resuelve el colapso (de NUESTROS datos, no a ojo)
De la corrida real, computar el N requerido por el criterio estándar de resolución SPH para colapso
gravitatorio: la masa de Jeans mínima del colapso debe resolverse con al menos ~2·N_neigh partículas.
(Este criterio se atribuye a Bate & Burkert 1997, MNRAS — CONFIRMÁ la referencia y el factor exacto en la
fuente antes de citarlo formalmente; no lo tomes de esta instrucción como dato firme.) Usar el N_neigh
REAL del kernel de Phantom en uso (no un número que yo ponga aquí), M_total, y la M_J mínima que alcanza el
gas al enfriarse. Reportar ese N_min. NO elegir N a mano; sale del criterio físico.

## PASO 2 — Correr la ignición a N ≥ N_min (subir hasta cruzar, escalonado)
- Correr a N_min (y si conviene, una escala por encima para confirmar convergencia). v inicial y todo lo
  demás igual; polyk físico (c_s² del piso de enfriamiento), idéntico REAL/NULL.
- Verificar que a ese N el error de momento angular Y de energía caen bajo umbral POR resolución (deben
  bajar al subir N, si el diagnóstico de sub-resolución era correcto — es la confirmación de que era eso).
- Si a N_min los errores aún no bajan, subir otra potencia y reportar la tendencia de |ΔL/L| y |ΔE/E| vs N
  (deben decrecer). Es la prueba de que el problema era resolución.

## PASO 3 — El observable de cierre (REAL vs NULL, sin cambios)
A la escala resuelta, ≥5 semillas × ≥8 NULL:
- ¿Un núcleo cruza M_J local y COLAPSA (densidad crece varios órdenes) con energía y momento conservados?
- ¿REAL enciende más/antes que NULL? z-score. Los 3 resultados pre-inscritos:
  (A) cruza por colapso real, conservando, y gana al NULL → CIERRE POSITIVO: la primera estrella emerge del
      sustrato cosmosemiótico. Cierra Cosmogénesis.
  (B) con resolución suficiente y conservación OK aún NO cruza / REAL=NULL → negativo robusto, FÍSICO.
  (C) cruza pero no gana al NULL → parcial.

## Guardianes
G-CONSERVACION-ENERGIA + G-CONSERVACION-MOMENTO (|ΔE/E| y |ΔL/L| acotados; NUNCA I_WILL_NOT_PUBLISH_CRAP).
G-PARAMETROS-IDENTICOS-REAL-NULL. G-RESOLUCION-SIGUE-FISICA (N sale del criterio de Jeans, no se elige para
que encienda). El N sube por razón física; el resultado que salga a esa escala vale sea cual sea.

## Orden: 1 → 2 → 3. Reportar N_min y la tendencia de errores vs N antes del observable final.