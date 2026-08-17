# INSTRUCCIÓN PARA CC — Prueba de sensibilidad de tasa_expansion (0.02)
**De:** CS · **Fecha:** 20-jul-2026 · **Regla:** corres y mides, NO cambias el motor ni interpretas. Un desacuerdo con lo esperado es un DATO, no algo que ajustar.

## Por qué
El 0.02 quedó "elegido, no derivado": es el valor con el que la fórmula fraudulenta del 7:1 acierta, y se usó en TODO el arco (reloj de enfriamiento, a_final=√60, cada corrida de Phantom, los dos brazos de velocidad). No sabemos si el resultado REAL>NULL del puente sobrevive a otro valor, o si dependía por casualidad de ese número. Esta prueba lo decide. NO toca el 7:1 (ya aislado); prueba si las COMPARACIONES del arco son robustas a la constante.

## Qué correr — EXACTO
Repetir la comparación REAL-vs-NULL del PUENTE (la que dio z=6.92) con TRES valores de tasa_expansion: **0.01, 0.02, 0.03**. Nada más cambia entre las tres tandas — mismo N, mismas semillas, mismo layout reflectante, mismo número de repeticiones REAL y NULL, mismo discriminante (nº de clusters ligados, la misma métrica con la que salió z=6.92). El único parámetro que varía entre las tres tandas es tasa_expansion.

Para cada valor de tasa: reportar REAL (media±sd sobre las repeticiones), NULL (media±sd), y el z = (media_REAL − media_NULL)/sd_NULL, EXACTAMENTE como se computó el z=6.92 original. Misma cantidad de repeticiones que la corrida original (5 REAL × 8 NULL, o la que fue — usa la MISMA, no menos).

## GUARDIANES (estrictos — si uno se viola, PARA y reporta, no sigas)
1. **G-MISMO-DISCRIMINANTE**: el z se calcula con la MISMA fórmula y la misma métrica (clusters ligados) que dio 6.92. Prohibido cambiar la métrica "para que se vea mejor".
2. **G-SOLO-CAMBIA-TASA**: entre las tres tandas, lo ÚNICO distinto es tasa_expansion. Mismas semillas, mismo N, mismo todo. Si algo más cambia, la comparación no vale.
3. **G-NULL-MISMA-MAGNITUD**: el NULL se genera igual en las tres (barajado de aristas, misma cantidad), a la misma escala que su REAL.
4. **G-NO-TOCAR-PROHIBIDAS**: no tocar I_WILL_NOT_PUBLISH_CRAP, tolv, dt, ni el criterio de conservación. No tocar el 20.0 ni el freeze-out (es callejón sin salida, irrelevante aquí).
5. **G-REPORTA-EL-FEO**: si a 0.01 o 0.03 el z CAE (REAL deja de ganarle al NULL), ese es el resultado y se reporta tal cual. NO se ajusta nada para recuperarlo. Un z que cae ES el hallazgo: significaría que el arco dependía del 0.02.

## Cómo se lee (pre-inscrito, antes de correr)
- **Si z sigue siendo grande y positivo en 0.01 y 0.03** (mismo orden que 6.92) → el resultado REAL>NULL del puente es ROBUSTO a la constante; el 0.02 era convención arbitraria pero inocua; las comparaciones del arco sobreviven.
- **Si z cae a ~0 (o se vuelve negativo) en 0.01 y/o 0.03** → el resultado dependía del 0.02 específico; el arco NO es robusto y la comparación queda tan sin pedigrí como la constante. Se reporta como negativo, sin adornos.
- **Cualquier resultado intermedio** → se reporta el número crudo por cada tasa, sin forzar lectura.

## Entregable
Una tabla: tasa_expansion | REAL (media±sd) | NULL (media±sd) | z | ¿sobrevive? — para 0.01, 0.02, 0.03. Más una línea de veredicto crudo. NADA de interpretación cosmológica. El motor contesta; CS adjudica después.
