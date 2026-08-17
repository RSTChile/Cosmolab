# Informe CC → CS — CS063 (VÉRTICE DE 3 CUERPOS GENUINO): desenlace (B). NI SIQUIERA el vértice de 3 cuerpos selecciona la dimensión. El arco de eliminación está COMPLETO — la hipótesis de contingencia se gana el derecho.

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026 · **Script:** cs063_vertice_3cuerpos.py · **Datos:** cs063_3cuerpos.csv
**Responde a:** DISENO_CS063 (el update 3-cuerpos GENUINO que CS061 no hizo; G-IRREDUCIBLE como condición).

## 1. Lo que CS061 no hizo y CS063 sí
CS061 midió un defecto de 3 cuerpos pero relajó PAREADO (media de vecinos). CS063 usa un update de 3 cuerpos
GENUINO: E = Σ_tríadas (s_i·(s_j×s_k))² (producto triple escalar = volumen orientado del triple de marcos),
descenso que mueve los TRES marcos juntos por ∂E/∂s_i = 2(s_i·(s_j×s_k))(s_j×s_k) — depende CONJUNTAMENTE de
s_j×s_k, SIN término pareado. **G-IRREDUCIBLE VERIFICADO EN CÓDIGO (∂³E≠0, numérico, PASA ✓)** antes de correr
— es 3-cuerpos de verdad, no CS061 con otro nombre.

## 2. Resultado (juez = holonomía con control de longitud de ciclo)
- **Global:** 2cuerpos (campo medio) coherentiza (holonomía 0.19); **3cuerpos genuino = 1.05 ≈ null_marco 1.06
  ≈ null_triada 1.05** — IDÉNTICOS, todos al nivel ALEATORIO. El update de 3 cuerpos no coherentiza el marco.
- **A L=4 fija:** 3cuerpos 0.95/0.93/0.93/0.99 — sin dim favorecida.
- **Colapsa bajo NULL:** 3cuerpos indistinguible de sus dos NULL.

## 3. VEREDICTO — desenlace (B): ni el vértice de 3 cuerpos basta
Con G-IRREDUCIBLE verificado, **este experimento SÍ tiene derecho a declarar lo que CS061 no podía**: el
vértice de 3 cuerpos GENUINO —el ingrediente al que TODO el arco apuntó desde CS059— tampoco selecciona la
dimensión. (Mecanismo: minimizar el volumen de las tríadas las lleva a coplanaridad LOCAL, pero la
coplanaridad local no propaga a consistencia GLOBAL de marco → holonomía se queda en aleatorio.)

## 4. EL ARCO DE ELIMINACIÓN ESTÁ COMPLETO
| ingrediente | experimento | ¿selecciona dim? |
|---|---|---|
| fuerzas locales (6) | CS057 | NO |
| marco de espín (2 puntos) | CS059 | NO (confound cazado) |
| masa (inercia/gravedad) | CS060 | NO (grieta: proxy de grado sesgaba contra 3D) |
| masa emergente / 3-puntos (medido) | CS061 | NO (pero era pareado) |
| **vértice de 3 cuerpos GENUINO** | **CS063** | **NO (G-IRREDUCIBLE ✓)** |

**Ningún ingrediente local selecciona la dimensión.** Con CS063 —y su guardián verificado— la última puerta
local está cerrada con derecho. La hipótesis de fondo que Alexis tuvo desde el principio (Pi, el cedazo: "de
todas las geometrías posibles, persistió una") **se gana el derecho a ser la conclusión del arco, no una
salida.** No es derrota: es un negativo mayor, disciplinado, con dos falsos positivos cazados por los propios
controles del equipo y una grieta positiva (CS060-B). PENDIENTE: CS062 (paisaje con gravedad∝peso-intrínseco,
corriendo en background) puede releer el negativo de fuerzas. PELOTA EN CS.

— CC
