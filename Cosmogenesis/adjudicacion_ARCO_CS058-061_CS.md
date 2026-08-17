# Adjudicación CS — ARCO DE CIERRE CS058-CS063 (energía oscura, marco, masa, vértice 3-cuerpos GENUINO): NEGATIVO GRANDE con mecanismo. Ni fuerza local, ni marco, ni masa, ni el vértice de 3 cuerpos genuino seleccionan la dimensión → la CONTINGENCIA se gana el derecho a ser la conclusión. DOS falsos positivos cazados por los controles. CS058 corregido con la corrida completa (real-pero-débil). G-IRREDUCIBLE de CS063 verificado por CS. CS062 corre en background.

**De:** CS · **Fecha:** 5-jul-2026 · **Corrido por:** CC · **Fuente auditada (DATOS abiertos y analizados por
CS):** cs061_masa.csv (339 filas), cs060_gravmasa_exp.csv (1350), cs060_gravmasa.csv (450), cs060_leptones.csv
(480), cs058_zoom.csv (17.316 filas) + los CÓDIGOS reales (cs061_masa_emergente.py — leído el update del
marco; cs060 imports) + INFORME_CONSOLIDADO_CS058-061_PARA_CS.md. Los CINCO CSVs fueron abiertos y sus
números recomputados por CS (no leídos del informe). CS058 al 65% de corrida al momento de esta adjudicación:
el CSV parcial (17.316 filas) ya da el veredicto de artefacto firme; se confirma al cierre.

---

## 0. VEREDICTO EN UNA LÍNEA
**ACEPTO el arco CS058-CS061.** Es un negativo grande, disciplinado, con dos falsos positivos cazados por
los controles y una grieta positiva concreta. CC reportó con honestidad ejemplar — incluido el caveat que yo
tenía que dirimir, y que confirmé en su código. Con UNA salvedad de enunciado (abajo, §4): CS061 es
desenlace **(C/D mezclado)**, no (C) puro — y esa distinción importa para qué sigue.

## 1. CS058 (energía oscura): REAL-PERO-DÉBIL — CORREGIDO CON LA CORRIDA COMPLETA (cs058_zoom.csv, 25.272 filas)
**CORRECCIÓN (CC lo destapó, CS lo verifica y asume el mismo error):** la v2 de esta adjudicación declaró
"ARTEFACTO firme" desde el PARCIAL (17.316 filas). La corrida completa (25.272 filas, 1404 puntos) matiza el
veredicto. Verificado por CS en el CSV completo:
- **SUPERA al NULL:** accrob real 0.115 vs null 0.069, ratio 1.66 → hay SEÑAL REAL, no puro artefacto de
  medición.
- **PERO decae con la resolución:** accrob 0.170 (×1) → 0.106 (×2) → 0.069 (×4) → NO es robusta.
- **NO vive en la frontera curva:** máximo en d4 (0.256), no en curv (0.136) → desacoplada de R7.
**Veredicto corregido:** ni artefacto limpio, ni candidato firme a energía oscura, ni puente a R7 — un
fenómeno PROPIO, débil, para una línea aparte. **Lección (de CC, asumida por CS): no declarar veredicto firme
desde un parcial.** CC la anotó primero; CS la confirma habiéndola cometido en la v2. Es la cuerda
anti-Shannon aplicada al propio adjudicador.

## 2. CS059 (marco pareado): NO selecciona — con el confound CAZADO (esto es ciencia de la buena)
CC no se creyó su propio positivo aparente: notó que el orden de holonomía era idéntico al orden de longitud
de ciclo, hizo el control a igual longitud, y la "selección" se desintegró. Falso positivo #1, cazado por el
autor. Verificado.

## 3. CS060 (masa fenomenológica + gravedad con masa): DOS resultados, ambos verificados
- **Misión A (leptones e/μ/τ):** VERIFICADO por CS en cs060_leptones.csv (480 filas). frame_burgers por
  brazo: electrón 0.242 y alineado 0.222 (COHERENTES) vs muón 1.086, tauón 1.100 ≈ nulo 1.088 (INCOHERENTES
  como aleatorio). Es UMBRAL, no gradiente: el muón ya es demasiado pesado, cae al nivel del tauón/nulo. Y el
  brazo electrón "discrimina" curv 0.113 < d4 0.187 < d3 0.293 < d2 0.374 = OTRA VEZ el orden de longitud de
  ciclo (confound, no selección de dim; el brazo "alineado" da el mismo patrón, curv 0.105 < d4 0.162 < d3
  0.267 < d2 0.354). Desenlace G2: la masa toca la coherencia del marco, no la geometría.
- **Misión B (gravedad ∝ masa vs proxy de grado — la que Alexis destapó):** VERIFICADO en cs060_gravmasa_exp.csv:
  acople=grado da viab(3D+4D)=0.106; acople=masa da 0.211 (~2×). PERO acople=null da 0.211 también —
  IDÉNTICO a masa. Falso positivo #2, cazado por el NULL: el efecto NO es de la masa (barajarla da lo mismo),
  sino de que la gravedad se acople a un PESO INTRÍNSECO FIJO en vez de al grado (que se auto-amplifica:
  hubs atraen más → colapso a dim baja → sesgo contra 3D). El guardián corr_masa_grado_ok=1 en todas las
  filas (la masa es variable separada del grado, como pedí). Esto es real y es la grieta positiva (§5).

## 4. CS061 (masa emergente / vértice 3-puntos): el CAVEAT que verifiqué — (C) de dinámica, no (C) de estructura
Los datos confirman el reporte de CC exactamente:
- **Colapsa bajo NULL:** 3punto μ=0.567 ≈ null_campo 0.549 ≈ null_vertice 0.547. La baja de holonomía es
  alineación genérica, no estructura del 3-puntos. Verificado en cs061_masa.csv (80 filas por brazo).
- **No selecciona dim:** el "orden" curv 0.24 < d4 < d3 < d2 0.77 es OTRA VEZ el orden de longitud de ciclo
  (curv tiene ciclos cortos) — el mismo confound de CS059, no selección. Verificado.
- **Espectro trivial:** razón alto/bajo 2.14 (real 3477), frac_sin_masa 0.006 (sin ceros tipo fotón). No
  reproduce la jerarquía. Verificado.
**PERO — y esto es lo que CC marcó honestamente y yo tenía que dirimir leyendo el código:** el vértice de 3
cuerpos está en la MEDICIÓN (el `_defecto_triada` j→i→k vs j→k es genuinamente de 3 cuerpos), pero NO en la
DINÁMICA. Leí `_relaja_inercia`: el update del marco es `w = s[i] + align·(media_de_vecinos − s[i])` — cada
nodo se alinea hacia el PROMEDIO de sus vecinos (campo medio PAREADO). La inercia de 3 cuerpos m_i entra solo
como un ESCALAR de amortiguación (cuánto se alinea), no cambia la ESTRUCTURA pareada del update.
**Consecuencia de adjudicación:** CS061 NO probó un update genuinamente de 3 cuerpos. Probó una dinámica
pareada informada por una inercia de 3 cuerpos. Por lo tanto el negativo es **(C) para "el 3-puntos como
inercia-que-amortigua-relajación-pareada"**, pero deja ABIERTO el **(D): un update donde los tres marcos de
la tríada se muevan JUNTOS (no cada uno hacia la media)**. La distinción no es pedante: es la diferencia
entre "el vértice de 3 puntos no basta" (lo que el titular sugiere) y "no probamos el vértice de 3 puntos en
la dinámica" (lo que el código muestra). CC lo dijo; lo confirmo y lo elevo a condición de enunciado.

## 4-bis. CS063 (vértice 3-cuerpos GENUINO): NEGATIVO (B) — VERIFICADO EN CÓDIGO Y DATOS. Cierra el (D) de CS061.
Este es el experimento que CS061 dejó sin hacer, y el que da derecho (o no) a la hipótesis de contingencia.
Por eso CS lo auditó a fondo — código Y datos, no la prosa:
- **G-IRREDUCIBLE PASA DE VERDAD (re-ejecutado por CS):** el término es E=Σ(s_i·(s_j×s_k))² (producto triple
  escalar al cuadrado = volumen orientado de la tríada, cero ⟺ coplanar). CS recomputó ∂³E/∂s_i∂s_j∂s_k =
  1.96 ≠ 0 → es 3-cuerpos GENUINO, no reducible a pares. NO es CS061 con otro nombre.
- **El update mueve los TRES marcos (verificado en código):** grad[i]+=2·tp·(s_j×s_k), grad[j]+=2·tp·(s_k×s_i),
  grad[k]+=2·tp·(s_i×s_j) — descenso conjunto sobre la tríada, SIN término pareado (sin media de vecinos). Es
  exactamente lo que faltaba en CS061.
- **COLAPSA bajo NULL (verificado en cs063_3cuerpos.csv, 160 filas):** 3cuerpos μ=1.050 ≈ null_marco 1.062 ≈
  null_triada 1.053. Por dimensión casi plano (curv 0.95 → d2 1.12), no discrimina. El control de longitud de
  ciclo ya no hace falta cazar nada: no hay selección que explicar.
**Desenlace (B) confirmado: ni el vértice de 3 cuerpos GENUINO selecciona la dimensión.** Este es el negativo
que CS061 NO tenía derecho a declarar y CS063 SÍ. Cierra el (D). El arco de eliminación local está COMPLETO.

## 5. LA SÍNTESIS (el cuadro, honesto)
| ingrediente | exp | ¿selecciona dim? |
|---|---|---|
| fuerzas locales (6, barridas) | CS057 | NO — físico=curvo |
| energía oscura (candidato) | CS058 | REAL-PERO-DÉBIL (ratio 1.66 vs NULL, decae con resolución, máx en d4 no curv) |
| marco de espín (2 puntos) | CS059 | NO (confound long. ciclo cazado) |
| masa = inercia del marco | CS060-A | NO (coherencia sí, geometría no) |
| masa / vértice 3 puntos, dinámica PAREADA | CS061 | NO — pero era update pareado, (D) abierto |
| vértice 3 CUERPOS GENUINO (∂³E≠0) | CS063 | NO — COLAPSA bajo NULL. Cierra el (D). **Arco de eliminación local COMPLETO** |

Ni las fuerzas, ni el marco pareado, ni la masa (dada o emergente), ni el vértice de 3 cuerpos GENUINO
seleccionan la dimensión. Con CS063 el arco de eliminación local está COMPLETO — y por eso la hipótesis de
fondo (**la dimensión es CONTINGENTE, no seleccionada por ningún ingrediente local**) se GANA el derecho a
ser la conclusión del arco, no una salida. Encaja con la imagen de Alexis desde el principio: de todas las
geometrías posibles, persistió una. Negativo grande, con mecanismo, con dos falsos positivos cazados por los
propios controles del equipo. Es exactamente lo que Alexis pidió asentar "antes de celebrar con vino".
(Salvedad viva: CS062 corre en background — si la gravedad∝peso mueve el paisaje, el "COMPLETO" se relee.)

## 6. LA GRIETA POSITIVA (real, y es de Alexis por partida doble)
El proxy de grado de la gravedad de CS054-057 sesgaba ACTIVAMENTE contra el 3D (auto-amplificación tipo
enlace-preferencial). Con gravedad acoplada a un peso intrínseco fijo —como la gravedad real, que Alexis
destapó que faltaba— el 3D/4D es ~2× más viable. NO cierra el hueco (el efecto es independencia-del-grado, no
masa), pero RELEE el negativo central: CS057 probó en parte un artefacto del proxy. Esto vale un experimento.

## 7. LOS HILOS — estado actualizado (uno cerrado, uno vivo, la conclusión ganada)
1. **★ Re-correr el paisaje de CS057 con gravedad ∝ peso-intrínseco (no grado) = CS062.** BARATO (reusa
   CS057), ataca el hallazgo más concreto, puede mover el negativo central del arco. **CORRE EN BACKGROUND
   (CC, overnight).** ÚNICO hilo vivo: si mueve el paisaje, el "arco COMPLETO" de abajo se relee.
2. ~~El vértice 3-cuerpos GENUINO (cerrar C vs D de CS061)~~ **= CS063 → CERRADO (ver §4-bis).** CORRIÓ y
   ADJUDICADO NEGATIVO (B): G-IRREDUCIBLE verificado por CS (∂³E=1.96≠0), update mueve los tres marcos sin
   término pareado, COLAPSA bajo NULL. El arco YA PUEDE decir "ni el vértice de 3 cuerpos genuino selecciona"
   — el (D) que CS061 dejó abierto está cerrado. Este hilo ya NO está abierto.
3. **La hipótesis de fondo: la dimensión es CONTINGENTE, no seleccionada — YA SE GANÓ EL DERECHO.** Con
   CS063 cerrado, el arco de eliminación local está COMPLETO (fuerza, marco pareado, masa, vértice 3-cuerpos
   genuino: todos descartados). NO es derrota — es un resultado teórico grande (encaja con "de todas las
   geometrías posibles, persistió una"). Pasa de "se deja nombrada" a CONCLUSIÓN del arco. Única reserva: la
   salvedad de CS062 (1) — si el paisaje con gravedad correcta cambia, se relee.

## 8. NOTA DE HONESTIDAD SOBRE ESTE ARCO
Dos falsos positivos (CS059 confound, CS060-B NULL) los cazó CC solo, sin que yo se lo pidiera. Eso es la
cuerda anti-Shannon funcionando dentro del equipo, no impuesta desde fuera. El único punto que yo agrego es
el §4: el titular "el 3-puntos no selecciona" debe leerse "la dinámica pareada informada por inercia de
3-cuerpos no selecciona" — porque el update real, verificado en el código, es de campo medio pareado. CC ya
lo había señalado; yo lo confirmo y lo hago condición.

— CS. La corrida, los controles y la honestidad del caveat son de CC. La observación que abrió la grieta
positiva (gravedad sin masa) es de Alexis. La verificación de datos y código, la distinción (C)/(D) del
vértice, y esta adjudicación de prioridades, mías. Registrar CS058/059/060/061/063 como corridos-adjudicados
(CS063 negativo B, G-IRREDUCIBLE verificado, cierra el arco de eliminación local). Único hilo vivo: CS062
(paisaje con gravedad ∝ peso intrínseco) corriendo en background — si mueve el paisaje, el cierre se relee.
Próximo número libre: CS064.
