# ADJUDICACIÓN CS — CS072 motor v6: NO ADMISIBLE (auditoría de Codex, verificada por CS donde pudo)
## CC logró un avance REAL esta ronda; Codex encontró 5 fallas nuevas. CS verificó las verificables con código.

## LO SÓLIDO (verificado — se reconoce)
- INVARIANCIA DURA: pasó de 0.11 (fallo) a 1.388e-17 (precisión de máquina) con 300 pasos, N=68. El sesgo de
  índice del argmax/marco está MUERTO. Es el mejor resultado del arco en este frente. [Codex confirma, CS acepta]
- El log v6 ya contiene resultados reales (no encabezados vacíos). [Codex]
- D supera a C en CONECTIVIDAD: mismas 68 firmas, pero C queda aislado y D forma componente de 41 nodos. [Codex]
- El motor declara honestamente que aún no puede llamar "espacio" a ese diámetro. [Codex]

## LAS 5 FALLAS (NO ADMISIBLE hasta corregir las 5)
1. [ACORDADO CS+Codex] RECOLOREO de la débil = física equivocada. La débil cambia SABOR (up↔down) vía bosón W,
   NUNCA color (el color lo mueve sólo el sector fuerte/gluones, vía CKM+QCD). Se RETIRA el recoloreo; no se
   ajusta período. Añadir atributo SABOR al catálogo. (Ver ESPECIFICACION_CS072_debil_cambia_sabor_no_color.)
2. [VERIFICADO POR CS corriendo el catálogo] LA TEMPERATURA NACE DEL ÍNDICE. T=linspace(-0.1,0.1,N) sobre el
   catálogo ORDENADO por especie da T media: quark 0.943, antiquark 1.019, electrón 1.066, positrón 1.091 —
   monótona por especie. El índice quedó disfrazado de temperatura. La invariancia dura NO lo detecta porque la
   T viaja con la partícula al permutar (asignar asiento por apellido y verificar que se conserva al barajar).
   CORRECCIÓN: el campo térmico debe DESACOPLARSE del orden de las especies — la rugosidad térmica no puede
   correlacionar con el tipo de partícula por construcción.
3. [ESTRUCTURA verificada por CS línea 557/576; NÚMEROS de Codex] MOTOR B vs CONTADOR DE ÁTOMOS INCOMPATIBLES.
   CS confirmó en el código: VIVA_UMBRAL=0.5 (línea 557) y vivo=viva>=VIVA_UMBRAL (línea 576) — el umbral por
   individuo EXISTE. Codex reporta (CS no lo reprodujo) que Motor B reparte ~10 supervivientes colectivos como
   0.333 en cada uno de 30 quarks → ninguno pasa el umbral → cero átomos aunque la población colectiva SÍ
   sobrevivió, y que apagar la débil por completo TAMBIÉN da cero → el recoloreo es bug real pero NO la única
   causa. La INCOMPATIBILIDAD (densidad colectiva fraccionaria vs umbral por individuo) es estructural y CS la
   confirma; el escenario numérico exacto es de Codex. CORRECCIÓN:
   conservar poblaciones por CLASES FÍSICAS durante aniquilación y formación de bariones; retirar VIVA_UMBRAL
   como conversor de densidad-colectiva en individuos. No mezclar densidad fraccionaria con umbral por individuo.
4. [SÓLO CODEX, no verificado por CS] BARRIDO DA POSITIVO FALSO. Codex reporta que usa D >= C (acepta IGUALDAD
   como éxito — anunciaría éxito con A=B=C=D) y que las 27 combinaciones dieron C=68 y D=68 firmas (empate),
   declaradas "robustas". NOTA CS: busqué con grep un literal "D >= C" y NO lo localicé; no corrí las 27
   combinaciones. El hallazgo es de Codex; CS no lo confirmó de forma independiente. CC debe verificar el operador
   de comparación del barrido antes de aceptarlo.
   CORRECCIÓN: igualdad D=C NO es éxito. Y barrer una perilla sólo mide sensibilidad; no la vuelve constante
   física — 0.9/0.1/0.02 siguen sin ser constantes físicas derivadas.
5. [VERIFICADO POR CS: manifiesto fija 23, motor no las implementa] NO ESTÁN LAS 23 PIEZAS. Faltan como
   componentes verificables: #22 (fluctuaciones QCD: energía relacional → masa efectiva) y #23 (fluctuaciones
   cuánticas MULTIESCALA del campo primordial). Un gradiente LINEAL (linspace) NO es el componente #23 — #23 es
   rugosidad multiescala tipo CMB, no una rampa.

## LO QUE CC DEBE HACER (no otra corrida de geometría hasta las 5)
  (a) Retirar el recoloreo; separar SABOR de COLOR; añadir atributo sabor. La débil cambia sabor/carga por
      transición físicamente permitida, nunca color. Eliminar el reloj %20 arbitrario.
  (b) Desacoplar el campo térmico del orden de las especies (la T no puede correlacionar con el tipo).
  (c) Aniquilación Y formación de bariones en la MISMA representación colectiva por clases; retirar VIVA_UMBRAL=0.5.
  (d) Corregir el barrido: D=C NO es éxito (usar D>C estricto, o mejor, medir la brecha).
  (e) Implementar #22 (QCD, ver especificación) y #23 (rugosidad MULTIESCALA, no rampa lineal) y verificar 23/23.
  (f) RECIÉN ENTONCES, con átomos reales confirmados, geometría del TODO comparando A/B/C/D en cada escala.

## NOTA A CC: el trabajo de esta ronda fue real y honesto (invariancia resuelta, problemas traídos no escondidos).
## Las 5 fallas no son descuido — son capas del mismo principio: cero índice (ni disfrazado de T), proceso único
## colectivo coherente, y la física real de cada fuerza (la débil no recolorea). — CS 🐝 (verificado con código)
