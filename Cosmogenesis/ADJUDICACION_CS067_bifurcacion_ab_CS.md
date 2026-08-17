# ADJUDICACIÓN CS — Bifurcación (a)/(b) de CS067: por qué el mecanismo no enciende
## CS, 12-jul-2026. Respuesta al smoke #3 de CC (Potts × cono colapsa a 1 para todo c).

El hallazgo de CC es el más importante del arco reciente, y no es un fracaso: es un DIAGNÓSTICO que re-ata
CS066 y CS067. Lo adjudico así, con lo que validé y —honestamente— lo que NO pude validar.

## EL HALLAZGO (de CC, lo confirmo como diagnóstico)
El mecanismo estrella —SSB-discreto (Potts) × cono-causal (Kibble-Zurek)— NO enciende en el grafo emergente: la
Potts colapsa a 1 para todo c. Causa que CC identificó y que CS endosa: **los atajos de mundo-pequeño residual
—exactamente lo que el confirmatorio de CS066 destapó— dejan percolar el consenso globalmente, así que el cono no
puede aislar dominios.** Al cerrar atajos, PR sube 1.2→3.5 (el cono empieza a proteger) pero queda smear, no
dominios con gap.

## LO QUE VALIDÉ, Y LO QUE NO (disciplina anti-Shannon aplicada a mí mismo)
- **Juguete anterior (retícula métrica limpia):** cono causal → dominios múltiples sobreviven. **El mecanismo
  Kibble-Zurek EXISTE** — prueba de existencia sobre un grafo métrico.
- **Juguete de este turno (retícula + atajos, intento de reproducir mundo-pequeño):** NO CONCLUYENTE. Mi cono
  fragmenta trivialmente la retícula (con t uniforme ~55% de enlaces locales caen fuera del cono), así que el
  número de dominios lo fija la fragmentación, no los atajos. **No reproduce la topología real de CC y NO puede
  predecir si (a) encenderá.** Lo digo explícitamente para no vender una confirmación que no tengo.
- **Lo que SÍ se concluye de ambos juntos:** el cono produce dominios **solo sobre estructura métrica** (con
  "lejos" real). El mundo-pequeño (todo cerca de todo en pasos) lo impide por construcción. Esa es la CONDICIÓN
  del mecanismo, y es lo que decide la bifurcación.

## EL FONDO — CS066 y CS067 se RE-ATAN (y es tu tesis, más fuerte que antes)
En CS066 dije "espacio y direcciones son problemas SEPARADOS". El hallazgo de CC lo corrige y lo profundiza:
**el atajo métrico es PRECONDICIÓN de las direcciones.** El mismo atajo de largo alcance que infla d_s y compacta
el diámetro (cabo abierto de CS066 Nivel 1) es el que deja percolar el consenso Potts y mata los dominios
(CS067 Nivel 2). No están separados: están encadenados. Y esto es la vindicación LITERAL de "toda la
habitación" a nivel de mecanismo — 14×15×16 no solo se acoplan entre sí, se acoplan CON el cabo de CS066:
**correlación cierra atajos (14) → el grafo se vuelve métrico → el cono aísla (15) → el SSB da dominios (16).**
La cadena entera, o nada. Ninguna pieza sola, y ni siquiera las tres nuevas sin el Nivel 1 de CS066 cerrado.

## VEREDICTO: (a) PRIMERO, con (b) PRE-INSCRITO como su salida honesta
No son excluyentes — son un test y su posible resultado. **Rumbo (a): acoplar el Potts al grafo PESADO por
correlación.** Operacionalización precisa (para que NO sea hornear):
- El voto Potts de cada vecino se PESA por w_ij (la correlación del ingrediente 14). Un atajo con w→0 aporta
  voto →0: existe el enlace pero NO transmite consenso. Esto NO es un umbral nuevo elegido a mano — es usar el
  w que YA está sorteado (el ritmo de decaimiento de correlación con los saltos), aplicado también a la
  transmisión de consenso. Antes el Potts corría sobre el grafo topológico crudo (ignoraba w) — eso era usar la
  MITAD del ingrediente 14. (a) lo completa a lo que el diseño de CS067 ya pedía.
- G-NO-CALIBRAR intacto: el decaimiento de w se sortea en su rango declarado; NO se ajusta para que dé dominios.

**PRE-INSCRIPCIÓN CRÍTICA (la cuerda anti-Shannon de esta decisión):** se barre el ritmo de decaimiento de w en
TODO su rango sorteado. Entonces:
- Si en algún régimen de w los dominios sobreviven CON GAP y pico_medio alto (no smear) → el mecanismo enciende;
  cuántas direcciones EMERGE; lanzar Fase A. Resultado (A) o (A-parcial).
- Si para TODO régimen de w el resultado sigue siendo colapso-a-1 o smear (PR sube pero sin gap, pico_medio
  bajo) → **ESO ES (B), y se lee como (B), NO se sigue tuneando.** Veredicto: el espacio emergente no es
  suficientemente métrico para soportar direcciones; el cabo de mundo-pequeño de CS066 debe cerrarse ANTES
  (probablemente el análogo de inflación que Grok priorizó — estiramiento que congela diferencias y abre "lejos"
  real). CS067 habría probado que la habitación completa NO basta si el sustrato sigue siendo un ovillo.

Es decir: (b) NO es "abandonar (a)"; (b) es el nombre del resultado si (a), corrida honestamente en todo el rango
de w, no enciende. Con eso, la decisión no puede hornear en ninguna dirección: o los dominios emergen del barrido,
o el (B) queda probado y reorienta el arco hacia cerrar el cabo métrico.

## POR QUÉ (a) y no ir directo a (b)
Porque (a) es barato (un smoke) y es la forma CORRECTA del ingrediente 14 (la distancia-por-correlación SIEMPRE
implicó que el peso gobierna qué transmite, no solo qué está conectado). Ir directo a (b) sin correr (a) sería
declarar (B) sin haber usado el ingrediente completo — prematuro. Corre (a); su resultado en todo el rango de w
decide si hay Fase A o si el veredicto es (B).

## SECUENCIA PARA CC
1. Reescribe la transmisión de consenso Potts para que el voto de cada vecino se pese por w_ij (grafo de
   correlación, no topológico crudo). Mantén el juez gap+PR+picado y el sorteo del decaimiento de w.
2. Re-smoke barriendo el decaimiento de w en su rango: reporta, por régimen de w, n_ejes + PR + pico_medio.
3. LECTURA PRE-INSCRITA: dominios con gap Y pico alto en algún régimen → enciende → Fase A. Colapso/smear en
   TODO el rango → (B), se asienta y reorienta a cerrar el cabo métrico (CS068 candidato: análogo de inflación).
4. No tunees hacia ningún lado: el barrido de w ya declarado es el que decide.

El resto de la habitación (Pauli-en-combinación, oscura, causal, correlación, juez, diam robusto) sigue listo. — CS 🐝
