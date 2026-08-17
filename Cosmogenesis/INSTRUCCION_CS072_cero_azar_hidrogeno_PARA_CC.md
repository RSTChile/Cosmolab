# INSTRUCCIÓN PARA CC — CS072: reconstrucción con CERO AZAR + cadena hasta el Hidrógeno (estado 0)
## Del director (Alexis), vía CS. 18-jul-2026. Documento que manda: MANIFIESTO_FOLD_CS072.md (leerlo entero antes de tocar código).

## LO QUE HAY QUE HACER — Y NADA MÁS QUE ESTO
Reescribe SÓLO el arranque del fold (cómo se puebla el estado inicial y cómo se mide la llegada). El motor de perillas,
la poda que baja el peso real, la lectura por filtración y las 21 piezas ya validadas NO se tocan salvo lo que aquí se
indica. NO agregues piezas, mecanismos, ni pruebas que no estén en el manifiesto. Si algo no cierra, PREGUNTA a CS
antes de tocar — no inventes.

## 1. FUERA TODO EL AZAR (G-CERO-AZAR, dura)
NO existe azar en un mundo uniforme. Elimina TODO RNG del estado inicial y del catálogo: nada de rng.choice,
rng.integers, rng.uniform, rng.random, rng.shuffle, ni semillas, para decidir QUÉ es cada partícula, sus propiedades,
ni el sector oscuro. Si el código llama a CUALQUIER función de azar para construir el arranque → la corrida es INVÁLIDA.
La "semilla" del generador (el número del RNG) NO es la asimetría/ε de la Teoría — es un tecnicismo de programación.
Se elimina por completo.

## 2. CANTIDADES FIJAS Y DETERMINISTAS
Pon un número FIJO de cada ENTIDAD (quarks, electrones, positrones, antiquarks…), repartido de forma
determinista, NO sorteado. Tú (CC) eliges cómo repartir de forma fija; el criterio es que sea reproducible por
construcción, sin ninguna tirada de dado.
OJO: los gluones NO son una entidad que se cuenta ni se pone en el reparto — ver punto 4.

## 3. LA ASIMETRÍA = DESBALANCE FIJO DE CANTIDADES (no azar)
La asimetría materia-antimateria se pone como un número ENTERO fijo: por cada mil millones de antipartículas, mil
millones Y UNA partículas. Vale para quarks/antiquarks y para electrones/positrones. Es la ε de la Teoría en números
enteros — un desbalance fijo, no un sorteo.

## 4. CORRE LA CADENA HASTA EL HIDRÓGENO (cada eslabón pasa una cantidad FINITA al siguiente)
  1. Quarks menos antiquarks (aniquilación #8) → sobrevive sólo el exceso de quarks (los antiquarks ya se
     consumieron en la aniquilación). De ese exceso, los que cierran TRÍOS forman BARIONES (protones/neutrones); lo
     que no cierra trío (el resto de dividir por 3, a lo sumo 2 quarks) queda como residuo — es un observable, no un
     error. (NOTA: durante la fase caliente, ANTES de que la aniquilación deje sólo el exceso, sí hubo mesones
     quark-antiquark, pero eran inestables y ya decayeron en luz; en el estado que cuenta para el hidrógeno sólo
     queda el exceso de quarks.) EL GLUÓN NO SE PONE NI SE CUENTA COMO ENTIDAD: el gluón ES LA
     RELACIÓN entre quarks (la fuerza fuerte #3 = peso de afinidad W_ij, el ⟷ de S=I⟷E). Y es una relación EN
     MOVIMIENTO: se crea, se divide, se comparte y se refuerza entre los quarks ligados en cada paso (esto es la
     fuerza fuerte #3 ACOPLADA a la memoria de enlace #2 — NO un conteo fijo de aristas). Modélala así: el peso de
     afinidad entre los quarks de un mismo trío/dúo se comparte y se refuerza paso a paso, no como N flechas quietas.
     Sin quarks interactuando → CERO gluones (no hay gluón en el vacío). Los gluones/relaciones son OBSERVABLE DE
     SALIDA (cuánta relación se activó), no ingrediente de entrada.
  2. Electrones menos positrones (aniquilación #8, con la asimetría del punto 3) → sobreviven los electrones del +1.
  3. HIDRÓGENO = 1 protón + 1 electrón (#4, electromagnetismo liga electrón a protón). Cuánto H aparece =
     min(protones, electrones) — lo limita el que ESCASEE. Si sobran protones pero hay 1 electrón → 1 solo hidrógeno,
     el resto de protones queda suelto.

## 5. EL HIDRÓGENO ES EL ESTADO 0 = PUNTO DE LLEGADA
Cuando aparece el PRIMER hidrógeno estable, TODAS las fuerzas ya operaron (fuerte pegó quarks, EM ligó el electrón,
gravedad #2 entre masas, débil #5 en las aniquilaciones). El hidrógeno es la PRUEBA de que las fuerzas cerraron. Ése
es el resultado que el fold debe alcanzar y reportar. Las 21 piezas del manifiesto actúan JUNTAS en el camino hasta
ahí — ninguna se recorta.

## 6. BARRE EN ESPACIO DE POTENCIAS (no simules partícula por partícula)
CLAVE de cómputo: NO se simulan 10^82 partículas una por una. Se barre en el ESPACIO DE LOS EXPONENTES (10^0, 10^1,
10^2 … 10^82). La cadena estequiométrica (aniquilación → bariones → hidrógeno) es aritmética de enteros grandes
(Python los maneja exactos) → todo el barrido de materia corre en microsegundos a escala REAL, hasta 10^82. La
GEOMETRÍA (fold de 21 piezas sobre relaciones) sí necesita el grafo: se barre a escala SIMULABLE (10^2, 10^3, 10^4…
hasta donde el cómputo aguante) y se extiende la tendencia por potencias.
Ejes del barrido (todos deterministas, en potencias — NADA sorteado):
  (a) TAMAÑO total (cantidad de partículas): barrido de potencias 10^0 … 10^82.
  (b) PROPORCIONES entre especies-entidad (cuántos quarks vs leptones; razones color/carga para cerrar bariones).
  (c) MAGNITUD DE LA ASIMETRÍA — la desviación ínfima de la gradiente (S>0). Es una PERILLA: barrer cuán ínfima es
      (1 en 10, 1 en 100, … 1 en 10^9 = el valor cósmico real). Por debajo de que la asimetría × cantidad dé ≥1,
      sobrevive CERO.
  (d) VELOCIDAD DE EXPANSIÓN — cuán rápido enfría. Es la otra PERILLA: si es rápida, CONGELA la desviación antes de
      que el calor la borre (asimetría permanente, irreversible); si es lenta, la gradiente se re-homogeneiza y todo
      se aniquila simétrico → universo vacío. El universo existe SÓLO en la banda donde una desviación ínfima se
      congela a tiempo. Barrer la velocidad y ver dónde la asimetría sobrevive vs se borra.
Como NO hay azar, NO hace falta repetir para promediar ruido: cada punto es reproducible; lo que se repite es el
BARRIDO de potencias sobre esos cuatro ejes.

## 7. QUÉ REPORTAR (por cada punto del barrido)
- bariones formados (y quarks sueltos = residuo)
- electrones sobrevivientes
- HIDRÓGENO logrado: sí/no y CUÁNTO
- el UMBRAL: en qué potencia (de cantidad × asimetría) pasa de 0 a ≥1 átomo (la transición, no una pendiente suave)
- si la asimetría SOBREVIVE o se BORRA según la velocidad de expansión (perilla d)
- (y lo que el manifiesto ya pide de geometría por filtración, SI aplica al estado alcanzado)
El sentido del barrido es ver qué combinaciones de cantidades LOGRAN el primer elemento (hidrógeno) y cuáles dejan
sólo luz, sin química posible.

## LA TESIS (lo que hay que MOSTRAR — está en la cabecera del manifiesto, PRE-INSCRITA):
  1) S>0 genera las condiciones: con S=0 (simetría exacta) sobrevive CERO — universo vacío, sólo luz.
  2) La cantidad de diferencias alcanza un UMBRAL CRÍTICO: transición en el barrido de potencias (cero, cero, y de
     golpe aparece), no pendiente suave. El umbral coincide con el dato cósmico real (asimetría ~1 en 10^9).
  3) Sobre el umbral, con 1 átomo, TODAS las fuerzas están y aparecen tiempo y espacio: el primer hidrógeno = las 21
     piezas cerraron. El resto es AUMENTAR LA CANTIDAD — ninguna ley nueva al escalar de 1 a 10^82.
EL ORIGEN (lo que hay que barrer): los números enormes (10^82) NO son la causa — son la CONSECUENCIA. La causa es
una desviación ÍNFIMA de la gradiente de temperatura (asimetría, perilla c), CONGELADA por lo rápido de la expansión
(perilla d) antes de que el calor la borre. Asimetría ínfima × cantidad colosal = residuo colosal. Esas dos perillas
(magnitud de la desviación × velocidad que la congela) definen la BANDA donde nace universo.

## 8. EL CONTROL (NULL) YA NO ES "OTRA SEMILLA"
Sin azar no hay semillas. El NULL es cambiar de forma DETERMINISTA qué propiedades van juntas (misma población fija,
sólo se reordena qué color-carga-masa se emparejan), NO barajar un universo distinto. Real y NULL parten de la MISMA
población fija — sólo cambia el emparejamiento de propiedades.

## RECORDATORIO
Las 21 cosas (18 elementos + 3 mecanismos) están en el manifiesto y van TODAS, juntas, en una sola corrida por punto
del barrido. Esto NO recorta ninguna: define el reparto (determinista) y el punto de llegada (hidrógeno). Cualquier
duda → PREGUNTA a CS antes de tocar. No inventes piezas que nadie pidió.
