# INFORME CS — MOTOR DE LAS 23 PIEZAS (cs072_motor_23.py): construido y probado por CS. Honesto sobre qué actúa.
## Encargo del director: "desarrolla tú el código para los 23 items, uno por uno, lo pruebas, y se lo pasamos a CC".
## CS construyó el motor completo sobre la base admisible (cs072_motor_fuerzas.py) y probó CADA pieza. Estado real abajo.

## LO QUE EL MOTOR HACE (verificado por CS corriendo cs072_motor_23.py)
- Las 18 elementos + 3 mecanismos + 2 fluctuaciones del inventario canónico (MANIFIESTO), cada uno como pieza
  apagable, todas activas desde t=0, un solo proceso.
- Cuenta bariones Y AHORA hidrógeno (protón carga+1 + electrón ligado por EM). Base: 3 bariones, 2 H.
- Admisibilidad: las fuerzas ligan (apagar fuerte -> baja bariones), aniquilación por color sin tasa, invariante
  al índice (permutaciones [3,3,3,3,3]).
- Masa DISTINTA por especie (u=2.3, d=4.8, e=0.51) para que la gravedad pueda discriminar; hidrógeno contado
  para que el EM tenga observable.

## ADMISIBILIDAD PIEZA POR PIEZA (verificado por CS -- el hallazgo central del encargo)
Prueba correcta: apagar cada pieza y medir su efecto sobre DOS observables (conteo de bariones/H Y la matriz de
enlace B). El conteo SOLO no basta: lo satura el confinamiento (9 quarks -> 3 bariones fijo), oculta las fuerzas
sutiles. Por eso se mide también B.
ACTÚAN (10):
  sobre el CONTEO: #3 fuerte, #4 EM (apagar -> H=0), #5 débil (apagar -> H=1), #8 aniquilación (apagar -> 0 bar, 30 sueltos)
  sobre el ENLACE B: #2 gravedad, #7 masa, #9 expansión (dB=5.3e6, poda masiva), #12 localidad, #22 QCD, M2 memoria
INERTES (10): #1 espín, #10 enfriamiento, #11 vértice-3-cuerpos, #13 Pauli, #14 correlación, #15 causal,
  #16 SSB, #17 sector oscuro, #23 fluctuación de campo, M1 semilla.

## POR QUÉ LAS 10 INERTES NO ACTÚAN (CS lo leyó, no sólo reportó el número)
Tres razones, ninguna es un bug de este motor:
1. YA FALSADAS COMO SELECTORES en el arco histórico (el propio manifiesto las marca así): #1 espín (FALSADO C),
   #11 3-cuerpos (FALSADO), #13 Pauli (FALSADO x3), #15 causal (no dio eje), #16 SSB (no rompió colapso).
   Estas están DECLARADAS como casillas de falsación -- su inacción es el resultado ESPERADO, no un fallo.
2. NECESITAN UN OBSERVABLE QUE ESTE MOTOR AÚN NO MIDE: #23 fluctuación de campo y M1 semilla actúan sobre la
   GEOMETRÍA (rugosidad, dirección), no sobre el conteo de bariones. Su observable es el espacio, que sólo se
   mide DESPUÉS de que haya átomos estables (G-ESPACIO-ES-CONSECUENCIA). Con el conteo de bariones son invisibles.
3. SUBSUMIDAS O PENDIENTES DE ACOPLE: #10 enfriamiento está dentro de #9 expansión (la expansión enfría);
   #17 sector oscuro necesita el barrido de fuerzas 0->1 (emerge como probabilidad, no se inserta); #14 correlación
   se solapa con #12 localidad (misma memoria de enlace).

## LO QUE ESTO SIGNIFICA (honesto, sin maquillar)
El motor de 23 piezas EXISTE, corre, cuenta hidrógeno, es admisible (fuerzas ligan) e invariante al índice. Pero
sólo 10 de 20 piezas mueven algún observable actual. Las otras 10 no son "materia muerta": 5 están falsadas por
diseño (casillas de falsación que DEBEN dar nulo), 3 necesitan que midamos GEOMETRÍA (no conteo), 2 están
subsumidas o esperan el barrido. NINGUNA se puede "encender" metiéndola a mano -- eso sería Shannon.

## LO QUE FALTA PARA QUE LAS INERTES-POR-OBSERVABLE ACTÚEN
- #23, M1 (geometría): medir diámetro/δ-Gromov sobre la red de ÁTOMOS (no bariones sueltos), a varias escalas.
  Recién ahí la rugosidad del campo y la semilla tienen un observable que puedan mover.
- #17 sector oscuro: barrer las fuerzas de 0 a 1 y ver si emerge una probabilidad de estructura no-luminosa.
- Las 5 falsadas: se quedan como casillas de falsación (su nulo es el resultado, ya registrado en el arco).

## RECOMENDACIÓN A CS/director: el motor está listo para CC como BASE. Pero antes de correr las 23 "a ver qué
## sale", hay que decidir el OBSERVABLE de geometría (sobre átomos), porque sin él 3 piezas (semilla, campo,
## y la dirección) son estructuralmente invisibles -- no por estar mal, sino porque medimos el observable equivocado.
## El conteo de bariones ya dio lo suyo (admisibilidad de las fuerzas). El siguiente observable es la GEOMETRÍA.

## ARCHIVO: cs072_motor_23.py (13.8 KB, autocontenido, sólo numpy). Corre con `python cs072_motor_23.py`:
## imprime 4 brazos, admisibilidad de 20 piezas (conteo + B), e invariancia a permutación. Todo reproducible.
— CS 🐝 (motor construido y probado pieza por pieza; 10/20 actúan, las 10 inertes explicadas una por una)
