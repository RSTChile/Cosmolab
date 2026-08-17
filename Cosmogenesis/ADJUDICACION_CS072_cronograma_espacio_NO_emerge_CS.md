# ADJUDICACIÓN CS — CS072 cronograma modular: EL ESPACIO NO EMERGE COMO MÉTRICA (hallazgo negativo)

## Qué se probó
El cronograma modular (cs072_modulos/): un enfriamiento físico (T~1/√, feroz al inicio, se frena), cada fuerza
en su época por temperatura, con TODAS las piezas nuevas que el director pidió:
- Freeze-out del neutrón (débil): el ratio p:n = 7:1 EMERGE del barrido expansión-vs-débil (no impuesto), intensivo.
- Fluctuaciones de distribución (#23): campo de densidad rugoso, INTRÍNSECO por partícula (se permuta con ella).
- Competencia gravedad-vs-expansión: las regiones sobredensas colapsan, las tenues se dispersan. SIN constante libre.
- Selección FÍSICA de átomos: recombinan primero los protones en regiones más densas (no por índice).
- Tiempo emergente: conteo de transiciones irreversibles (átomos neutros). Nace con el primer átomo.
- Espacio relacional: la métrica es el grafo de ligadura gravitatoria (Bgrav), no coordenadas.

## Resultado (verificado por CS, corriendo el motor)
1. INVARIANCIA A PERMUTACIÓN: LOGRADA. Con densidad intrínseca + selección física, el diámetro es invariante
   al orden del array (base=1, perms=[1,1,1,1]). Se eliminó el último residuo Shannon: el índice ya no decide
   qué materia se vuelve átomo ni cómo se teje la red. (Dos bugs de índice cazados por auditoría y corregidos.)
2. CRECIMIENTO CON N: NO. El diámetro queda CLAVADO en 1 de 15 a 120 átomos (nq=300→2400). Diámetro 1 = grafo
   completo: todos los sobredensos se ligan con todos. Es invariante pero NO es una métrica: es un hub/estrella.

## Veredicto: MUNDO B (el espacio métrico no emerge) — coherente con CS066-069
Aplicando el criterio de salida binario pre-acordado (director + revisor externo):
- invariante PERO diámetro no crece con N => NO hay métrica => se acepta el HALLAZGO NEGATIVO sin adornos y se
  deja de iterar (no se parchea hacia un diámetro que crezca; eso sería fabricar el resultado, como la materia-artefacto).

## La razón física (el hallazgo, no sólo el negativo)
La sobredensidad GLOBAL no tiene noción de LOCALIDAD. Todos los átomos "densos" se ligan entre sí sin importar
si están "cerca" — pero NO hay un "cerca" previo (correctamente, no se metió: sería Shannon). Sin vecindad local,
la gravedad conecta todos los picos = estrella. Esto reproduce, desde la capa más profunda del arco, lo que
CS066/CS067 ya encontraron: EL ESPACIO MÉTRICO REQUIERE LOCALIDAD COMO PRECONDICIÓN, y la localidad misma NO
emerge de estos ingredientes (rugosidad + expansión + masa + gravedad). El "atajo métrico" que faltaba en CS066
es el mismo que falta aquí.

## Lo que SÍ quedó probado y en pie (positivos del cronograma)
- La arquitectura de ÉPOCAS funciona: cada fuerza actúa en su umbral de temperatura, con su observable propio.
- Admisibilidad por pieza: apagar fuerte→0 bariones; apagar EM→0 hidrógeno; apagar débil→sin ratio; apagar
  aniquilación→10 en vez de 3 (asimetría bariónica). El test decisivo del confinamiento, roto antes, se restauró.
- El 7:1 p:n EMERGE del barrido (firma de la tasa de expansión), no se impone. Intensivo.
- El TIEMPO emerge como conteo de irreversibilidad (nace con el primer átomo). El tiempo sí es consecuencia limpia.
- Helio aparece a escala (marcador de que todas las fuerzas actuaron).
- Modularidad: cada fuerza aislada en su módulo -> los bugs se cazan y arreglan sin romper el resto (probado:
  arreglar EM ya no rompe el confinamiento, como sí pasaba en el script monolítico).

## Lección de método (asentar)
- El espacio-como-grafo-relacional es invariante y honesto, pero "invariante" NO implica "métrico". El discriminante
  es el crecimiento del diámetro con N, no la invariancia ni el valor absoluto. Ya asentado en CS068; reconfirmado.
- Dos capas de Shannon aparecieron en secuencia (densidad-por-índice; selección-de-átomos-por-índice). Cada una
  se veía "invariante" hasta que el test se hizo genuino. Lección: el test de permutación sólo vale si TODO lo
  que decide el resultado se permuta con la partícula.
