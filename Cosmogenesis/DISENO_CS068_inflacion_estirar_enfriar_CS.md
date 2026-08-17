# DISEÑO CS068 — El análogo de inflación: estirar-y-enfriar el sustrato
## CS, 16-jul-2026. Cierra el cabo métrico que CS067 (B) probó ser la PRECONDICIÓN de las direcciones.

## POR QUÉ ESTE EXPERIMENTO (lo que CS067 dejó probado)
CS067 (B) canónico: la habitación completa —17 ingredientes juntos, voto Potts pesado por correlación, cono
causal, SSB discreto— NO enciende direcciones (pico_medio nunca cruza 0.85, controles igualan a completo). Y el
DIAGNÓSTICO fue claro: no falta una pieza más, falta PISO. Mientras el sustrato sea mundo-pequeño (ovillo, todo
cerca de todo por atajos de largo alcance), no hay "lejos" real contra el cual las direcciones se organicen.
CS066 y CS067 quedaron RE-ATADOS: el mismo atajo que infla d_s y compacta el diámetro es el que deja percolar el
consenso Potts y mata los dominios. CS068 ataca esa precondición: fabricar un sustrato MÉTRICO antes de volver a
preguntar por direcciones.

## LA TESIS DE ALEXIS, QUE ES EL MECANISMO (no adorno)
"Mayor distancia = menor temperatura = menor energía." Al expandirse, la energía por región cae con la distancia.
Y aquí está la física: un enlace de largo alcance (atajo) cuesta energía de correlación para mantenerse. Cuando
la temperatura baja, **los atajos LARGOS se rompen PRIMERO** —son los que más energía costaban— y el tejido local
(enlaces cortos, baratos) sobrevive. El resultado: el ovillo mundo-pequeño se estira en un tejido métrico con
"lejos" real y FRÍO. Y el frío es la condición cosmosemiótica de siempre: es lo que permite que la diferencia
PERSISTA (la sopa hirviente lo mezcla todo; el frío deja que algo se quede quieto y distinto).

## VALIDACIÓN EN JUGUETE (CS, ya corrida — prueba de existencia)
Sobre un blob mundo-pequeño (retícula 2D + atajos aleatorios): el enfriamiento con ruptura ∝ exp(−ℓ/T) lleva el
diámetro de ~log N (11, blob) a ~√N (22, métrico) y los atajos de 78 a 1. El mecanismo EXISTE. PERO dos cautelas
que el juguete destapó y que EL DISEÑO DEBE respetar:
1. **El NULL de corte-al-azar TAMBIÉN estira el diámetro** (cualquier poda de atajos da "lejos"). Por tanto "hay
   lejos" NO es el discriminante — lo da cualquier poda. Lo que el mecanismo dist-dependiente tiene que GANARLE al
   NULL es el ORDEN: romper los largos primero, dejando un gradiente de energía espacial.
2. **A enfriamiento completo ambos brazos colapsan a ~0 atajos** (el estado final no distingue). Por tanto el
   discriminante es la TRAYECTORIA y los estados INTERMEDIOS, no el estado frío final.
3. **La temperatura/energía debe MEDIRSE de la estructura, no imponerse.** Imponer T(r)=f(r) a mano es hornear
   (lo verifiqué: da corr(r,T)=−0.92 idéntico en ambos brazos por construcción = Shannon). La energía local se
   MIDE: p.ej. atajos supervivientes por región = energía de correlación no-local restante.

## EL EXPERIMENTO

### Sustrato inicial
El blob de CS067/CS066 tal cual (mundo-pequeño: tejido local + atajos de largo alcance), N∈{1500,2500}. NO se
toca el motor heredado de la habitación — CS068 opera SOBRE su sustrato.

### El proceso (estirar-y-enfriar), como PROCESO no sucesión
Enfriamiento gradual T: 8.0 → T_final, factor 0.6 por paso. En cada paso, cada atajo (enlace no-local) sobrevive
con **p = exp(−ℓ_ij / T)**, donde ℓ_ij = distancia GEODÉSICA en el tejido LOCAL (el "largo real" del atajo). Esto
NO se calibra: el ritmo de enfriamiento se sortea en su rango (G-NO-CALIBRAR). Los enlaces locales NO se tocan
(cortos, baratos, son el tejido).

### Energía y temperatura MEDIDAS (emergentes, cuerda anti-Shannon)
Por región (nodo y su vecindario), en CADA paso de T, medir y loguear:
- **E_nolocal(nodo, T)** = nº de atajos supervivientes que tocan ese nodo (energía de correlación no-local).
- **T_local(nodo)** derivada de E_nolocal, NO impuesta: más atajos vivos = más caliente. Verificar que decae con
  la distancia al centro de expansión — pero MEDIDO, no asignado.
- **d_s, diámetro, clustering** del gigante en cada T (¿pasa de log N a √N durante la trayectoria?).

### Los cuatro brazos (falsación)
- **inflar_dist** (real): ruptura ∝ exp(−ℓ/T), los largos primero.
- **null_corte_azar**: rompe el MISMO nº de atajos por paso, pero al azar (ignora ℓ). Discriminante clave: debe
  dar "lejos" (diámetro √N) pero SIN gradiente de energía ordenado.
- **null_sin_enfriar**: sin proceso de T (poda instantánea). Prueba que el PROCESO importa, no solo el resultado.
- **inflar_barajado**: enfría, pero baraja qué nodo tiene qué E_nolocal al final (rompe la correlación
  espacio-energía manteniendo los totales). Aísla si el gradiente es espacial o solo un histograma.

### El juez — IRC-SEGURO por construcción (aporte del paper de Schwartz, integrado)
El tensor de orientación T (= tensor de esfericidad de QCD) que juzga direcciones DEBE ser infrared-collinear
safe: su conteo de ejes NO cambia si (IR) se añade un modo de energía casi cero, ni si (colineal) una dirección
se subdivide en dos casi-paralelas. Verificado por CS: el juez crudo cuenta direcciones FANTASMA de modos suaves
(1 jet real + soft en 2 ejes → crudo dice 3, IRC-seguro dice 1). Requisito pre-inscrito:
- **Piso de energía (IR):** ignorar modos de orientación bajo umbral de "energía" (peso). Los modos FRÍOS no
  cuentan como direcciones hasta ganar energía.
- **Invariancia colineal:** el conteo no cambia si se subdivide una dirección (test plantado obligatorio).
- **Conexión con el mecanismo:** la seguridad IR ES el enfriamiento. Un modo frío (energía→0) es infrarrojo y
  sale del conteo — que es lo que el enfriamiento-por-distancia hace físicamente. El juez IRC-seguro y la criba
  térmica son LA MISMA criba. Por eso no es higiene pegada: es el mecanismo mirado desde la medición.

## LECTURA PRE-INSCRITA (sin tunear tras ver números)
1. **¿El sustrato se vuelve métrico?** inflar_dist debe llevar diámetro de ~log N a ~√N Y d_s hacia ~2-3 estable,
   SUPERANDO a null_corte_azar en el GRADIENTE (no en el diámetro, que ambos logran). Discriminante:
   corr(distancia_al_centro, E_nolocal) fuertemente negativa en inflar_dist, ~0 en null_corte_azar y en barajado.
   Si null_corte_azar iguala el gradiente → el orden no importa → mecanismo no específico (negativo honesto).
2. **Sobre el sustrato métrico, ¿ahora encienden las direcciones?** Re-correr el juez IRC-seguro de CS067 sobre
   el tejido inflado. Si pico_medio cruza 0.85 CON especificidad (completo > controles) donde CS067 falló → la
   precondición métrica era lo que faltaba: el arco avanza. Si sigue smear aun sobre tejido métrico → las
   direcciones necesitan algo más que "lejos" (resultado que reorienta de nuevo, honesto).

## QUÉ NO HACE CS068
No fija dimensión. No calibra el ritmo de enfriamiento para que dé 3. No impone T(r). No toca el motor heredado.
Mide un proceso (estirar-enfriar) y pregunta si el sustrato resultante es métrico y si SOBRE él las direcciones
que CS067 no pudo encender, ahora encienden. El número de direcciones, si aparece, EMERGE.

— CS 🐝
