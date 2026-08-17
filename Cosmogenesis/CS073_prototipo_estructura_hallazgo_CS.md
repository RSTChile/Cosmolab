# CS073 — Prototipo: emergencia de estructura por gravedad-vs-expansión (hallazgo)

**De:** Claude Science (CS) — prototipo, NO toca el motor compartido (`p02_gravedad` congelado).
**Fecha:** 19-jul-2026
**Regla:** desarrollo y pruebo yo primero; esto es la base de la coordinación con CC. No cierra ningún experimento.

## Qué se probó
Sobre la red `Bgrav` que una corrida post-átomo ya produjo (152 átomos, 5738 enlaces, peso =
masa×densidad), simular la **expansión** como un umbral creciente θ que corta los enlaces
débiles (regiones tenues se estiran) y deja los fuertes (regiones sobredensas resisten).
Pregunta: ¿el hub se fragmenta en múltiples estructuras distinguibles?

## Resultado (crudo, sin maquillar)

| θ (percentil) | enlaces | #estructuras | gigante % | diámetro gigante |
|---|---|---|---|---|
| 0 (hub)  | 5738 | 1 | 100% | 4 |
| 50 | 2869 | 1 | 68% | 3 |
| 70 | 1722 | 1 | 49% | 2 |
| 85 | 861 | 1 | 34% | 2 |
| 92 | 459 | 1 | 26% | 2 |
| 96 | 230 | 1 | 20% | 2 |
| 98 | 115 | 1 | 14% | 2 |

**El gigante se encoge (100%→14%) pero NUNCA se fragmenta en múltiples estructuras:** siempre
1 sola comunidad, y el diámetro BAJA (4→2, más compacto, no métrico). Emergió **un grumo que se
erosiona soltando polvo**, no una red cósmica. La señal real ya no muestra estructura — no hizo
falta el NULL.

## Diagnóstico (el ingrediente que falta, sin ambigüedad)
La gravedad del motor liga por **peso = masa×densidad, sin localidad espacial**. Sin "cerca/lejos",
todo átomo pesado se liga con TODOS los demás pesados → colapsan en **un único grumo**, no en
estructuras separadas. **Estructura (múltiples entidades) requiere gravedad LOCAL:** un átomo
colapsa con sus vecinos, no con todo el universo sobredenso a la vez.

## Conexión con el arco (confirmación por el lado negativo)
- Distancia/dimensión/dirección se **fosilizan** con el átomo (potencial) — ya probado.
- Para volverse **campo vectorial medible** (estructura → expansión → física plena), la gravedad
  debe operar sobre la **métrica desplegada como POSICIONES**, no sobre el grafo relacional sin lugar.
- El motor actual **fosiliza la métrica pero no la despliega**: por eso da hub, y por eso el corte
  de expansión da un grumo, no una web. **La métrica de Bohr hay que USARLA.**

## Diseño del paso 4 (a coordinar CS+CC+Alexis, NO escribir aún)
1. **Desplegar la métrica fosilizada como posiciones**: distancias de grafo (que Bohr inauguró)
   → embedding métrico → cada átomo obtiene un lugar. (Análisis, reversible.)
2. **Gravedad LOCAL** sobre esas posiciones, compitiendo contra la expansión (Jeans: gravedad gana
   en escalas sobredensas, expansión gana en las tenues).
3. **Observable:** ¿emergen múltiples estructuras ligadas, con distancia/dirección medible ENTRE
   ellas, y distribución de masa/tamaño que le gane al NULL?
4. **NULL:** campo de densidad #23 barajado (sin sobredensidades coherentes → sin estructura).
5. **Guardián anti-Shannon:** cero centros sembrados; el umbral es sobre peso físico, no índice;
   la estructura debe GANARLE al NULL barajado o no cuenta.

## Estado
Prototipo negativo informativo: la estructura NO emerge del grafo relacional sin localidad.
El paso 4 requiere desplegar posiciones + gravedad local — a coordinar. Motor congelado.
