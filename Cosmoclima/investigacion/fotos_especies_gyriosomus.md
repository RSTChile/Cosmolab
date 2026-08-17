# Fotos reales por especie en el mapa (05-ago-2026)

A pedido de Alexis: generalizar lo que antes solo hacía *G. kingi* (foto + icono
propio en el mapa) a todas las especies posibles, sin fabricar ni reusar fotos de
otra especie "parecida".

## Cómo se buscaron
`Web/prueba_de_concepto/buscar_fotos_especies.py` — primero se probó la API de GBIF
(fallaba: solo 3 de 44 especies, porque el backbone taxonómico de GBIF no tiene
alineadas la mayoría de las 44 especies del género a nivel de especie). Se cambió a
la API de iNaturalist directamente (su propia taxonomía, no depende del backbone de
GBIF) — para cada especie, busca observaciones con foto, filtra por licencia
Creative Commons real (no "todos los derechos reservados"), y **verifica que el
nombre de la observación calce EXACTO** con la especie buscada (no acepta el género
solo, ni una especie distinta).

## Resultado: 25 de 44 especies con foto real
Guardado en `Web/prueba_de_concepto/datos_fuentes/fotos_especies.csv`
(especie, url_foto, licencia, atribución del usuario de iNaturalist, link a la
observación original). Las 19 especies sin foto simplemente no tienen ninguna
observación de iNaturalist identificada a nivel de especie exacta — quedan sin foto
en el mapa, no se les asignó una de otra especie para "rellenar".

## Cómo se integró al mapa
`generar_mapa.py` ahora lee ese CSV y agrega un campo `foto` a cada especie dentro de
`especiesData`. En el HTML, el mecanismo de *G. kingi* (icono a medida, PNG propio)
se dejó intacto tal cual estaba — es un asset curado a mano, con datos
biogeográficos escritos específicamente para esa especie. Para las otras 24 con foto
nueva se generalizó un mecanismo genérico: un marcador circular con la foto de
iNaturalist en miniatura, que al pincharlo muestra la foto grande, la especie, el
usuario de iNaturalist que la tomó, la licencia, y un link a la observación original.
El ícono 📷 en el listado lateral ahora aparece para cualquier especie con foto, no
solo kingi.

## Licencias — ojo si el mapa se vuelve público/comercial
La mayoría de las fotos encontradas son **CC BY-NC** (uso no comercial). Para un
instrumento de investigación/divulgación como este está bien — si en algún momento
el mapa se usa con fines comerciales, revisar `fotos_especies.csv` columna
`licencia` y filtrar por CC BY / CC0 solamente.

## Para ampliar más adelante
Correr `python3 buscar_fotos_especies.py` de nuevo en cualquier momento (por ejemplo
si aparecen observaciones nuevas en iNaturalist para alguna de las 19 especies sin
foto) y después `python3 generar_mapa.py` para que el mapa las recoja.
