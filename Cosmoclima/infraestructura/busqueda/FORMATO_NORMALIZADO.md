# Formato de salida para integrar al índice de activos

Un CSV por lote en `busqueda/normalizado/<lote>.csv`, con **exactamente** estas
columnas y en este orden:

    item,elemento,nombre,lat,lon,comuna,region,fuente,confianza

- `item`      número del ítem MICR (texto: "441", "847"…)
- `elemento`  nombre EXACTO del ítem tal como aparece en el catálogo
- `nombre`    nombre del activo. Vacío si la fuente no lo trae; **nunca "None"**
- `lat`,`lon` grados decimales EPSG:4326. Chile: lat entre −56 y −17,
              lon entre −110 y −66. Descarta lo que caiga fuera y dilo
- `comuna`    nombre de comuna que declare la fuente (respaldo cuando el punto
              cae en el mar y no resuelve por polígono). Vacío si no la trae
- `region`    igual
- `fuente`    organismo · producto, corto ("SUBTEL · Ley de Torres")
- `confianza` `consolidado` (registro público georreferenciado) ·
              `derivada` (coordenada heredada de otro activo) ·
              `baja` (fuente no oficial o coordenada aproximada)

## Reglas

1. **Un activo por fila.** Si una capa es de líneas o polígonos, usa el vértice
   central de la traza (NO el centroide: en un canal curvo cae fuera) y anótalo
   en el informe.
2. **No inventes coordenadas.** Nada de geocodificar direcciones. Si la fuente
   no publica coordenada, ese activo no entra.
3. **Asigna un ítem sólo si el dato lo identifica.** Si una fuente mezcla varios
   ítems y no puedes separarlos, déjala fuera y dilo. Ya nos pasó con
   «Producción de alimentos» del RETC: repartirlo habría sido inventar.
4. **No dupliques lo que ya está.** El índice ya tiene 44 ítems poblados; si tu
   fuente cubre uno de ésos, decide si es mejor o si duplica, y dilo. La fusión
   descarta automáticamente lo que caiga a menos de 150 m de un activo ya
   existente del mismo ítem, pero mejor no mandarlo.
5. **Declara lo que dejas fuera** al final de tu informe: qué fuente, cuántos
   registros y por qué.
