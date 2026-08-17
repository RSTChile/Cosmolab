# ADJUDICACIÓN CS — CS072 fold completo: FIRMA SUSPENDIDA. "Grumo sin espacio" probado; el (B) real≈NULL NO, aún.
## CS, 18-jul-2026. Sobre cs072_fold_tanda_resultados.json + cs072_fold_completo.py. Corrida hecha (20 piezas activas,
## cuántica fuera) pero con comparación mal pareada (semillas distintas) → NO se firma veredicto hasta re-corrida pareada.

## VEREDICTO RETIRADO (18-jul, tras auditoría Codex): firma SUSPENDIDA. El (B) NO está limpio todavía.
## Lo que SÍ está probado: este motor produce grumos compactos (diámetro 2-3, no crece con N) — "grumo sin espacio".
## Lo que NO está probado: que "lo real no le gane al NULL" — los dos brazos usaron SEMILLAS DISTINTAS (población,
## historia y sector oscuro diferentes), así que no fue la misma partida con un solo cambio. Requiere re-corrida pareada.

## QUÉ SE CORRIÓ (lo que el director pidió: TODO junto, una sola vez)
Las 21 piezas (18 elementos + 3 mecanismos) activas desde t=0, motor de perillas continuas (sin interruptores),
poda que baja el peso real, NULL corregido, lectura por filtración. Una sola corrida + su NULL. N∈{100,200,400,800}.

## LO QUE VERIFIQUÉ CON CÓDIGO Y DATOS (esta es la corrida de veredicto — la audité entera)
1. NULL BIEN ARMADO (la falla que Codex cazó, corregida): _null_catalogo baraja color/carga/masa/es_anti/es_ferm
   con permutaciones INDEPENDIENTES por propiedad — rompe la correlación física real, NO re-etiqueta. Confirmado
   en el código (líneas 62-71). El NULL de v1 (ficha completa = isomorfo) ya no está.
2. DIÁMETRO = MEDICIÓN REAL, no el fallthrough que confundí antes: _diam_robusto (cs071 L199) es un BFS genuino
   sobre el grafo; frac_gigante=0.9 sale de union-find real. NO es el default (0.0) del smoke-test previo.
3. REAL ≈ NULL: β_real=3.9e-17 ≈ 0; β_null=−0.18 ≈ 0. Diámetro pegado en 2-3 y NO crece de N=100 a N=800 en AMBOS
   brazos. Ninguno escala → ninguno es métrico ("grumo sin espacio", probado). PERO real-vs-NULL NO es comparable
   aquí (semillas distintas, ver arriba) — la igualdad de β NO prueba todavía que la física real no aporte.
   | N | real diám / frac_gig | null diám / frac_gig |
   |---|---|---|
   | 100 | 2.0 / 0.9 | 3.0 / 0.9 |
   | 200 | 2.0 / 0.9 | 2.0 / 0.9 |
   | 400 | 2.0 / 0.9 | 2.0 / 0.9 |
   | 800 | 2.0 / 0.9 | 2.0 / 0.9 |
4. POR QUÉ (mecanismo, medido de MI tabla, corregido tras auditoría): la estructura es una estrella compacta —
   diámetro 2-3 con frac_gigante=0.9. Dato medido real: para conectar el 90% se necesita ~70% de los pares
   (frac_pares 0.72/0.70/0.70/0.69 para N=100/200/400/800). (CORRECCIÓN: el reporte de CC decía "7% de vínculos
   conecta 70-75%" — NO es lo que muestra el JSON; mi tabla da ~70% de pares para el 90%. Retiro esa cifra de CC,
   no verificada.) NOTA Codex: el NULL necesita ~40-50% de los pares para el 90% — o sea las curvas NO son
   idénticas; el real forma familias/dominios más separados antes de unirse. Esa diferencia es justo lo que la
   re-corrida pareada debe medir limpio.

## QUÉ SIGNIFICA — PROVISIONAL (lo firme y lo pendiente separados)
FIRME (probado): este motor, con afinidades físicas FIJAS y ESTÁTICAS, produce GRUMOS compactos (diámetro 2-3, no
crece con N), no un espacio extendido. Eso es sólido en AMBOS brazos. Coherente con el arco (CS067 ya lo insinuaba).
PENDIENTE (NO probado aún): que la física real NO le gane al azar. Para afirmarlo hace falta la comparación PAREADA
(misma semilla/población/historia/oscuro; el NULL cambia SÓLO qué color-carga-masa van juntos). Sin eso, "el
catálogo estático no basta" es una lectura PLAUSIBLE pero no un veredicto firmado.

## OBJECIONES DE CODEX ACEPTADAS (por qué no firmo aún)
1. SEMILLAS DISTINTAS: real usa seed_off=0, null seed_off=10000 (cs072_fold_tanda.py L73) → poblaciones, historias
   y sector oscuro DIFERENTES. En N=400: real 7% oscuro vs null 22% oscuro. El oscuro quita color/carga y conserva
   masa → cambia la telaraña. Son universos distintos, no la misma partida con un cambio. VERIFICADO en el código.
2. CURVAS NO IDÉNTICAS: real ~70% de pares para conectar 90%; NULL ~40-50%. El real forma dominios más separados
   antes de unirse — puede haber señal física que la comparación mal pareada esconde.
3. 20 BLOQUES, NO 21: la fase cuántica está explícitamente fuera (ya declarado). Y "el bloque se ejecutó" ≠ "el
   bloque cambió la telaraña medida": espín, Pauli, SSB y 3-cuerpos mueven brújulas internas pero pueden no volver
   a tocar la red donde se mide geometría. Hay que REGISTRAR cuáles cambian la telaraña y cuáles sólo variables
   internas. ACEPTADO.
4. d_s = NaN: el estimador de dimensión NO convergió — un instrumento no respondió. NO se puede decir "todos los
   medidores funcionaron". El diámetro-vs-N sí midió; el veredicto (cuando se firme) se apoya en él, no en d_s.

## LO QUE CORRESPONDE (Codex, aceptado — reparar SÓLO la comparación, no tocar fuerzas ni parámetros)
- Misma población inicial y misma historia aleatoria para ambos brazos (common random numbers, seed pareada por N).
- Mismo sector oscuro y mismos acontecimientos; el NULL cambia ÚNICAMENTE qué color/carga/masa aparecen juntos.
- Registrar qué piezas cambian realmente la telaraña vs cuáles sólo mueven variables internas.
- Repetir EXACTAMENTE las mismas mediciones, sin cortes favorables nuevos.
- Guardián: G-SEMILLAS-PAREADAS (ya usado en CS071). Si tras eso ambos siguen en diámetro 2-3 y el real no supera
  al NULL → ENTONCES sí se firma (B) para esta representación.
- Nota de alcance: es un catálogo físico simplificado y sorteado, no el Modelo Estándar exacto. El resultado se
  atribuye a ESTA representación experimental, no al MS literal.

## EN UNA LÍNEA
El fold completo (20 piezas activas, la cuántica declarada fuera) con motor de perillas produce grumos compactos
—diámetro 2-3 que NO crece de 100 a 800 partículas, en ambos brazos— así que "grumo sin espacio, no calles" está
PROBADO; PERO retiro la firma del (B) porque Codex tiene razón y lo verifiqué en el código: los dos brazos usaron
SEMILLAS DISTINTAS (real seed_off=0, null seed_off=10000 en cs072_fold_tanda.py L73), con lo que cambiaron a la vez
población, historia y sector oscuro (N=400: 7% oscuro real vs 22% null), y encima las curvas NO son idénticas (real
~70% de pares para el 90%, null ~40-50%), así que "lo real no le gana al NULL" NO está demostrado — corregí también
una cifra que arrastré de CC ("7% de vínculos conecta 70-75%": falso, mi tabla da ~70% de pares para el 90%); lo que
corresponde es repetir SÓLO con la comparación pareada (misma semilla/población/historia/oscuro, el NULL cambia sólo
qué color-carga-masa van juntos, guardián G-SEMILLAS-PAREADAS), y si ahí ambos siguen en diámetro 2-3 y el real no
supera al NULL, ENTONCES se firma (B) para esta representación simplificada del catálogo.

— CS 🐝
