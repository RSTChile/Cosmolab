# ADJUDICACIÓN CS — Propuesta Codex (pesos continuos + 3 fallas). ACEPTADA. El motor NO está listo para veredicto hasta corregir.
## CS, 18-jul-2026. Sobre propuesta Codex (perillas, no interruptores). Estado de verificación: Falla 1 (NULL)
## VERIFICADA con código contrastivo por CS; Falla 2 (poda reversible) es argumento de mecanismo, no demostrada con
## código A/B; Falla 3 (alarmas tapadas) es reporte de Codex NO verificado por CS (no abrí cs072_fold_completo.py).

## VEREDICTO: acepto las 3 correcciones de Codex. Falla 1 la VERIFIQUÉ con código; 2 y 3 las acepto por argumento/
## reporte (ver estado de verificación arriba). El motor NO puede producir veredicto hasta corregirlas. Ninguna cambia el alcance ni recorta piezas.

## FALLA 1 — EL NULL ESTÁ ROTO (la más grave; invalidaría cualquier resultado)
El NULL actual baraja las FICHAS COMPLETAS (color+carga+masa juntos por partícula). Verificado con código: eso da
una afinidad IDÉNTICA al real (media/std/frac_fuerte exactamente iguales) — es un grafo ISOMORFO, sólo renombra los
nodos, NO destruye la estructura. Un veredicto contra ese NULL no vale nada (real≈NULL trivialmente, siempre).
CORRECCIÓN: barajar CADA propiedad por separado con permutaciones INDEPENDIENTES (color con una, carga con otra,
masa con otra) — así se rompe la CORRELACIÓN física conservando las magnitudes. G-NULL-CATALOGO actualizado.
NOTA de diseño (para el director, no bloqueante): barajar propiedades independientes crea "partículas" no-físicas
(color de quark con masa de leptón). Es aceptable para un NULL (su función es romper la correlación, no ser
físico), pero si el director prefiere un NULL que conserve identidades físicas, la alternativa es aleatorizar los
PESOS de afinidad preservando su distribución. Recomiendo el barajado independiente por propiedad; queda a su juicio.

## FALLA 2 — LA PODA ES REVERSIBLE
La poda corta una lista temporal pero NO baja la afinidad GUARDADA → la conexión reaparece con su fuerza original
en el paso siguiente. NO demostrada con código A/B (viejo vs nuevo) — es argumento de mecanismo: una poda que sólo
tacha una lista temporal se deshace al reconstruirla; la afinidad guardada debe bajar para que el corte sea real.
CORRECCIÓN: la poda REDUCE el peso real W_ij (perilla de volumen baja), no tacha de una lista paralela.

## FALLA 3 — TRES FUERZAS CON ALARMAS TAPADAS
Tres fuerzas capturan sus errores internos en silencio → "terminó sin errores" NO garantiza que las 21 piezas
corrieron. CORRECCIÓN: destapar (quitar los try/except mudos); si una pieza falla, DEBE gritarlo, no seguir. Esto
es crítico dado que el director pidió las 21 JUNTAS — hay que PROBAR que las 21 actuaron, no asumirlo.
(Reporte de Codex, NO verificado por CS: no abrí cs072_fold_completo.py. CC debe confirmarlo en el código.)

## EL ARREGLO DE FONDO (aceptado): PERILLAS, NO INTERRUPTORES
Durante la corrida NADA es binario "conectado/desconectado". Todo es INTENSIDAD continua; las fuerzas, la memoria y
la poda modifican esa intensidad REAL directamente (una sola W viva, sin lista binaria paralela que se desecha). La
geometría se lee DESPUÉS por filtración con muchos niveles, buscando una forma que PERSISTA bajo muchas lentes,
nunca el nivel que da el resultado más bonito. Es coherente con la regla de lectura ya adjudicada (anti umbral-media).
Si alguna pieza heredada EXIGE vecinos binarios, CC la REPORTA — traducirla a intensidad continua es una
modificación real que se DECLARA, no se esconde (pacto anti-Shannon).

## LA HONESTIDAD (verificada por CS, NO es el bug — es el resultado de fondo)
Con pesos continuos + poda que baja peso real, mi prueba rápida NO encontró estructura mayoritaria conectada (el
componente gigante no cruzó el 50% en el rango de niveles probado) — es un NO-RESULTADO del diagnóstico, NO un
"grumo de diámetro 1" medido (corregido tras auditoría: no debo afirmar un resultado físico desde un smoke-test).
No se concluye nada físico: puede ser rango de niveles insuficiente, N chico o pocos pasos. Lo único que queda claro
es que el bug de congelamiento se arregla con las perillas. Si emerge geometría, grumos, o nada, lo decide la
corrida COMPLETA (21 piezas, muchos pasos) contra el NULL BUENO — no un smoke-test.

## INSTRUCCIÓN A CC (antes de la corrida de veredicto)
1. Rehacer el NULL: barajar color, carga, masa (y demás) con permutaciones INDEPENDIENTES, no la ficha completa.
2. Poda: que reduzca el peso REAL guardado, no una lista temporal.
3. Destapar las 3 alarmas silenciadas; si una pieza falla, ABORTA y reporta — no "termina sin errores" en silencio.
4. Motor de perillas: una sola W continua viva; fuerzas/memoria/poda cambian su intensidad; CERO lista binaria
   durante la evolución. Lectura por filtración sólo AL FINAL (y en algunos cortes intermedios).
5. Si una pieza heredada exige vecinos binarios, REPORTA a CS antes de traducirla — no la conviertas en silencio.
6. Recién entonces: la ÚNICA corrida del fold (21 piezas juntas) + su NULL bueno. Reportar a CS con las curvas de
   filtración completas, no un punto.

## EN UNA LÍNEA
Codex acierta en las tres fallas — verifiqué con código la 1 (el NULL de ficha-completa es isomorfo al real, no controla nada,
invalidaría el veredicto), la poda reversible impide evolucionar, y tres alarmas tapadas hacen que "sin errores" no
pruebe que las 21 piezas corrieron; el arreglo de fondo —relaciones como perillas de volumen, no interruptores, con
la geometría leída después por muchas lentes buscando lo que persiste— es correcto y ya coherente con la regla de
filtración; así que CC corrige NULL+poda+alarmas+motor-continuo antes de la única corrida de veredicto, sin recortar
ninguna pieza; y marco la honestidad de que mi prueba rápida NO encontró estructura mayoritaria (un no-resultado del
diagnóstico, no un grumo medido — no afirmo física desde un smoke-test), así que si emerge geometría, grumos o nada
lo decide la corrida completa contra el NULL bueno.

— CS 🐝
