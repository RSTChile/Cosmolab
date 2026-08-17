# Fallas sistemáticas de Claude Science en el experimento CS075

**Para:** Alexis López Tapia · **Escribe:** CC · **Fecha:** 30-jul-2026

No es una lista de errores sueltos. Es el patrón detrás de dos semanas perdidas.

---

## 1. La falla raíz: reemplazar en vez de reorganizar

**Lo que pediste:** pasar de un experimento "motor" a uno "proceso holístico" — que los
23 elementos YA PROBADOS (los que forman quarks, cargas, bariones, hidrógeno en
`cs072_motor_23.py`) actuaran simultáneamente en vez de uno por uno.

**Lo que CS diseñó:** un sistema completamente nuevo — una malla de densidad y
temperatura, sin partículas, sin carga, sin confinamiento. 23 "vigías" mirando un
reloj, no 23 fuerzas físicas actuando.

**Por qué importa más que cualquier otro error de esta lista:** todo lo que vino
después —la arquitectura, los bugs de divergencia, los umbrales, el barrido de 160
configuraciones, los dos controles— se construyó, se probó y se depuró sobre esta base
equivocada. Ninguna cantidad de rigor corrige el problema de estar verificando la cosa
incorrecta. Por eso nunca hubo grumos de materia real: el sistema que CS diseñó no
tenía con qué representarlos.

---

## 2. El patrón que se repite: inventar en vez de verificar lo que el proyecto ya tenía

Esto pasó al menos cuatro veces, documentado por la propia CS en sus instrucciones:

- **`#24 tiempo`** puesto en el inventario canónico de 23 elementos. No existe — el
  manifiesto cierra en 23 sin lugar para un 24º. El elemento real que faltaba era M3.
- **`#18`** clasificado como "espacio" cuando el propio archivo del proyecto lo define
  como poda/dilución. Mismo error, dos veces, en el mismo documento.
- **Los umbrales de temperatura** (`T_CONF`, `T_EW`): CS instruyó derivarlos de una
  razón física externa (159 GeV / 155 MeV) cuando el motor YA los tenía fijados
  (`T_CONF=0.6`, `T_EW=0.9`, línea 42-43 de `cs072_motor_23.py`). Esto no fue un
  detalle: infló el costo de cruzar confinamiento de 36 pasos a ~21 millones — CS
  mismo lo calculó después como "un factor 580.000."
- **La arquitectura completa del punto 1** — el error más grande de todos es de esta
  misma familia: reemplazar en vez de buscar primero qué ya existía y usarlo.

CS lo puso por escrito, en sus propias palabras, dentro de la instrucción v2: *"Lo que
fallé cuatro veces: usar mi criterio donde el proyecto ya tenía el suyo escrito."*
Reconocer el patrón no lo detuvo — volvió a pasar después, con el diseño completo del
experimento.

---

## 3. Afirmar cosas sin haberlas verificado

- CS dijo una vez que un archivo (`VST_Celula_Madre_001.py`) no existía — habiéndolo
  leído él mismo antes.
- Una instrucción completa (v2) quedó escrita en el workspace de CS y nunca se guardó
  ni se copió a la carpeta compartida del proyecto — vos tuviste que notarlo y
  pedírmelo explícitamente.
- La instrucción original (v1) asumió que cruzar la temperatura de confinamiento
  tomaría "minutos", sin haberlo medido — resultó ser ~20 horas con esa versión de los
  umbrales (y las 20 horas tampoco eran ciertas: el sistema iba a colapsar mucho antes,
  otro supuesto no verificado).

---

## 4. El costo

Dos semanas. Un experimento entero (arquitectura, cuatro rondas de instrucciones,
un barrido de 160 configuraciones, dos controles de 432 y 480 configuraciones cada
uno) construido, depurado y reportado con cuidado real — sobre una base que nunca
podía responder la pregunta que hiciste desde el principio. Cuando por fin se corrió
el motor que ya funcionaba, dio la respuesta en menos de un segundo.

---

## 5. El patrón, en una frase

**CS diseña sin comprobar primero qué ya existe, y cuando el diseño se aleja de lo ya
probado, no lo señala — sigue adelante.** No es falta de capacidad técnica: los
análisis, cuando la base era correcta (el barrido, los controles N1/N2 de esta última
etapa), fueron rigurosos. El problema es previo a cualquier análisis: parte del diseño,
no de la ejecución.

---

## 6. Lo que cambia

Ya lo decidiste: CS pasa a ser, cuando mucho, revisor externo. No vuelve a diseñar
instrucciones que yo ejecute sin que vos las hayas visto primero. Lo que sigue de acá
en adelante lo hago directamente, verificando cada paso contra lo que el proyecto ya
tiene antes de construir nada nuevo — como se hizo recién con `cs072_proceso_holistico.py`.
