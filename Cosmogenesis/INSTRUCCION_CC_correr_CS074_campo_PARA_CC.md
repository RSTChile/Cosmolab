# INSTRUCCIÓN PARA CC — CORRER CS074 (campo continuo). SOLO EJECUTAR.

**Fecha:** 21-jul-2026
**De:** CS (director del experimento, vía Claude Science)
**Para:** CC
**Archivo a correr:** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cs074_persistencia_campo.py`
(idéntico al artifact version_id `b5c48879-2925-4804-8bd5-f474ea5b1f92`)

---

## REGLA CERO — NO TOCAR EL CÓDIGO. NI UNA LÍNEA.

El código ya pasó cuatro rondas de revisión del equipo y está adjudicado por CS.
Tu tarea es **CORRERLO, NO MODIFICARLO.**

- **PROHIBIDO** cambiar, "mejorar", "optimizar", "corregir", refactorizar o ajustar
  cualquier parte del código — funciones, constantes, parámetros, el observable,
  el NULL, la difusión, la expansión, las escalas, el barrido. NADA.
- **PROHIBIDO** cambiar N, pasos, semillas, la lista de ε o la lista de H.
- Si crees ver un error: **NO lo arregles. PÁRATE, NO corras, y repórtalo a CS**
  con la línea exacta y por qué. CS decide. Un desacuerdo tuyo es un DATO para CS,
  no una licencia para editar. (Ya van varias veces que el experimento se retrasa
  porque se modificó a criterio propio: esta vez no.)
- No crees una "versión CC", no dupliques el archivo con cambios, no parchees en
  caliente. Corres EXACTAMENTE ese archivo, tal cual está en el disco.

## REGLA UNO — EJECUTAR COMPLETO, DE UNA SOLA VEZ. NO POR PARTES.

- Corres el **barrido de producción entero en una sola ejecución**:
  ```
  cd /Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis
  python3 cs074_persistencia_campo.py produccion
  ```
  (usa el intérprete con numpy que ya usas para el motor.)
- Es N=800, pasos=120, semillas=12, ε=[0, 1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0],
  H=[0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]. **Déjalo correr entero.**
- **PROHIBIDO** trocear el barrido, correr "solo unas filas para ver", hacer un
  smoke previo, o correr por etapas. El experimento se mira COMO TODO, no por pedazos.
- Si tarda, que tarde. Hemos corrido cosas de horas. No lo interrumpas por lentitud.
- Si el proceso FALLA (excepción, se cae), **NO arregles el código**: manda a CS el
  traceback completo tal cual, y el punto donde falló. CS aborda dónde falló.

## QUÉ ES ESTE EXPERIMENTO (para que corras entendiendo, NO para que cambies)

Campo continuo con una variación de amplitud ε (una "mancha", como una mancha solar
= misma sustancia a distinto valor, NO un conjunto de cosas). Compiten difusión
(el campo se re-aplana, borra la diferencia) y expansión (estira el dominio, corta
acoplamientos, congela la diferencia). Se mide si la diferencia PERSISTE vs un NULL.

**NOTA QUE NO DEBES "CORREGIR" — es deliberada y es la premisa del experimento:**
La difusión corre **SOLO por acoplamientos vivos**. Cuando la expansión corta un
acoplamiento, corta también el canal de difusión por ahí. Esto es CORRECTO y a
propósito: **expandir ES aislar** — dos regiones que se separan dejan de poder
intercambiar; no hay difusión "a través" de lo ya separado (eso presupondría un
espacio métrico con distancias fijas, que este experimento NIEGA). La difusión sí
sigue actuando DENTRO de cada región aún conexa (ahí está la carrera real). Si
"arreglas" esto haciendo que todo difunda con todo, ROMPES el experimento. No lo toques.

Igual: las escalas de temperatura (10^20→10^10 K) y tiempo (10^-20→10^-4 s) son
SOLO reporte/mapeo — NINGUNA regla dinámica las lee. No las conviertas en motor.

## QUÉ ENTREGAR A CS (resultado crudo, sin interpretar, sin adjudicar)

1. La **salida JSON completa** que imprime el script (todas las filas del barrido).
   No la resumas ni la recortes: la curva ENTERA (todos los ε × todos los H).
2. Tiempo total de corrida y pico de RAM (como sueles reportar).
3. Si hubo algún warning de numpy, cópialo tal cual.
4. **NO adjudiques.** No digas "persiste" o "no persiste". Eso lo hace CS con la
   curva completa a la vista. Tú entregas los números crudos.

---

**Resumen en una línea:** corre `cs074_persistencia_campo.py produccion` tal cual,
entero, de una vez; no cambies ni una línea; si algo falla o ves algo raro, párate
y repórtalo a CS con el detalle exacto — no lo arregles tú.
