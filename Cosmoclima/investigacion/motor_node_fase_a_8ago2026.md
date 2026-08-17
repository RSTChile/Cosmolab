# Fase A: motor de física portado a Node.js — verificado

Parte del plan "Instrumento EIT-3 Térmico — granularidad fina + workflow de
experimentos" (`/Users/alexis/.claude/plans/majestic-whistling-canyon.md`).

## B.0 — resuelto primero (era el mayor riesgo del plan)

`cr2_prAmon_2019/cr2_prAmon_2019.txt` (producto MENSUAL de CR2, estación
Huintil 04723002) para 2017-06 a 2018-12 coincide **exacto, mes a mes**
(incluido el sentinel -9999→null) con `PLUVIOSIDAD_MENSUAL` del
instrumento. Confirma: ese tramo de 19 meses es real pero solo existe en
mensual — no hay diario disponible ahí, es un tercer tramo de resolución
honesto, no un error ni una fuente sin documentar.

**Tres tramos de resolución de la lluvia real, confirmados:**
- 1966-01 a 2017-05: diario real (estación Huintil, CR2)
- 2017-06 a 2018-12: mensual real (mismo CR2, producto mensual, sin diario)
- 2019-01 en adelante: mensual real (NASA POWER/DMC)

## Motor portado a Node.js (`Web/prueba_de_concepto/motor/`)

- `generar_motor_node.py`: extrae la física del HTML (`prueba_de_concepto_ET3-Termico_con_mapa.html`)
  por nombre de símbolo, con matching de llaves balanceadas — copia literal,
  nunca reescribe. Único cambio deliberado: `KAPPA_V/O/LF/DELTA` pasan de
  `const` a `let` (vía getter/setter en el export) para que el runner de
  baterías pueda pisarlos entre corridas. Salida: `motor_fisico.generado.js`
  (nunca se edita a mano, se regenera corriendo el script de nuevo).
- Iteración real: la primera corrida de prueba tiró 5 `ReferenceError`
  distintos (`DAISY_K`, `SAT_VENTANA`, `pseudoNoise`, `abioticTf` faltantes)
  — cada uno se agregó a la lista de extracción y se corrigió, confirmando
  que el diseño "falla fuerte e inmediato ante una función faltante" del
  plan funciona como se esperaba (no silenciosamente).

## Verificación (Fase A.2 del plan) — máxima confianza posible

1. **5.000 ticks, misma semilla, navegador real vs. Node**: 5 de 6 campos
   (Tf, LF, A_sys_env, e_R, zona) salieron **idénticos byte a byte**. El
   sexto (Δ_struct) difiere en el dígito 13 (ruido de punto flotante entre
   motores V8 distintos, no un error de lógica — la grilla 64×64 suma miles
   de valores, y esa acumulación puede diferir en el último bit entre
   Chrome y Node aunque el código sea idéntico).
2. **Corrida completa real, la prueba que de verdad importa**: corrí
   `verificar_experimento_completo.js` — la MISMA configuración que el
   botón "▶ Experimento Completo" del HTML (parámetros de fábrica, Día/Noche
   +Estaciones on, semilla `regimen1966-2027`, 60 días de asentamiento,
   1966-2027 completo, 1,36 millones de ticks) — y comparé el CSV resultante
   contra el CSV que Alexis ya había bajado del HTML real
   (`regimen_1966_2027_calibracion_real (2).csv`). **`diff` da CERO
   diferencias — los 62 años, idénticos.**

## Velocidad real medida (no prometida de antemano)

La corrida completa de verificación tardó **10.8 minutos** (1,36M ticks),
con el ritmo subiendo de 1.639 a 2.094 ticks/seg a medida que V8 optimiza
las funciones más calientes (JIT warm-up) — más rápido que el rango medido
en el navegador (~900-1.850 ticks/seg), aunque no una mejora de órdenes de
magnitud. La ganancia real para "varios agentes en paralelo" sigue siendo
correr N configuraciones a la vez, no que una corrida individual sea
dramáticamente más rápida — tal como preveía el plan.

## Runner de batería (`runner_bateria.js` + `experimentos/child_worker.js`)

Un proceso de sistema operativo por configuración (`child_process.fork`),
concurrencia acotada a núcleos-1. Probado con un arnés de humo (4 configs,
una diseñada para fallar a propósito): confirmado que una config que falla
no tumba a las demás, y el resumen final junta correctamente solo las que
sí terminaron. Config de ejemplo en `experimentos/ejemplo_kappa.json` (4
configuraciones: κ actuales, κ viejos baseline, κ con p75, diagnóstico
aislado — listas para correr en cuanto se decida usar la batería).

## Qué falta de la Fase A (menor, no bloqueante)

El plan proponía un arnés de verificación con Playwright instalado aparte;
en su lugar usé la herramienta de automatización de navegador ya disponible
en esta sesión, que logra exactamente la misma verificación (punto 1 y 2
arriba) sin agregar una dependencia nueva al repo — decisión pragmática,
mismo resultado. Si en algún momento hace falta reverificar sin esta
herramienta disponible, `verificar_experimento_completo.js` + el propio
botón del HTML + `diff` bastan (ya lo demuestran los resultados de arriba).
