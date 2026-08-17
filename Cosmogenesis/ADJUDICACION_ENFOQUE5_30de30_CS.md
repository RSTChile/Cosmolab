# Adjudicación del Enfoque 5 (energía · exergía · entropía) — los 30 experimentos
### Veredicto de CS · en lenguaje simple · verificado en disco, no de palabra

**Director:** Alexis López Tapia · **Adjudica:** Claude Science (CS) · **Fecha:** 25-jul-2026
**PROPUESTA DE CIERRE — no cierra sin tu autorización explícita** (regla permanente).

---

## Lo que verifiqué yo mismo — y lo que NO (honestidad de método)

Antes de adjudicar abrí los datos crudos. Sé preciso sobre qué comprobé con mis manos y
qué tomé del informe de CC sin verificar, porque la regla de esta sesión es esa:

1. **VERIFICADO EN DISCO — el experimento clave (falsación contra el 5%/31,5%):** abrí el
   JSON de E5.3-5 y lo confirmé. Las celdas cuya eficiencia cae casi exacta sobre 4,9%
   tienen **z=0** (no distinguen nada del azar — el número lindo no significa nada), y la
   única celda con señal real (z=2,4) da 27,25%, lejos del 31,5%. **Donde hay número lindo
   no hay señal; donde hay señal el número no es lindo. Nunca coinciden.** El negativo es
   real. Esta es la afirmación de la que depende el veredicto, y está verificada.

2. **REPORTADO POR CC, matizado tras mirarlo yo — la "ceguera de la regla común":** CC
   afirma que la regla común da "z=0.0 exacto, bit a bit" contra el control de barajado.
   Abrí el JSON de E5.5-3 y **la afirmación no aparece en esa forma limpia:** lo único que
   da cero exacto es un chequeo físico distinto (`max_diff_Xcanonica=0.0`, que la
   definición canónica es consistente), NO un z contra el barajado. El "boost" de la regla
   común de hecho varía (0 a 0,998; 999 de 1664 filas dan boost positivo). Así que la idea
   de fondo —que la regla común, al ser una suma que no mira el orden, es mal instrumento
   para "¿es distinto del azar?"— es razonable, PERO el "z=0.0 exacto" tal como CC lo
   redactó no lo confirmé en los datos, y en una mirada rápida no calza. **Queda como
   afirmación de CC pendiente de verificación limpia, no como hecho establecido.**

---

## EL VEREDICTO EN UNA FRASE

**La idea de fondo se sostiene: la energía útil (exergía) es real, medible, se apaga
cuando todo se mezcla y se rescata cuando el universo se expande. PERO el modelo NO
reproduce el 5% de materia visible del universo real — da otro número (~27%), estable y
honesto. Y eso es una buena noticia, no una mala: significa que no hicimos trampa.**

---

## LO QUE QUEDÓ PROBADO (11 experimentos con PASS claro)

En simple, lo que el enfoque estableció de forma sólida:

- **La contabilidad de la energía cuadra.** Cuando la energía útil baja, el desorden sube
  exactamente lo mismo, sin inventar ni perder nada (E5.2-2: correlación −0,9999; E5.2-4
  y E5.2-5: la energía se conserva con precisión altísima). La tríada energía-exergía-
  entropía es coherente, no una metáfora.
- **La muerte térmica tiene energía pero no sirve — y NO es la Nada.** En el límite donde
  la diferencia se apaga, la energía total se queda completa (~1,0) pero la energía útil
  cae a cero (E5.5-4: en 240 de 240 casos). Confirma empíricamente la distinción del
  poema: muerte térmica (todo, inútil) ≠ Nada (∅). Esto es un resultado bonito.
- **Las estructuras grandes se congelan primero** (E5.4-4: 16 de 16 casos, precisión de
  máquina) — y una sola estructura grande se queda con el 84-88% de la energía útil.
- **La energía útil, una vez formada, se queda** (E5.3-3: se congela y no cambia por
  100.000 pasos).
- **Nuestro modelo NO tiene un "enfriador oculto"** (E5.4-5: el control con enfriador
  tramposo rompe la contabilidad; sin él, se respeta perfecta). Prueba de honestidad.
- **El resultado no es un truco del anillo 1D** (E5.1-3: se replica en 2D).

## LO QUE FALLÓ, HONESTAMENTE (4 con FAIL bien diagnosticado)

Ninguno es un fracaso vergonzante — cada uno enseña dónde está el límite:

- **La energía útil no se puede atar al enfriamiento por la vía que probamos** (E5.4-1,
  E5.4-2): el medidor no distinguía el caso real del control, porque comparten el mismo
  reloj de mezclado. La diferencia real está en cuánta energía queda al final, no en cómo
  se enfría. Hay que rediseñar ese medidor.
- **Las dos formas de medir energía útil no coinciden lo suficiente** (E5.6-1): coinciden
  en la forma, no en la escala. La definición todavía no está madura del todo.

## EL EXPERIMENTO CLAVE — el negativo que vale (E5.3-5)

La pregunta central del enfoque era: **¿el modelo escupe solo el 5% (o el 31,5%) de
materia del universo real, sin que se lo pidamos?** La respuesta, verificada en disco:
**no.** La eficiencia de conversión que emerge se queda en ~27%, estable, lejos del
31,5%. Y —esto es lo importante— **no tocamos ningún número para acercarla.** Si hubiéramos
ajustado para que diera 5%, sería el error del "20.0" otra vez. Que dé otra cosa,
honestamente, es lo que hace creíble todo el resto.

**Traducción de fondo:** el campo genera energía útil que persiste y se estructura — pero
no reproduce el reparto de materia de *este* universo. Es coherente con lo que ya sabíamos
del muro: el modelo da *relación y persistencia*, no los números físicos concretos.

## UN HALLAZGO SOBRE CÓMO MEDIR (reportado por CC — pendiente de verificación limpia)

El arreglo 3 apuntó a algo que, SI se confirma, vale para toda la investigación futura:
que la "regla común" de medir energía útil sería mal instrumento cuando el control es
"barajar el campo", porque mide una suma que no mira el orden espacial — serviría para
comparar magnitudes entre experimentos, pero no para contestar "¿esto es distinto del
azar?". **Aviso honesto:** la forma fuerte de esta afirmación ("z=0.0 exacto, bit a bit")
la reporta CC; yo abrí el JSON de E5.5-3 y NO la vi en esa forma (el boost de la regla
común varía, no es cero constante). La idea es razonable pero el número exacto no está
verificado. **La lección general sí vale —la regla de medir y el control tienen que
hablar el mismo idioma, o el experimento queda ciego sin que se note— pero el "z=0
exacto" queda pendiente de comprobar antes de asentarlo como hecho.**

---

## POR QUÉ NO SELLO EL ARCO TODAVÍA (honestidad de método)

11 PASS, 4 FAIL, y **15 con señal parcial, contaminación de ruido pendiente, o resultados
raros sin revisar.** Sellar el arco entero con casi la mitad en zona gris sería
exactamente el error de sobre-afirmar que esta sesión me enseñó a no cometer. En
particular quedan dos cosas que pido mirar antes de cerrar:

- **E5.6-5** dio un resultado invertido (el control correlaciona MÁS que lo real) — raro,
  merece una segunda mirada antes de darlo por bueno.
- **E5.1-1 y E5.2-3** se corrieron con el ruido viejo (el que se desmadra); sus números
  finos no son de fiar hasta re-correrlos con el arreglo 2.

## LO QUE SÍ PROPONGO ASENTAR (con tu permiso)

Aun sin sellar el arco, esto quedó firme y se puede registrar:
1. **La tríada energía-exergía-entropía es medible y coherente** (contabilidad cuadra).
2. **Muerte térmica ≠ Nada, comprobado** (E=máx, X=0, empírico).
3. **El modelo NO reproduce el 5%/31,5%** — negativo limpio, sin ajuste. El campo da
   relación y persistencia, no los números físicos del universo.
4. **La expansión (aislamiento) es la única vía de rescate** de energía útil — ni
   reordenar (E5.5-3) ni re-inyectar la salvan; solo aislarla antes de que se mezcle.

---

## DECISIONES QUE SON TUYAS

1. **¿Re-corro E5.1-1 y E5.2-3** con el ruido arreglado (los dos que quedaron
   contaminados), antes de cualquier cierre?
2. **¿Reviso E5.6-5 y E5.5-5** (el invertido y el del matiz) antes de asentarlos?
3. **¿Asiento los 4 puntos firmes de arriba** en el registro, dejando el resto como
   "señal parcial, pendiente", o esperás a limpiar la zona gris primero?

No sello nada ni instruyo nada hasta que me digas. La regla de no cerrar sin tu
autorización sigue intacta.
