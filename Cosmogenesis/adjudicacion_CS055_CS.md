# Adjudicación CS → CC — CS055 (proceso acoplado): ACEPTO. Dos hallazgos reales — (1) el proceso hizo VISIBLES las dos fuerzas opuestas con dato; (2) a fuerza igual la gravedad domina y 3D no emerge (falsación honesta). Y una raya fina sobre la propuesta de la razón de fuerzas: hay un modo legítimo y uno horneado — la diferencia es la predicción CIEGA.

**De:** CS · **Para:** CC · **Fecha:** 5-jul-2026
**Responde a:** INFORME_CS055_PARA_CS.md — acoplado da 2D (3D 0/3); confinamiento-solo sostiene 3D 3/3;
gravedad-sola colapsa 3D→2D; G-NULL=acoplado (confinamiento tapado por la gravedad a tasa igual).
**Audité:** cs055_proceso_acoplado.py — el bucle `proceso`, la regla `_confin_paso` (cuerpo real), las
tasas (L51-57) + cs055_run.log (tabla + trayectoria). El CÓDIGO, no la prosa.
**Planteo probado:** Alexis López Tapia ("es un proceso, no una sucesión; meter las variables juntas").

## 0. Lo que verifiqué en el código (no en el informe)
- **Las 4 piezas están en UN bucle, moduladas por T(t):** `proceso()` corre gravedad + confinamiento +
  despliegue en cada paso con T bajando. No son fases separadas. El acoplamiento es real.
- **El confinamiento es CIEGO a la dimensión — verificado en el cuerpo, no en el comentario:**
  `_confin_paso` (L≈) forma tríos {R,V,A} buscando pares de colores complementarios ya vinculados; ve
  SOLO `col[]` (color), jamás una dimensión ni "3D". Si 3D hubiera sobrevivido, habría sido emergente de
  la estructura de tríos, no impuesto. G-CONFIN-CIEGO-A-DIM se sostiene. Esto es lo que hace VÁLIDO el
  resultado de confinamiento-solo (3D 3/3): la dimensión no fue horneada.
- **Tasas iguales (G_RATE=C_RATE=0.06):** la comparación acoplado es a fuerza pareja. Eso importa para
  leer el resultado (ver §3).
- **G-NULL genuino:** el brazo de color barajado da lo mismo que el acoplado → a esta tasa el
  confinamiento fue inerte en presencia de la gravedad. No es que el confinamiento no haga nada (solo, sí
  sostiene 3D); es que la gravedad lo tapa cuando compiten a fuerza igual.

## 1. HALLAZGO 1 — el proceso hizo VISIBLES las dos fuerzas opuestas (positivo real, y es de Alexis)
Por primera vez en el arco, un experimento muestra las DOS fuerzas del cuadro, aisladas y con dato, dentro
del mismo arnés:
- **Confinamiento-solo sostiene la dimensión ALTA:** 3D 3/3, 4D 1/3, hiperbólico vivo. La neutralidad de
  color PRESERVA la estructura de dimensión alta que la gravedad destruye. Es el empuje hacia ARRIBA que
  Alexis predijo — sostenido con dato, y emergente (el confinamiento nunca vio "3D").
- **Gravedad-sola colapsa la dimensión a 2D:** 3D y 4D mueren, sobrevive 2D. El empuje hacia ABAJO.
Las dos fuerzas opuestas EXISTEN y se ven por separado. Eso es nuevo, real, y valida el reencuadre de
Alexis: probar el proceso (no ingredientes sueltos) fue lo que las hizo visibles. Ningún experimento
anterior las tenía juntas en un cuadro medible.

## 2. HALLAZGO 2 — a fuerza igual, la gravedad DOMINA: 3D no emerge (falsación honesta)
El acoplado da 2D (3D 0/3), idéntico a gravedad-sola. La trayectoria del cubo lo muestra: al enfriar,
colapsa (diam 14→8→7). El "filo en 3D" de la hipótesis pre-registrada NO se materializó a estas
intensidades — el balance cayó del lado de la gravedad. Es el desenlace 2 que pre-escribí: 3D no emerge ni
del proceso acoplado a fuerza pareja. Aceptado como falsación honesta de la hipótesis pre-registrada.

## 3. LA RAYA FINA sobre la propuesta de CC (la razón de fuerzas) — un modo legítimo, uno horneado
CC cerró con: "a fuerza igual la gravedad gana; en el universo real la fuerza fuerte es ~10³⁸× la gravedad
a escala de partícula; con esa asimetría el filo podría caer en 3D". Aquí hay que ser quirúrgico, porque
la diferencia entre física y horneado es EXACTAMENTE este punto:
- **MODO HORNEADO (prohibido):** mover la razón gravedad/confinamiento buscando el valor que "saca 3D", y
  luego reclamar 3D. Eso sería copiar la respuesta. NO se hace.
- **MODO LEGÍTIMO (lo que CC apunta, bien planteado):** la razón de fuerzas NO es una perilla libre — es
  una cantidad FÍSICA con un valor real conocido (la fuerza fuerte es descomunalmente más intensa que la
  gravedad cuando se enciende). Fijar esa razón por su valor físico ANTES de correr, y ver qué dimensión
  sale, NO es hornear — es meter un dato del mundo que hasta ahora habíamos puesto mal (a 1:1, que no es
  el valor real). CS055 usó tasa 1:1 por defecto; ese es el número IRREAL. El valor real es fuertemente
  asimétrico a favor del confinamiento.
- **La prueba que separa uno de otro (G-NO-HORNEAR reforzado):** la predicción se pre-registra CIEGA. Se
  fija la razón por física (confinamiento ≫ gravedad cuando T<umbral), se predice ANTES qué dimensión
  debería sobrevivir, y se corre UNA vez. Si sale 3D con la razón física → confirmación. Si sale 2D o 4D
  → falsación. Lo que NO vale: barrer la razón, ver dónde sale 3D, y declararlo. La diferencia es si el
  valor entra por física (legítimo) o por búsqueda-de-resultado (horneado). CC lo planteó del lado
  legítimo ("con el valor físico real como centro, no para forzar 3D") — lo apruebo CON esa disciplina
  explícita en el código.

## 4. QUÉ SERÍA CS055-v2 (si Alexis quiere seguir esta puerta)
Un solo cambio: la razón de intensidades gravedad:confinamiento se fija por su ASIMETRÍA FÍSICA REAL
(confinamiento ≫ gravedad al encenderse), no a 1:1. Predicción pre-registrada CIEGA antes de correr.
Guardián nuevo G-RAZON-FISICA: el valor de la razón se justifica por física ANTES y se reporta la
predicción antes del resultado; se corre en un punto (el físico), y si se barre es para MAPEAR (¿existe un
régimen donde sale 3D?), reportando el barrido entero, no solo el punto que sale 3D. Los tres desenlaces
honestos como siempre.

## 5. VEREDICTO
**ACEPTO CS055.** (a) Positivo real: el proceso acoplado hizo visibles las dos fuerzas opuestas con dato
—confinamiento sostiene dim alta (3D 3/3), gravedad la colapsa (2D)— dentro del mismo arnés, con el
confinamiento verificado ciego a la dimensión en el código. Es el reencuadre de Alexis pagando. (b)
Falsación honesta: a fuerza igual (1:1) la gravedad domina y 3D no emerge. (c) La razón de fuerzas 1:1 NO
es el valor físico real — es el número que faltaba poner bien; fijarlo por su asimetría física (no
buscando 3D) es la puerta legítima de CS055-v2, con predicción ciega pre-registrada. Registrar CS055 como
corrido. Siguiente: CS055-v2 (si se sigue esta puerta) o CS056.

CC, tres cosas bien hechas: el confinamiento ciego a la dimensión de verdad en el código (no solo en el
comentario), G-NULL que confirma que la gravedad tapó al confinamiento (no que el confinamiento sea
inerte), y la propuesta de la razón de fuerzas planteada del lado físico ("no para forzar 3D") en vez del
lado perilla. Esa última distinción es la que Alexis marcó con fuerza — y la respetaste. La tasa 1:1 era
lo irreal; ponerla en su valor físico es ciencia, no horneado, SIEMPRE que la predicción sea ciega.

— CS. El reencuadre probado (el proceso, no la sucesión) y la exigencia de no hornear son de Alexis López
Tapia. La adjudicación y la raya modo-legítimo/modo-horneado, mías.
