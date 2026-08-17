# DISEÑO CS073 — Emergencia de estructura: del gas atómico al universo en expansión

**De:** Claude Science (CS) — diseño para coordinación CS + CC + Alexis.
**Fecha:** 19-jul-2026
**Regla:** desarrollado y pre-probado por CS. Toca la física del motor → NO se escribe código
hasta acuerdo de los tres. No cierra ningún experimento (nota permanente vigente).

---

## Premisa (lo ya establecido, no se re-discute)
- El primer átomo **fosiliza** el potencial geométrico: métrica (radio de Bohr), dimensión, π,
  distancia, dirección — como *potencial*, no como campo medible. Probado (CS072).
- El gas atómico primordial es un **campo cuántico**: posición/dirección individuales son
  **indefinidas** (Heisenberg). No hay canicas. Correcto que el motor dé un hub térmico.
- Dimensión/distancia/dirección se vuelven **campo vectorial medible** SÓLO cuando la gravedad
  condensa el gas en **entidades espaciales distinguibles** (estrellas/galaxias). Ése es este
  experimento — la física de este universo empezando a operar.

## Lo que dos prototipos de CS ya DESCARTARON (para no repetirlo)
1. **Gravedad global** (liga todos los pares sobredensos, regla actual `p02_gravedad`): al aplicar
   expansión, el hub se encoge a un **grumo único** que suelta polvo — nunca fragmenta. (Tabla en
   `CS073_prototipo_estructura_hallazgo_CS.md`.)
2. **Gravedad "local térmica"** (k vecinos con |ΔT| mínimo): TAMPOCO fragmenta, y da **REAL = NULL
   exactamente**. Razón: la temperatura es un **escalar 1D**; "vecinos en temperatura" forman una
   cadena, no una vecindad 3D. **No se puede fabricar localidad espacial desde un campo escalar.**

**Conclusión de diseño (inequívoca):** ningún truco a nivel de grafo recupera estructura. La
estructura EXIGE desplegar la métrica fosilizada de Bohr como **POSICIONES reales**, y gravedad
local **en ese espacio**. Eso es el cambio de motor a coordinar.

---

## MECANISMO VALIDADO POR CS (prototipo 19-jul, corregido) — reemplaza el enfoque "desplegar posiciones"

El acuerdo con Alexis afinó el mecanismo: **#23 siembra las sobredensidades, el criterio de Jeans
decide cuáles colapsan, la expansión compite.** Los ejes/posiciones NO se despliegan desde la malla
causal (eso arrastraba la salvedad A.4) — **NACEN con el colapso** (decoherencia por interacción
gravitacional masiva). Prototipo validado:

- Campo #23 = gaussiano MULTIESCALA P(k)~k^n en malla 3D (post-átomo: el espacio ya está fosilizado
  por Bohr, un fondo espacial ES legítimo aquí). Amplitud inicial δ_rms=0.01 — elegida sólo por estar
  muy por debajo del umbral de colapso (NO es el valor del CMB, que es δ~10⁻⁵; el valor exacto no
  cambia la conclusión cualitativa, sólo el D al que nace la 1ª estructura).
- **Batalla de Jeans:** gravedad (∝masa·densidad) vs presión térmica (∝T). Masa de Jeans
  M_J∝T^(3/2)/√ρ. Colapso donde masa_local > M_J_local. La expansión ENFRÍA → M_J cae → fragmentación.

**Resultado (criterio de Jeans completo; control = #23 REAL vs BARAJADO, mismo enfriamiento):**

| etapa (T, D) | M_J medio | REAL: #estr / tam medio | NULL: #estr / tam medio |
|---|---|---|---|
| gas caliente (3.0, 10) | 5.2 | 0 / — | 0 / — |
| enfriando (1.2, 60) | 2.5 | 968 / **80** | 3396 / 21 |
| frío+grav (0.5, 120) | 1.9 | 30 / **5857** | 3 / 73358* |
| muy frío (0.25, 200) | 1.0 | 47 / **3609** | 4 / 47593* |

Discriminante = **coherencia, no conteo**, y la firma CAMBIA de signo con el enfriamiento:
- **Fase enfriando (T=1.2):** REAL da menos estructuras pero más masivas (968, tam 80) vs NULL
  muchas difusas (3396, tam 21) — aquí REAL concentra.
- **Fases frías (T≤0.5):** se INVIERTE — REAL FRAGMENTA en muchas estructuras medianas separadas
  (30-47, tam ~3600-5900) mientras el NULL colapsa en POCOS grumos gigantes indiferenciados
  (3-4, tam ~47000-73000). Ésta es la firma clave: las semillas #23 coherentes producen una
  **jerarquía fragmentada** (protogalaxia → múltiples estrellas Pop III), el barajado sólo un
  monoblob sin estructura interna.
M_J cae 5.2→1.0 con el enfriamiento = la fragmentación jerárquica de Gemini/Jeans. Lo que discrimina
no es "REAL siempre concentra" (falso), sino que **REAL mantiene estructura MÚLTIPLE Y SEPARADA en
todo el rango, mientras el NULL o dispersa (caliente) o funde en un blob (frío)** — nunca fragmenta
coherentemente.

**Shannon cazado:** el control ON/OFF de gravedad NO sirve — el enfriamiento solo baja M_J y dispara
colapso sin gravedad (OFF dio 1872). Control correcto = #23 barajado con enfriamiento idéntico.

## El experimento (CS073)

### Guardián de fondo — G-DIFERENCIA-INTERNA (principio, aportado por Alexis)
Toda diferencia/asimetría que el motor produce debe ser **el campo diferenciándose de sí mismo**,
NUNCA un patrón inyectado desde fuera del campo. Es el fundamento de todos los NULL del arco:
- El campo #23 **barajado** cumple: sigue siendo el mismo campo, solo desordenado → control legítimo.
- Un umbral "a ojo", una semilla de colapso sembrada, una temperatura ABSOLUTA asignada a la
  Singularidad → diferencias desde FUERA del campo = Shannon.
Fundamenta y reemplaza al "G-SOLO-RAZONES-INTERNAS": la razón por la que sólo valen razones internas
es que medir ES poner dos términos del MISMO campo en relación (S = I ⟷ E). Ver nota conceptual
`NOTA_medicion_campo_S_I_E_CS.md` (candidata a Canónica).

### Premisa energética (guardián conceptual, aportada por Alexis)
En todo el mecanismo NO entra energía nueva. M_J∝T^(3/2)/√ρ depende sólo de T y ρ, ambas herencia
directa de la Singularidad: T = la energía primordial enfriándose por la expansión; ρ = esa misma
masa-energía redistribuyéndose. El colapso no añade nada — es la energía original REORGANIZÁNDOSE al
bajar T. La Singularidad tuvo que ser de energía enorme porque es la ÚNICA energía que existe; todo
lo posterior (átomos → estructura → estrellas → complejidad) es esa energía cambiando de forma.
**Guardián G-SIN-ENERGIA-NUEVA:** si el motor necesitara inyectar energía para colapsar, es Shannon;
la estructura sale del presupuesto heredado (T y ρ), no de un aporte externo. Corolario: la
temperatura decreciente ES el reloj (coherente con "el cronograma mide caída de T, no tiempo").

### Paso A — Campo de densidad #23 sobre malla 3D (post-átomo)
Post-recombinación el espacio ya está fosilizado (Bohr) → un fondo espacial 3D es legítimo (el
purismo "sin coordenadas" era para el régimen pre-atómico, ya cruzado). El campo #23 se despliega
como **contraste de densidad δ(x) gaussiano multiescala** P(k)~k^n sobre esa malla, amplitud
primordial realista **δ_rms~0.01** (como CMB) — todo muy por debajo de δ_c. **Sin picos sembrados:**
las sobredensidades salen del espectro. Esto reemplaza el enfoque "desplegar posiciones desde la
malla causal" (que arrastraba la salvedad del marco A.4): los ejes NO se despliegan, **nacen con el
colapso**.

### Paso B — Inestabilidad de Jeans: gravedad vs PRESIÓN TÉRMICA
**[REVISADO — la presión térmica es el ingrediente que faltaba; el motor ya tiene el campo T]**
- **Batalla física:** gravedad (∝ masa·densidad, atractiva) vs **presión térmica** (∝ temperatura,
  repulsiva). NO gravedad vs expansión sola — ése era mi prototipo incompleto.
- **Masa de Jeans** M_J ∝ T^(3/2)/√ρ: una región colapsa si su masa local supera M_J local.
- **La expansión ENFRÍA** (T baja con t — el motor ya lo hace) → M_J CAE (prototipo: 5.2→1.0) →
  la supernube colapsada **se fragmenta** en trozos pequeños = jerarquía en dos etapas
  (protogalaxia monstruosa primero, estrellas Pop III al fragmentar). Esto es lo que Gemini precisó
  y coincide con la formación real.
- **Estructuras** = regiones conexas donde masa_local > M_J_local (descartando ruido de 1 celda).

### Paso B-bis — GUARDIÁN corregido (Shannon cazado en el prototipo)
El control ON/OFF de gravedad **NO sirve aquí**: como M_J∝T^(3/2), el enfriamiento SOLO baja M_J y
dispara colapso aunque la gravedad no amplifique nada (prototipo: OFF dio 1872 estructuras = fuga de
Shannon). **Control correcto = NULL de campo #23 BARAJADO con enfriamiento IDÉNTICO en ambos brazos.**
Aísla lo que aportan las semillas COHERENTES vs el enfriamiento.

### Paso C — Observables (lo que DISCRIMINA es la COHERENCIA, no el conteo)
El prototipo mostró que el conteo bruto NO discrimina (el enfriamiento genera nubes en ambos brazos).
Lo que separa REAL de NULL es la **coherencia**:
1. **Fragmentación jerárquica (no "concentración"):** el #23 real mantiene estructura MÚLTIPLE Y
   SEPARADA en todo el rango de T; el barajado o dispersa (caliente) o funde en 1-4 blobs gigantes
   (frío). La firma es que REAL fragmenta coherentemente donde el NULL no — no que REAL "concentre
   más". Cuantificar como nº de estructuras separadas vs tamaño, no sólo tamaño medio.
2. **Jerarquía:** ¿la supernube se fragmenta al caer M_J (dos etapas), como en formación real?
3. ¿Hay **distancia y dirección medibles ENTRE** las estructuras masivas (centros de masa con posición
   real — vector galaxia→galaxia, definido porque son objetos, no átomos de gas)?

### Control anti-Shannon = #23 REAL vs BARAJADO, enfriamiento idéntico
(reemplaza el ON/OFF de gravedad, contaminado por el enfriamiento — ver Paso B-bis). La señal es que
**REAL fragmenta en estructura múltiple separada, NULL no**: en frío REAL mantiene 30-47 estructuras
medianas mientras el NULL funde en 3-4 blobs gigantes; en enfriando REAL concentra en menos-masivas
vs muchas-difusas del NULL. Cuantificar formalmente en la corrida definitiva con un discriminante que
capture "múltiple y separado" (p. ej. nº de estructuras > umbral de masa, o entropía de la
distribución de tamaños), NO sólo tam medio (que cambia de signo con T).

### Guardianes anti-Shannon
- **G-SIN-SIEMBRA:** cero picos/centros impuestos; las sobredensidades salen del espectro #23.
- **G-UMBRAL-FISICO:** el colapso lo decide δ_c=1.686 (colapso esférico), no un umbral a ojo.
- **G-AMPLITUD-PRIMORDIAL:** δ_rms inicial diminuto (~0.01); si arranca cerca de δ_c estaríamos
  sembrando estructura (fue el bug del 1er prototipo, corregido).
- **G-CAUSAL-ON-OFF:** apagar gravedad → 0 estructuras, o el resultado no cuenta.
- **G-EXPANSION-ISOTROPA:** a(t) uniforme (Hubble); cualquier eje preferido debe salir de la
  distribución de masa, no de la expansión.

---

## Qué toca del motor (a decidir entre los tres)
- **Nueva pieza / época "estructura"**: tras la recombinación, desplegar δ(x) #23 en malla 3D y
  evolucionar Jeans. NO modifica la física ya validada (S>0→átomo); la extiende hacia adelante.
- **`p02_gravedad`**: hoy teje `Bgrav` por umbral térmico (correcto para el gas, da hub). La gravedad
  de ESTRUCTURA es un régimen nuevo (crecimiento δ∝D + colapso Jeans), no el mismo mecanismo. Decidir
  si es pieza aparte (`p02b_estructura`) o extensión de la existente.
- **Escala:** el prototipo corrió en malla 64³. Para distribución de masa realista y estadística de
  la 1ª estrella (Pop III) puede requerir malla mayor. Coste a evaluar.
- **Puente con lo ya hecho:** el campo #23 de `catalogo.py` (densidad multiescala) es la MISMA semilla;
  hay que conectar esa densidad primordial con δ(x) de la malla, no crear un campo nuevo desconectado.

## Estado
Diseño pre-probado (dos negativos que delimitan el mecanismo). Requiere cambio de motor coordinado.
Motor CONGELADO hasta acuerdo CS+CC+Alexis. No cierra experimento.
