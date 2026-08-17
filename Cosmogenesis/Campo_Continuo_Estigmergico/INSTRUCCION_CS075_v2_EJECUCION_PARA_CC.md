# INSTRUCCIÓN CS075 v2 — EJECUCIÓN. Un error mío costaba 141 horas; corregirlo es el paso 1

**Encarga:** Alexis López Tapia (director) · **Diseña:** Claude Science · **Ejecuta:** CC
**Fecha:** 30-jul-2026 · **Reemplaza:** `INSTRUCCION_CS075_PARA_CC.md` (v1, con dos errores míos)

---

## 0. Lo que ya está adjudicado y NO hay que rehacer

CC implementó el motor, corrió las pruebas, encontró y arregló un bug de divergencia que yo no
había visto. Todo eso está verificado en disco y adjudicado en
`ADJUDICACION_cs075_23_sobre_fisica_CS.md`:

- **Inventario cerrado en 23**, con M3 como el elemento que faltaba (no #24, que era mi error).
- **El bug tipo Riccati de `22_qcd`** (`(ρ/media−1)²` sin acotar) corregido, más la misma clase
  de defecto en `2_gravedad`, `17_oscuro` y `23_campo`, todos con el patrón `tanh` que ya
  existía en el archivo.
- **E1, E2, E3, E4, E6 PASAN.** E2 (nadie madruga) y E3 (cero exacto dormido) eran las dos no
  negociables: la tesis del director se sostiene con cero violaciones.
- **Tasa medida:** 6,05 ms/paso. Verifiqué su aritmética de forma independiente y cierra en tres
  puntos (21.064.463 pasos; 141,6 h en serie contra 141,5 reportadas; 0,24 % del camino).

**No toques nada de eso.** Los tres archivos de `Campo_Continuo_Estigmergico/` son la base.

---

## 1. EL ERROR MÍO QUE HAY QUE CORREGIR PRIMERO (y es la causa de las 141 horas)

Mi instrucción v1, §3.2, decía: *"La razón física 159 GeV / 155 MeV ≈ 1026 fija la separación
entre `T_bajo_electrodebil` y `T_bajo_confinamiento`; usá esa razón, no dos números elegidos por
separado."*

CC hizo exactamente eso, y bien: ancló `T_EW = T_inicial` y derivó `T_CONF = T_inicial/1026`.
De ahí salen los ~21 millones de pasos y las ~35,4 h por configuración.

**El problema: el proyecto YA tiene esos dos umbrales fijados, y no son esa razón.**
Verificado en disco, `cs072_motor_23.py`:

```
l.42:  T_CONF=0.6            # umbral de enfriamiento: confinamiento actúa con universo frío
l.43:  T_EW=0.9              # umbral electrodébil: la débil actúa con universo aún caliente
```

Y se usan tal cual en las puertas del motor: `l.130: if '3_fuerte' not in apagar and T_ef <
T_CONF` y `l.147: if '5_debil' not in apagar and T_ef > T_EW`. Son temperaturas **normalizadas**
del modelo, con razón `0.9/0.6 = 1.5`, no 1026.

**Lo que hice mal:** le pedí a CC derivar de la razón física en GeV/MeV cuando el proyecto ya
tenía la traducción hecha a sus propias unidades. Inventé una escala donde había una. Es el mismo
error que cometí con `#18` (lo clasifiqué como "espacio" teniendo su definición a la vista) y con
`#24` (lo puse en el inventario donde aparece cero veces): **usar mi criterio donde el proyecto
ya tenía el suyo escrito.** Tres veces el mismo error.

### 1.1 Qué hacer

**Reemplazar los dos umbrales por los del motor**, citando archivo y línea en el código:
`T_CONF = 0.6` y `T_EW = 0.9`, aplicados sobre `T` normalizada por `T_inicial` (es decir, la
puerta compara `T/T_inicial` contra esos valores, porque en el motor `T_ef` es relativa a `T0`).

**Después, MEDIR el costo de nuevo y reportarlo.** No lo estimo yo: bajar el umbral de
`T_inicial/1026` a `0.6·T_inicial` reduce muchísimo el camino de enfriamiento, y con la ley
`T = T₀/√(1+k·t)` el número de pasos escala con el **cuadrado** de la razón de temperaturas. El
número puede caer varios órdenes de magnitud, pero **no lo escribo acá porque no lo medí** — es
lo primero que hay que imprimir.

**Si al reemplazarlos algo no cuadra** — por ejemplo si con `T_CONF=0.6` el confinamiento se
cruza tan temprano que los niveles altos no alcanzan a diferenciarse — **eso es un desacuerdo y
un desacuerdo es un dato: pará y reportá.** No ajustes el umbral para que el experimento "se vea
bien".

---

## 2. PASO 2 (bloqueante): cerrar E5 antes de cualquier barrido

E5 es la única prueba que falla: la exergía **sube** (0,0099 → 0,0567) en vez de bajar. CC no la
forzó a bajar tocando fórmulas, que era lo correcto, y propuso una lectura física: `2_gravedad` y
`3_fuerte` concentran densidad y generan diferencia local, compitiendo contra la difusión.

**Verifiqué el control que invoca y existe:** en `cs075_resultado_base_fisica.json` la base sin
agentes va de X = 0,005528 a X = 0,000208 — baja. Con los 23 agentes, sube.

Pero eso no lo prueba: la diferencia entre esas dos corridas son **los 23 agentes**, no esos dos.
La prueba directa cuesta minutos:

**E5b — apagar `2_gravedad` y `3_fuerte`, dejar los otros 21, y medir X.**

- Si **X vuelve a bajar** → la explicación de CC queda medida, no supuesta. Se registra como
  hallazgo físico: en este modelo los agentes de estructura local generan exergía.
- Si **X sigue subiendo** → la explicación es falsa y hay otro mecanismo (o un bug más). **Pará
  y reportá.** No corras el barrido con una dirección termodinámica sin explicar.

Corré también la variante con esos dos encendidos y los demás apagados, si es barato: si X sube
sólo con ellos dos, el aislamiento es completo.

**Esto es el calibrador de este experimento**, y va primero por la misma razón que en cs074D: se
sabe en minutos, no después de gastar el cómputo grande.

---

## 3. PASO 3: el smoke, con el rango corregido

Sólo si el paso 1 dio un costo viable y el paso 2 cerró E5.

**Cuatro configuraciones**, `amp_asimetria` = 0,01 / 0,1 / 0,5 / 2,0, malla 16³, `dt=1e-3`,
`k_enfriamiento=50`, corriendo hasta cruzar `T_CONF=0.6` con margen.

**Ojo con el rango, por el hallazgo lateral de CC:** con el bug corregido, las configuraciones de
baja asimetría (0,01 y 0,1) **sí alcanzan sobredensidad**, cuando antes nunca llegaban en la
ventana de prueba. **El bug estaba ocultando estructura real.** Si al correr con el umbral
corregido las cuatro configuraciones dan estructura, el rango interesante puede estar más abajo
de 0,01 — reportalo, no lo barras por tu cuenta.

### Qué entregar

- `cs075_resultado_23_sobre_fisica_v2.json` con el registro completo por paso
- **la tabla del orden de despertar** por configuración: qué agente despertó en qué paso
- **cuántos de los 23 quedaron dormidos, y por cuál hito faltante** — un agente dormido con su
  hito identificado **es un resultado**, no un fallo. Si `#24`… (perdón: si `M3`, `15_causal`,
  `18_poda`) quedan dormidos porque no hubo red persistente, eso es la arquitectura funcionando.
- el costo medido: ms/paso, pasos hasta `T_CONF`, horas por configuración

### Y ahí PARÁS

No corras barrido grande. No cierres el experimento. Requiere autorización explícita del director.

---

## 4. LO QUE NO HAY QUE HACER

- **No hay NULL en esta etapa.** El director lo excluyó: primero se prueba la arquitectura.
- **No inventes constantes ni escalas.** `p_expansion.py` lo dice textual: *"NO se inventa una ley
  nueva -- se deriva del propio reloj de enfriamiento que el motor YA tiene... ninguna constante
  nueva"*. Yo violé esa regla cuatro veces en este diseño (`H_post`, `fin_inflacion`, la razón
  1026, y `#24` en el inventario). **Antes de fijar cualquier valor, buscá si el proyecto ya lo
  tiene.** `grep` en `cs072_modulos/` y en `cs072_motor_23.py` es más rápido que derivarlo.
- **No fuerces E5 a pasar** tocando fórmulas. Ya lo hiciste bien una vez.
- **No enciendas agentes a mano** ni en un paso fijo. La puerta lee el estado.

---

## 5. Estado de confianza de este documento

Lo que está **verificado en disco por mí**: los umbrales `T_CONF=0.6` / `T_EW=0.9` de
`cs072_motor_23.py` l.42-43 y su uso en l.130 y l.147; el control de exergía de la base sola
(0,005528 → 0,000208); la aritmética de la tasa de CC.

Lo que **no** está medido y por eso no lleva número: el costo con los umbrales corregidos. Es la
primera cosa a imprimir.

Lo que ya falló tres veces: **mi criterio cuando el proyecto tenía el suyo escrito.** Si algo de
esta instrucción choca con un archivo del proyecto, **el archivo gana y vos reportás el choque.**

---

*Verificá en disco, no de palabra: antes de escribir "verifiqué X", el valor de X tiene que estar
impreso en la salida que estás mirando. Nada se cierra sin autorización del director.*
