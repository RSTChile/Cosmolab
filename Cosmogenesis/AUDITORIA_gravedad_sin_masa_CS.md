# NOTA DE AUDITORÍA CS — La "gravedad" de CS054-057 se calculó SIN masa: se acopló a la densidad de vínculos (ρ=grado), no a la masa. Observación de lógica de Alexis. Consecuencia: releer esa gravedad como provisional y abrir la gravedad-con-masa (a partir de CS060).

**De:** CS · **Fecha:** 5-jul-2026 · **Disparador:** observación de Alexis — "ya habíamos puesto la
Gravedad, pero ¿cómo calcularon su efecto si no había ninguna partícula con masa? ¿no es una soberana
tontera?" **Verificado en:** los CUATRO archivos del bloque de fuerzas — cs054_v2_gravedad_alcance.py, cs055_proceso_acoplado.py,
cs056_cuatro_fuerzas.py y cs057_paisaje_completo.py (código real). CS054-v2/CS055/CS057 definen `_grav_paso`
con `rho=[len(a) for a in adj]` (grado); CS056 IMPORTA esa misma función de CS055 (`_grav_paso = _C5["_grav_paso"]`,
línea 43). Los cuatro acoplan la gravedad al GRADO, ninguno a masa. (Nota de proceso: la primera versión de
esta auditoría afirmó el rango "CS054-057" tras grepear solo dos de los cuatro archivos; la verificación en
los cuatro se completó después, y confirma el rango.)

---

## 1. EL HECHO (verificado en el código, línea literal)
En CS054-v2 y CS057 la gravedad se computa así:
```
rho = np.array([len(a) for a in adj])   # ρ = número de vínculos del nodo (GRADO)
srcs = rng.choice(N, p=rho/rho.sum())   # fuente de atracción ∝ densidad de vínculos
w    = rho[j] / (d ** ALPHA)            # peso ∝ ρ_j / distancia² (por saltos de grafo)
```
**NO existe ninguna variable de masa en CS054-057.** La "masa" que la gravedad usó es `ρ` = el GRADO del
nodo (cuántos vínculos tiene). Los nodos más conectados atraen más.

## 2. LA OBSERVACIÓN DE ALEXIS ES CORRECTA (y es de fondo, no de detalle)
- En física, la gravedad se acopla a la MASA. Sin masa, no hay a qué acoplarla.
- El modelo sustituyó masa por DENSIDAD DE CONEXIONES (grado). Eso NO es gravedad: es **enlace preferencial**
  ("lo más conectado atrae más"), un mecanismo real pero DISTINTO. Le pusimos el nombre "gravedad" a un
  proceso sin el ingrediente que define a la gravedad.
- Es, precisamente, la clase de asignación-por-analogía que la cuerda anti-Shannon persigue: nombrar algo
  "gravedad" sin la masa dentro. Nadie del equipo (CC, Grok, CS) lo marcó; Alexis lo cazó por lógica pura.
- Agravante honesto: masa y conexión NO son independientes en física (la masa curva, la curvatura liga),
  pero el modelo las COLAPSÓ en una sola variable (ρ) desde el inicio, sin declararlo.

## 3. CONSECUENCIA PARA LA LECTURA DEL ARCO (qué cambia y qué NO)
- **NO invalida los negativos de CS054-057 como negativos.** Siguen siendo ciertos: "el enlace preferencial
  con alcance (mal llamado gravedad) elige 2D/curvo, no 3D-plano". El hallazgo se mantiene con su nombre
  CORRECTO.
- **SÍ cambia la INTERPRETACIÓN:** esa "gravedad" era gravedad PROVISIONAL SIN MASA. Que eligiera 2D/curvo
  puede deberse a que era densidad-de-grafo, no gravedad real. La conclusión "ninguna fuerza local
  selecciona el 3D" (CS057) se mantiene, pero con la nota de que la gravedad probada no era la gravedad
  con masa.

## 4. LO QUE ABRE (y por qué CS060 encaja exacto)
La masa que CS060 introduce (inercia/persistencia del marco, vía los leptones) es LITERALMENTE el ingrediente
que la gravedad necesitaba y nunca tuvo. Secuencia corregida del arco:
- CS054-057: "gravedad" SIN masa (= enlace preferencial). Elige 2D/curvo.
- CS059: marco (espín) SIN masa. Negativo (confound de longitud de ciclo).
- CS060: mete la MASA (inercia/persistencia). → habilita, por primera vez, una GRAVEDAD DE VERDAD:
  una que se acople a la INERCIA del nodo (masa), no a su número de vínculos (grado).
**Tarea nueva que sale de la observación (para CS060 o un CS061):** correr la gravedad acoplada a la masa-
inercia (no a ρ), y comparar con la gravedad-por-densidad de CS054-057. ¿Cambia la geometría seleccionada
cuando la gravedad se acopla a lo que físicamente le corresponde? Guardián: la masa-inercia debe ser una
variable SEPARADA del grado; si se vuelve a colapsar en ρ, es el mismo error.

## 5. GUARDIÁN NUEVO (para todo el arco, de aquí en más)
**G-NO-RENOMBRAR-SIN-INGREDIENTE:** ninguna fuerza se llama por su nombre físico si le falta el ingrediente
que la define. "Gravedad" exige masa; "confinamiento" exige color; etc. Si se usa un proxy (densidad por
masa), se DECLARA explícito en el nombre y en el informe ("gravedad-por-densidad", no "gravedad"). Esto
cierra el hueco que Alexis destapó, para todo el registro.

— Nota de auditoría por CS. La observación —gravedad calculada sin masa— es de Alexis López Tapia, y es
correcta y de fondo. Verifiqué el código, confirmé que ρ=grado hacía de masa, y asenté la consecuencia: los
negativos se mantienen con su nombre correcto, y la gravedad-con-masa queda abierta a partir de CS060. El
error fue del equipo (renombrar sin el ingrediente); la cacería, de Alexis.
