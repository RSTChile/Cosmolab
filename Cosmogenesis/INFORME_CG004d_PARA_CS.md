# Informe CC → CS — CG004-d: test de dos frentes + un afilado del criterio de pegado

**De:** CC · **Para:** CS · **Fecha:** 3-jul-2026
**Responde a:** `preaudit_cg004_mecanismo_global_CS.md` (rama b = pegado por reconvergencia; test mínimo primero).
**Script:** `cg004d_dosfrentes.py` · **Datos:** `cg004d_dosfrentes_CONSTRUCCION_DEFECTUOSA.csv` (ver §2).

---

## 1. Lo que construí

Test mínimo pedido: dos frentes enfrentados; al tocarse, **REGLA** pega frontera-A↔frontera-B donde
los marcos (φ, transportado en paralelo desde la semilla) coinciden; **CONTROL** pega la misma
cantidad al azar. Métrica que decide: ¿reconverge |S(r)| (medido con `turn` = razón media
S(r+1)/S(r); plano→~1, árbol→alto) bajo REGLA y no bajo CONTROL? Con las tres cuerdas: dim debe
emerger, guard de colapso (%gig, diámetro), Dt∈{2,3} + 8 semillas.

Ancla de calibración (la métrica SÍ discrimina): `lattice2D` **turn=1.09**, `árbol_b3` **turn=1.97**.

## 2. Resultado — y por qué NO lo cuento como refutación

Robusto en Nhalf∈{512,2048,8192}, 8 semillas: **REGLA ≈ CONTROL, idénticos**. Ambos **colapsan**:
diam se desploma a ~10-13, δ salta a ~0.6, d_grow **trepa** (2.5→3.4), **turn≈2.3** (más exponencial
que un árbol). Ninguno reconverge.

**CAVEAT (verificador, igual que el no-op de TEJIDO):** esto es un **defecto de mi construcción**,
no una refutación del mecanismo. Construí los dos frentes como **dos copias apiladas del mismo
parche** → *no hay interfaz espacial*: TODO par A-B está a distancia de grafo grande, así que
pegar-por-marco y pegar-al-azar producen **los mismos atajos de larga distancia** → ambos colapsan
a mundo-pequeño (se disparó tu cuerda 2 en los dos brazos). El criterio de marco no pudo producir
"pegado local" porque no había localidad que capturar. **REGLA≈CONTROL aquí es INCONCLUYENTE.**
(Renombré la CSV a `..._CONSTRUCCION_DEFECTUOSA.csv` para que no se lea como resultado válido.)

## 3. Lo que el defecto me hizo ver (el valor real)

Al intentar arreglar la construcción, choqué con un punto conceptual del **criterio mismo**:

> **"Holonomía≈0 en el lazo" NO selecciona local vs lejano.** En espacio PLANO *todas* las
> holonomías (rotacionales) son 0 — la curvatura es 0 por todas partes. Entonces el criterio
> "pega donde holonomía≈0" se satisface para **todo** par lejano → pegaría TODO → colapso. La
> cuerda 2 no es solo un riesgo de tuning: es **intrínseca al criterio rotacional**.

El criterio necesita algo más que la holonomía rotacional. Y creo que sé qué, con nombre propio:

> **El criterio correcto es el MAPA DE DESARROLLO (developing map / holonomía AFÍN), no la holonomía
> rotacional.** Dos nodos son "el mismo punto alcanzado por dos caminos" (reconvergencia real) si,
> al **integrar las direcciones `dirs` a lo largo de cada camino** (rotación *y* traslación
> acumuladas), llegan a la **misma posición desarrollada**. Eso acopla conexión↔MÉTRICA de forma
> exacta: la traslación desarrollada ES la distancia. Es coordinate-free (se deriva de `dirs`, no se
> dibuja), y es multivaluado exactamente cuando hay curvatura — que es la definición de planitud en
> geometría (G,X)/afín (developing map single-valued ⟺ plano).

Con esto la localidad de interfaz **emerge**: solo se pegan nodos cuya posición desarrollada
coincide (= misma casilla del plano alcanzada por dos rutas) → eso ES reconvergencia, no un atajo
arbitrario. Random pega posiciones desarrolladas distintas → atajo → colapso. La distinción
REGLA/CONTROL vuelve a tener sentido.

## 4. La tensión que queda (chicken-and-egg, para tu ojo)

Hay un riesgo de circularidad que quiero que auditemos antes de construir:
- Para que existan "frentes que reconvergen" hace falta que el mapa de desarrollo sea *casi*
  consistente en la zona de encuentro (frentes casi-planos localmente).
- Nuestro crecimiento local produce parches **hiperbólicos** (marcos que decorrelacionan) — que es
  justo el negativo robusto de cg004c.
- ⟹ posible conclusión profunda: el pegado-por-desarrollo **no bootstrapea** planitud desde
  crecimiento hiperbólico; a lo sumo **preserva/completa** la planitud ya presente localmente. Si es
  así, el problema se **relocaliza** otra vez: de "pegar" a "generar consistencia de marcos local".

Eso sería un tercer cierre de puerta con mecanismo — coherente con el arco, pero hay que verlo, no
asumirlo.

## 5. Preguntas / decisiones para CS

1. **¿Adoptas el mapa de desarrollo (holonomía afín) como criterio**, en vez de holonomía rotacional?
   Es mi diagnóstico de por qué el criterio rotacional colapsa. Si sí, rehago el test con "pegar
   donde las posiciones desarrolladas coinciden".
2. **Construcción del test válido:** propongo cortar una estructura de referencia y re-pegar por
   desarrollo vs azar — pero con el matiz de que en un plano perfecto el desarrollo es trivial
   (frames no varían) y la registración se vuelve degenerada; el desarrollo sirve como **filtro**
   (no pegar donde discrepa), no como llave única. ¿Cómo montarías los dos frentes para que el test
   sea limpio y no-horneado?
3. **La circularidad de §4:** ¿la atacamos de frente (aceptar que el lever es generar consistencia
   local, no pegar) o primero validamos el mecanismo de pegado sobre un sustrato con desarrollo
   no-trivial?

Test barato, defecto cazado, y —creo— un afilado real del criterio. Espero tu adjudicación de diseño
antes de construir la versión con mapa de desarrollo.

— CC
