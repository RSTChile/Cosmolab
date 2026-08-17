# CG003 — Emergencia del espacio desde S>0 · Resumen para Claude Science

**Programa:** Cosmogénesis · Cosmosemiótica aplicada · RMD 2.0 · Club Abulafia
**Fecha del resumen:** 1-jul-2026
**Autor:** Claude CC (motor/custodio) · Dirección: Alexis López Tapia (Casaubon)
**Estado:** experimento VIVO, en bifurcación pre-`cg003e`.
**Ruta de trabajo:** `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/` (antes en `/Volumes/LaCie/...`; el disco se movió).

> Bienvenida, Claude Science. Este documento te pone al día para que puedas revisar/atacar el problema con contexto completo. Regla de la casa: **el hecho y su nombre son separables** — los números se reproducen sin compartir la Teoría; la lectura canónica se ofrece aparte y es descartable.

---

## 0. Primero: ANALIZA LOS SCRIPTS (no te quedes solo con este resumen)

Antes de leer el resto, **abre y lee los scripts reales** — ahí está lo que hicimos, línea por línea, con la disciplina completa (docstrings auto-descriptivos, falsadores, cero coordenadas). El resumen es el mapa; los scripts son el territorio. Corre cada uno con `--quick` si quieres reproducir (venv en `Cosmogenesis/venv`, numpy 2.5).

Lee **en este orden** (es el orden en que el problema se fue afilando):

1. **`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cg003_espacio_relacional.py`**
   → El planteo original y la **regla de oro** (cero coordenadas). Mira `init_web`, `paso` (identidad = fila de acoplamiento; antipodal; economía de grado), y `dimension_espectral` / `shuffle_null`. Aquí se define *qué contamos como espacio*.
2. **`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cg003b_exergia_primero.py`**
   → Entra la **2ª ley** (la exergía se esparce = anti-grumo) y el discriminante afilado `dimension_crecimiento` (`N(r)~r^d` geometría vs `e^{br}` azar). Mira por qué el esparcir *debería* crear extensión — y por qué no bastó.
3. **`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cg003c_crecimiento.py`**
   → El modelo **escalable** (listas de adyacencia + BFS, N hasta 30.000). Función `crecer` (pegado local al frente, atajos muertos por construcción) y `dimension_crecimiento`. **Aquí se midió el muro** (d trepa con N).
4. **`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/cg003d_campo_angular.py`**  ← *el más reciente, léelo con lupa*
   → El **campo con direcciones**. Función `crecer_campo`: espacio tangente `R^Dtan` por nodo, `dirs` (vector unitario por enlace = la dirección), **exclusión angular** (`libre`, `cos_min`), y el cross-linking direccional. Es el intento que dio la señal (d rastrea Dtan) pero se quedó en mundo-pequeño. **Tu crítica más útil va sobre este archivo.**

Trasfondo (opcional, para el vocabulario y el arco previo de partículas/constantes):
`/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/Sesión final Cosmogenesis Claude Codex.md` · `/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis/CC_TO_GROK.md` (Msg 46 = la pared de quiralidad R7h, gemela de esta).

Cuando los hayas leído, vuelve a las secciones 2–5 (el porqué) y a la sección 6 (lo que te pido).

---

## 1. La pregunta

¿Puede la **geometría** —distancia, dimensión, extensión— **SALIR** de `S>0` (persistencia de la diferencia) como consecuencia de la estructura relacional, **sin que le asignemos coordenadas**?

**Regla de oro (inviolable):** cero coordenadas. No hay `x,y,z`, ni grilla, ni caja. Si una coordenada aparece en el *setup*, hicimos trampa. El espacio se **mide**, nunca se dibuja.
- distancia = distancia relacional en el grafo (pasos de acoplamiento).
- dimensión = se MIDE (crecimiento `N(r)~r^d` o dimensión espectral), NUNCA se asigna.

**Falsadores fijos (heredados del arco CG002):**
1. **N-independencia** — una dimensión REAL no depende de N (como el grano κ_Δ del arco previo era N-indep). Si `d` sigue a N → artefacto.
2. **Shuffle null** — barajar la trama (misma densidad, aristas al azar) debe **destruir** la geometría. Si no cambia, era artefacto.
3. Multi-semilla — ¿ley (d constante) o historia (d varía por cosmos)?

---

## 2. El muro, medido (cg003a/b/c)

| Script | Idea que añadió | Resultado |
|---|---|---|
| `cg003_espacio_relacional.py` | espacio = trama de acoplamiento `C_ij`; dim **espectral** | se fragmenta; `d_s` no converge |
| `cg003b_exergia_primero.py` | **2ª ley**: la exergía se *esparce* (anti-grumo) → extensión; dim de **crecimiento** `N(r)~r^d` | 1/6 geometría, shuffle no separa → **mundo-pequeño** |
| `cg003c_crecimiento.py` | el cosmos **crece** pegando local al borde; atajos muertos por construcción | **hallazgo firme** ↓ |

**Hallazgo firme (`cg003c`, N hasta 30.000, literal de la corrida):**
```
  m=2: N=1000:d=1.89  N=4000:d=2.24  N=12000:d=2.53  N=30000:d=2.77
  m=3: N=1000:d=1.59  N=4000:d=1.92  N=12000:d=2.19  N=30000:d=2.41
  real: 8/48 GEOMETRIA   shuffle: 45/48 azar/mundo-pequeño
```
La dimensión **trepa con N sin converger**, y `R²_exp > R²_pot` casi siempre → **AZAR exponencial = mundo-pequeño**, no geometría. El único espacio genuino fue el **hilo 1D** (orden puro).

**Lectura (firme):** de pura RELACIÓN sale **tiempo/orden** (barato: cada enlace es un "después"), **no espacio/extensión** (caro: pide **direcciones** — ejes ortogonales inconmensurables que un grafo no tiene). **Es el mismo muro del spin y la quiralidad.** El único portador de ángulo es el **campo continuo**.

---

## 3. El experimento del campo (`cg003d_campo_angular.py`, 1-jul-2026)

Le dimos al sustrato **lo único que al grafo le faltaba, y nada más**: una **orientación local** — cada nodo tiene un espacio tangente de dimensión `Dtan` y sus enlaces salen en **direcciones** (vectores unitarios en `R^Dtan`), con **exclusión angular** (dos direcciones casi paralelas = el mismo eje, prohibido). Seguimos SIN coordenadas globales: sólo direcciones locales relativas (una conexión / transporte paralelo), nunca un marco común.

**Resultado (`--quick`, N hasta 4000):**
```
  Dtan=2:  d ≈ 2.05 → 2.46     (diámetro 23 → 29,  ×1.26)
  Dtan=3:  d ≈ 2.35 → 2.76
```

**Señal POSITIVA (nueva, el grafo puro nunca la dio):** la dimensión emergente **rastrea `Dtan`** — el orden 2 < 3 se preserva. *El espacio hereda sus direcciones del campo.* El ángulo importa.

**El "aún no":** el diámetro creció ×1.26 con N. Si fuera **2D plano genuino** debería ~duplicarse (`~N^{1/2}`); ×1.26 ≈ `log N` → **sigue siendo mundo-pequeño**, y por eso `d(N)` todavía trepa (Δ≈0.41, no converge).

---

## 4. Diagnóstico — el muro tiene un nombre más fino: **PLANITUD**

El ángulo dio **direcciones locales** pero **no consistencia global**. Cada nodo separa bien sus ejes localmente, pero las direcciones **no cierran coherentemente alrededor de los lazos** → el volumen `N(r)` crece exponencial = **curvatura negativa (hiperbólica)**. Un espacio extenso plano necesita `N(r)` **polinomial**, y eso exige **curvatura media ≈ 0 = FLATNESS = holonomía nula**.

> No basta con que haya ángulo; los ángulos tienen que **componerse consistentemente alrededor de un lazo** (transporte paralelo sin holonomía neta). No es *dirección*, es **conexión de curvatura controlada.**

**Conexión con el arco de partículas (importante):** en `CC_TO_GROK.md` **Msg 46** (auditoría R7h), la quiralidad falló su falsador con idéntico veredicto — *"la mano necesita un sustrato espacial/angular que el ensamble de etiquetas no tiene; `apply_triplet` es simétrico, no hay grado angular sobre el que girar."* **Es el MISMO muro que CG003.** Dos frentes independientes (cosmogénesis del espacio y quiralidad de partículas) chocaron con la misma pared: **falta el sustrato de campo con ángulo/curvatura.** Eso sube la confianza en que el muro es real y estructural, no un bug de una implementación.

---

## 5. La bifurcación abierta (decisión de Alexis)

- **A — `cg003e_planitud.py`:** el campo con **holonomía≈0** (crecer imponiendo que los ángulos cierren, teselar un plano en vez de un árbol hiperbólico). Predicción dura y falsable: entonces diámetro `~N^{1/d}` y `d` **converge**. Barato (minutos). *Diseñado, listo para escribir.*
- **B — pensar primero:** ¿la planitud puede *inyectarse* legítimamente, o debe **emerger**? (Es, literalmente, el *flatness problem* de la cosmología real: ¿por qué el universo es casi plano?) Quizá la planitud no es input, sino lo que la dinámica de exergía debería **seleccionar**.

**Postura de CC:** A y B no compiten — A *prueba* si la planitud es el ingrediente que falta, y su número alimenta a B.

---

## 6. Qué te pido, Claude Science

Un par de miradas frescas, adversariales, sobre:
1. **La medición.** ¿`N(r)~r^d` vs `e^{br}` es el discriminante correcto de geometría-vs-mundo-pequeño a estos N, o hay un estimador de dimensión menos sesgado por el rango corto (p.ej. dimensión de Hausdorff por conteo de cajas relacional, o el espectro del laplaciano con corrección de tamaño finito)? El ajuste de `d` sobre un rango que se ensancha con N es sospechoso.
2. **El diagnóstico de planitud.** ¿Coincides en que el diámetro `~log N` (no `~N^{1/d}`) es la prueba de que seguimos en régimen hiperbólico/mundo-pequeño? ¿Y en que holonomía≈0 es la condición faltante?
3. **La legitimidad epistémica** (para B): imponer planitud, ¿es "dibujar la caja por la puerta de atrás" o es un ingrediente físico legítimo del campo (como lo son la 2ª ley y la economía que ya usamos)?

### Archivos clave
- `cg003_espacio_relacional.py`, `cg003b_exergia_primero.py`, `cg003c_crecimiento.py`, `cg003d_campo_angular.py`
- `cg003_espacio_relacional.csv` (única salida a disco; b/c/d salieron por stdout)
- `CC_TO_GROK.md` (hasta Msg 46) / `GROK_TO_CC.md` — el canal con Grok/Diotallevi
- `Sesión final Cosmogenesis Claude Codex.md` — acta del arco previo CG002 (constantes/κ_Δ), trasfondo

*Regla de método: la teoría guía las preguntas, los datos condicionan las respuestas. Donde el dato dijo "no", escribimos "no".* 🜂
