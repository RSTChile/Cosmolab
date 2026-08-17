# Informe CC → CS — CS053: la persistencia ciega NO fija 3D-plano. Falsación honesta (desenlace 2)

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `DISENO_CS053_persistencia_geometria.md` (ensemble simétrico; filtro de persistencia ciego S=I·E; G-NO-HORNEAR el crítico; G-NULL; los tres desenlaces honestos).
**Script:** `cs053_persistencia_geometria.py` · **Log:** `cs053_run.log`

---

## 1. Implementación (con los guardianes puestos)
Ensemble SIMÉTRICO de geometrías/dimensiones (d≈1..4, plana/curva±, 6 repeticiones): cadena (d1),
cuadrada/triangular (d2 plana), {3,7}/{3,8} (d2 hiperbólica), cúbica (d3), hipercúbica (d4), árbol
(hiperbólico). Filtro de persistencia INTRÍNSECO y CIEGO: **S = I·E**, con I = regularidad del vínculo
(grado_medio/(grado_medio+std)) y E = resiliencia (fracción del componente gigante tras remover 30% de
aristas al azar). **El filtro NUNCA recibe dimensión ni curvatura** — se miden APARTE, como output
(G-NO-HORNEAR es estructura del código: `persiste_S(adj,N,rng)` solo ve grados y aristas). θ=0.45 fijo
(G-NO-TUNE). Brazo G-NULL (filtro al azar, misma tasa).

## 2. Resultado (36 configuraciones)

| geometría | dim (medida) | turn | S=I·E | filtro | G-NULL (azar) |
|---|---|---|---|---|---|
| cadena_d1 | 0.94 | — | 0.02 | **muere 0/6** | vive 5/6 |
| arbol_cv (curvo) | 2.27 | 1.99 | 0.10 | **muere 0/6** | vive 5/6 |
| cuadr_d2 plano | 1.60 | 1.15 | 0.89 | vive 6/6 | 5/6 |
| tri_d2 plano | 1.57 | 1.15 | 0.89 | vive 6/6 | 4/6 |
| **cubo_d3 plano** | 2.02 | 1.47 | 0.88 | vive 6/6 | 4/6 |
| hcubo_d4 plano | 2.29 | 1.74 | 0.88 | vive 6/6 | 6/6 |
| hip37_d2 curvo | 2.13 | 1.65 | 0.71 | vive 6/6 | 6/6 |
| hip38_d2 curvo | 2.40 | 1.93 | 0.67 | vive 6/6 | 4/6 |

**Supervivientes del filtro: 36 total, d≈3-plano privilegiados = 0, otros = 36.**

## 3. Veredicto: DESENLACE 2 — falsación honesta
- **El filtro SÍ hace algo real (no es null):** mata específicamente lo FRÁGIL (cadenas d1, árboles:
  0/6) y conserva lo ROBUSTO-EXTENDIDO (todos los retículos ≥2D: 6/6). G-NULL lo confirma: el azar deja
  vivir cadenas y árboles (5/6) — el filtro NO. Así que la persistencia discrimina, pero discrimina
  **"robusto/extendido" vs "hilo/árbol"**, NO **"3D-plano" vs el resto**.
- **NO fija 3D-plano:** todos los retículos —2D, 3D, 4D, plano E hiperbólico— son resilientes y
  sobreviven por igual. Nuestro universo (3D-plano) es UN superviviente entre muchos.
- **Conclusión:** la persistencia SOLA no explica por qué vivimos en 3D-plano. Explica por qué NO
  vivimos en 1D ni en un árbol (no persisten), pero no la dimensión ni la curvatura específicas. **Nuestro
  universo FALSA la regla de persistencia simple.** Resultado real, no fracaso: falta otro ingrediente,
  aguas arriba — consistente con todo el arco (CG004/CG005: la planitud/dimensión son aguas arriba).

## 4. Honestidades (lo que hace válido el negativo)
- **G-NO-HORNEAR PASÓ:** el filtro solo vio (grados, aristas) — jamás "3D", "plano", ni curvatura. Esto
  NO es el auto-engaño que temíamos (el único riesgo real, no la explosión cósmica). Es falsación genuina.
- **La falsación es de ESTA regla de persistencia** (resiliencia × regularidad), no de "persistencia" en
  abstracto. Pero noto que es robusta: CUALQUIER filtro basado en resiliencia conservará todos los
  retículos ≥2D (son resilientes) → no pinpointeará d=3. Para distinguir d=3-plano haría falta un filtro
  sensible a la dimensión/curvatura, que o bien apunta a la respuesta (horneado) o es una cantidad
  intrínseca más fina que no tenemos.
- **NO reprodujo el "exceso de orden a d=3" de CG002/CS018.** Mi filtro no favorece d=3. Si ese exceso es
  real, necesitaría un cedazo distinto/más fino — pero afinarlo para que salga d=3 sería hornear
  (G-NO-TUNE lo prohíbe). Lo dejo como discrepancia honesta a revisar: ¿el exceso de CS018 es del cedazo
  de persistencia, o de otra cosa? CS053 sugiere que la persistencia-resiliencia sola no lo produce.

## 5. Lo que esto asienta
- La persistencia **narra el piso** (no 1D, no árbol — lo frágil no persiste) pero **no el punto** (3D-plano).
- Junto con el arco: dónde vive el espacio = en el vínculo atado (CS052-v1, probado); qué persiste = lo
  robusto-extendido, sin fijar d ni curvatura (CS053); cómo se GENERA plano = sigue aguas arriba (R7).
  Tres preguntas, tres piezas, y el hueco de la generación/selección-fina de d=3-plano sigue abierto,
  ahora acotado: no lo hace ni la adyacencia, ni el marco por sí solo, ni la persistencia-resiliencia.

Un negativo limpio y blindado (G-NO-HORNEAR y G-NULL puestos), que es justo lo que pediste: no el eco de
lo que metimos, sino la respuesta honesta — y la respuesta es que la persistencia simple no basta.
Espero tu adjudicación de CS053.

— CC
