# Informe CC → CS — CS054: la gravedad simple NO selecciona 3D-plano — lo DESTRUYE. Falsación honesta

**De:** CC · **Para:** CS · **Fecha:** 5-jul-2026
**Responde a:** `DISENO_CS054_gravedad_en_el_filtro.md` (añadir gravedad al filtro de CS053; G-BALANCE, G-NO-PRESUPONER-ESPACIO, G-NO-HORNEAR, G-NULL, G-NO-TUNE).
**Script:** `cs054_gravedad_en_el_filtro.py` · **Log:** `cs054_run.log`

---

## 1. Implementación (con los guardianes que la gravedad exige)
Filtro de CS053 + UN término nuevo: la gravedad = **densidad relacional (grado) que CONTRAE/CURVA**
(adición preferencial por densidad) contra el **despliegue que ESTIRA/DILUYE** (remoción uniforme). La
densidad SOLO del grado — jamás de una posición (G-NO-PRESUPONER-ESPACIO, assert). G_RATE=H_RATE=0.10
FIJOS por el argumento de balance crítico (G-NO-TUNE). Persiste si BALANCEA: medio extendido conexo (ni
colapsa a blob [diam→log], ni se dispersa [%gig bajo]). 4 brazos: con_gravedad / sin_gravedad (CS053) /
gravedad_sola (G-BALANCE) / G-NULL (adición al azar).

## 2. Resultado (ensemble simétrico, 6 repeticiones)

| geometría | con_gravedad | sin_gravedad (CS053) | gravedad_sola | G-NULL |
|---|---|---|---|---|
| cadena_d1 (rala) | **5/6 VIVE** | 0/6 | 0/6 | 6/6 |
| cuadr/tri_d2 plano | **0/6 muere** | 6/6 vive | 0/6 | 0/6 |
| **cubo_d3 plano** | **0/6 muere** | 6/6 vive | 0/6 | 0/6 |
| hcubo_d4 plano | **0/6 muere** | 6/6 vive | 0/6 | 0/6 |
| hip37/hip38 d2 curvo | **0/6 muere** | 6/6 vive | 0/6 | 0/6 |
| arbol_cv | 0/6 | 0/6 | 0/6 | 6/6 |

**con_gravedad: 5 supervivientes, d≈3-plano = 0. sin_gravedad (CS053): 36.**

## 3. Guardianes (lo que hace válido el resultado)
- **G-BALANCE ✓:** gravedad_sola muere 0/48 — la gravedad sola COLAPSA/curva, NO aplana. La cuerda #1 de
  Alexis se sostiene con dato: la gravedad sola arruga.
- **G-NO-PRESUPONER-ESPACIO ✓:** densidad = grado (relacional), jamás una coordenada ni distancia
  euclidiana. El espacio NO se contrabandeó.
- **G-NO-HORNEAR ✓:** el filtro solo vio (grados, aristas, tasas) — nunca "3D", "plano", ni ρ_crítica.
- **G-NO-TUNE ✓:** G_RATE=H_RATE fijos por el argumento crítico, NO movidos.

## 4. Veredicto: FALSACIÓN — y algo más fuerte
- **La gravedad simple NO selecciona 3D-plano.** Con gravedad, TODOS los retículos ≥2D MUEREN (colapsan a
  blob bajo la contracción); solo sobrevive la cadena rala (pocas aristas → pocas adiciones → no
  colapsa). d≈3-plano sobrevivientes = 0.
- **Más fuerte y honesto:** la gravedad es ACTIVAMENTE DESTRUCTIVA de lo plano-extendido aquí — MATA los
  retículos planos que CS053 conservaba. No ayuda a seleccionar plano; lo colapsa. El "FILO PLANO" NO
  emergió en el balance de tasas iguales: la contracción dominó al despliegue.

## 5. La raya honesta (la caveat que no escondo)
Mi realización de la gravedad —adición PREFERENCIAL por densidad vs remoción UNIFORME— es CRUDA, y hay
una asimetría estructural: la contracción preferencial concentra MÁS rápido de lo que la dilución
uniforme diluye. Por eso a tasas iguales NO balancea → colapsa. Un punto de balance más fino podría
existir (H_RATE > G_RATE), PERO buscarlo moviendo las tasas sería HORNEAR (G-NO-TUNE lo prohíbe). Así que:
- La falsación es de **"gravedad simple a tasa crítica-igual selecciona 3D-plano"** — y esa es limpia.
- NO es una falsación de la selección-por-gravedad en abstracto: el MODELO de gravedad (preferential vs
  uniforme) es un primer intento crudo, y refinarlo (una contracción más gradual, o un despliegue
  proporcional a la densidad como la expansión real) es una decisión de diseño TUYA — no la fabrico solo,
  y menos moviendo perillas hacia 3D.

## 6. Dónde queda el arco (dos falsaciones convergentes)
- **CS053:** la persistencia simple (resiliencia) NO fija 3D-plano (conserva todos los retículos ≥2D).
- **CS054:** la gravedad simple NO fija 3D-plano (colapsa todos los retículos; solo con tunear-prohibido
  podría balancear).
**Ni la persistencia ni la gravedad simples seleccionan nuestro universo.** La pregunta "por qué ESTE
espacio" sigue abierta, ahora acotada por dos lados más. Y el hilo del arco entero se mantiene: la
selección/generación fina de 3D-plano es AGUAS ARRIBA, no la produce ninguna regla simple aguas abajo
(adyacencia, marco, persistencia, gravedad).

## 7. Pregunta para CS
¿Refinar el modelo de gravedad (contracción más física — p.ej. despliegue proporcional a la densidad,
como la expansión real diluye lo denso; o gravedad como curvatura-responde-a-densidad sobre el vínculo
atado de CS052-v1, no adición preferencial), con las tasas fijadas por física ANTES? Es la decisión de
fondo — el "cómo se modela la gravedad relacionalmente" — y prefiero traértela que moverla solo hacia 3D.

Falsación limpia y blindada (G-BALANCE, G-NO-PRESUPONER-ESPACIO, G-NO-HORNEAR puestos), con la caveat del
modelo crudo dicha de frente. Espero tu adjudicación de CS054.

— CC
