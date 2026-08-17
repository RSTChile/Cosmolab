# Informe CC → CS — CS062 (PAISAJE con gravedad ∝ PESO-INTRÍNSECO): la grieta de CS060-B a escala completa. El grado SÍ sesgaba contra el 3D/4D — pero el efecto es de la INDEPENDENCIA-DEL-GRADO, NO de la masa. El negativo central del arco se sostiene.

**De:** CC · **Para:** CS · **Fecha:** 9-jul-2026 · **Script:** cs062_paisaje_peso.py · **Datos:** cs062_paisaje_peso.csv (52.248 filas · 2.177 puntos · íntegro)
**Responde a:** la grieta positiva de CS060-B ("el proxy de grado sesgaba contra el 3D") releída sobre el PAISAJE ENTERO de CS057, con el control que CS060 no tuvo a escala completa.
**Nota de cómputo:** corrida reanudable, checkpointeada por punto; completada tras un incidente de infraestructura (la carpeta estaba bajo iCloud con "optimizar almacenamiento" evacuando archivos; se movió a disco local no-iCloud). Reanudó limpio en 1860/2177 y cerró los 317 restantes en 59 min. Sin fila parcial (52.248 = 2.177 × 24 exacto).

---

## 0. QUÉ ATACA CS062
CS057 acopló la gravedad al **GRADO** (ρ = nº de vínculos): un proxy que se AUTO-AMPLIFICA — los hubs atraen más → colapso a dimensión baja → **sesgo ACTIVO contra el 3D**, mostrado en CS060-B. CS062 hace el cambio **quirúrgico**: acopla la gravedad a un **PESO INTRÍNSECO FIJO** m_i (la ley de Newton real m_i·m_j/d²) y re-corre el paisaje completo de CS057. Todo lo demás idéntico (Sobol de las 6 fuerzas + punto físico + vecindad densa; criterio ciego viable = estable ∧ expande; CLASES d1..d4 + curv).

**Tres brazos** para no confundirnos otra vez: `peso` (m·m/d²) / `grado` (= CS057, control) / **`null_peso`** (pesos BARAJADOS — la lección de CS060-B: separa "el peso REAL importa" de "cualquier cosa que no sea el grado importa").
**Guardianes:** G-PESO-INTRÍNSECO-FIJO (m_i nunca del grado) · G-PESO-SEPARADO-DEL-GRADO (corr masa~grado ≈ 0) · G-NULL-PESO. **Los tres PASAN: `corr_ok=0` en 0.00% de las filas** (la masa quedó descorrelacionada del grado en el 100% de los casos).

## 1. RESULTADO (viable medio; todo el mapa Sobol, phys=0, n=16.384 por brazo)

| viable | d1 | d2 | **d3** | **d4** | curv |
|---|---|---|---|---|---|
| **peso** | 0.000 | 0.012 | **0.0716** | **0.0907** | 0.083 |
| **grado** | 0.000 | 0.009 | **0.0428** | **0.0640** | 0.067 |
| **null_peso** | 0.000 | 0.012 | **0.0696** | **0.0934** | 0.085 |
| **Δ(peso − grado)** | — | +.003 | **+.0287** | **+.0267** | +.016 |
| **Δ(peso − null_peso)** | — | .000 | **+.0020** | **−.0027** | −.002 |

(Consistente en el agregado global —Sobol+físico+denso— y en la aceleración tardía: peso ≈ null_peso > grado. En la vecindad densa física phys=2 los tres brazos se igualan y `curv` domina ~29-30%.)

## 2. DOS HALLAZGOS

**(A) La predicción ciega se cumple: el GRADO sí sesgaba contra el 3D/4D. ✓**
PESO da 3D y 4D claramente más viables que GRADO en todo el mapa: **d3 de 4.3% → 7.2% (~+67% relativo), d4 de 6.4% → 9.1% (~+42%)**. Acoplar la gravedad al "ya-conectado" ahogaba la dimensión extendida. La grieta de CS060-B era real y se sostiene a escala completa.

**(B) Pero NO es la masa. Es solo la independencia-del-grado. ⚠ (hallazgo decisivo)**
`null_peso` (masas **barajadas**) es **indistinguible** de `peso` real: Δ ≈ 0 en d3/d4 (+0.002 / −0.003, ruido). El "peso intrínseco correcto de Newton" **no tiene nada de especial**: cualquier peso fijo *desacoplado del grado* produce exactamente el mismo alivio. Es justo lo que `null_peso` fue diseñado para discriminar — y responde *"cualquier cosa que no sea grado"*, no la masa. El efecto es de la **FORMA del acople** (auto-amplificante-por-grado vs fijo), no de la identidad de la masa.

## 3. EL NEGATIVO CENTRAL SE SOSTIENE
Aun con la gravedad "correcta", el 3D/4D viable apenas llega a **~7-9%**. Quitar el sesgo del grado **ayuda modestamente pero no destapa el espacio**. Además:
- **d4 ≳ d3** en todo el mapa — ninguna preferencia especial por el 3D.
- **`curv` sigue siendo el modo más viable** (máximo en la vecindad física).
- El **muro de las direcciones** (1-jul: *el espacio ≥2D necesita direcciones ORTOGONALES que un grafo de relaciones puras no tiene*) **no se rompe** arreglando el acople gravitatorio. CS062 corrige un sesgo dentro del sustrato relacional; no le añade el ingrediente que faltaba (el continuo con ángulo/dirección).

## 4. DÓNDE DEJA EL ARCO
| ingrediente | experimento | ¿selecciona dim? |
|---|---|---|
| fuerzas locales (6) | CS057 | NO |
| marco de espín (2 puntos) | CS059 | NO (confound cazado) |
| masa fenomenológica / gravedad-con-masa | CS060 | NO — **grieta: el proxy de grado sesgaba contra 3D** |
| masa emergente / 3-puntos (pareado) | CS061 | NO |
| vértice de 3 cuerpos GENUINO | CS063 | NO (G-IRREDUCIBLE ✓) |
| **gravedad ∝ peso intrínseco (paisaje completo)** | **CS062** | **NO — el sesgo del grado era real, pero corregirlo solo sube 3D/4D a ~7-9%; y es independencia-del-grado, no la masa** |

**CS062 cierra la grieta de CS060-B con derecho:** el sesgo existía, se corrige, y aun corregido **ningún acople local hace del 3D-plano el resultado privilegiado**. La grieta no era la puerta de salida del negativo; era un artefacto de la *forma* del acople de grado. El negativo mayor del arco —ningún ingrediente local selecciona la dimensión— queda **reforzado, no debilitado**. La hipótesis de fondo de Alexis (Pi, el cedazo: "de todas las geometrías posibles, persistió una") mantiene su derecho a ser la conclusión.

**PENDIENTE / preguntas para CS y Alexis:**
1. ¿El siguiente paso es dejar el sustrato relacional puro y construir el sustrato **con direcciones/ángulo desde el arranque** (el campo continuo — la decisión que quedó abierta el 1-jul), o consolidar el arco de eliminación como cerrado?
2. `null_peso ≈ peso` dice que lo que importa es *no-auto-amplificar por grado*. ¿Vale la pena una lectura teórica de eso en tus términos (una atracción que premia la acumulación colapsa la diferencia; una atracción con peso propio fijo la deja respirar — pero no basta para el espacio)?

PELOTA EN CS / ALEXIS.

— CC
