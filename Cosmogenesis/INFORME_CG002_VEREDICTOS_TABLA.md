# CG002 — Tabla de veredictos por experimento

**Programa:** Cosmogénesis · RMD 2.0 · Club Abulafia  
**Versión:** autoritativa · 30-jun-2026  
**Equipo:** Alexis López Tapia (dirección) · CC (motor) · Grok (barridos) · Claude web (regla, verificación)

Leyenda de veredictos:

| Símbolo | Significado |
|---------|-------------|
| ✅ | Confirmado / derivación estructural |
| ⊘ | Parcial, calificado o tensión documentada |
| ❌ | Refutado o no emergente |
| — | N/A en este régimen |
| 🔒 | Cerrado (arco κ_Δ) |

---

## A. Arco fundacional (`cg002_experimentos_arco.py`)

| ID | Experimento | Script / comando | Nodo(s) | Pregunta | Resultado clave | Veredicto |
|----|-------------|------------------|---------|----------|-----------------|-----------|
| A1 | Baryogénesis | `baryo` | C-N2.5.10 | ¿Aniquilación y exceso sin programarlos? | ~50% sobrevive; a0→a1 amplificado ~25×; idénticas→100% sin estructura | ✅ Emergen |
| A2 | Asimetría = diferencia | (lectura A1) | C-N2.5.5 ⟺ C-N1 | ¿Contradicción asimetría vs θ=0? | Promedio cero ≠ sin diferencia; sin diferencia → sin exceso | ✅ Unificado |
| A3 | Flecha del tiempo | `flecha` | C-N3 | ¿Ω decrece monótono? | Vivos y entropía orientación decrecen, nunca suben | ✅ Flecha |
| A4 | Dimensión del sustrato | `dimension` | C-N2.5, C-N2.6 | ¿D_corr rastrea dim intrínseca? | S¹→0.97, S²→1.87, S³→2.69, S⁴→3.44 (R²≈1) | ✅ Herencia |
| A5 | Coexistencia | `coexistencia` | C-N2.5.9 | ¿Dominios locales coexisten? | rc bajo: local≫global; rc alto: un dominio | ✅ Coexistencia |
| A6 | Criticidad | `criticidad` | C-N2.7 | ¿Transición de fase por alcance? | Pico fluctuación rc≈3; crece con N (preliminar) | ✅ Transición |
| A7 | Auto-similaridad | `cg002_genesis_barridos.js` | C-N2.6.4 | ¿Ley de potencias en crítico? | R²=0.98 en rc≈1.0 | ✅ (⊘ ajuste generoso) |
| A8 | Inercia histórica | `inercia` | C-N2.5.8, C-N5.1 | ¿Duración da resistencia? | Banda dura: umbral ×2 siempre; blanda: >×8 no voltea | ⊘ Tensión C-N5.1 |
| A9 | Constantes vs historia (20) | `constantes` | C-N2.8.15 | ¿Razones convergen entre cosmos? | f≈0.509, \|orden\|≈0.516 (CV<2%); dirección ~90° | ✅ Ley vs historia |
| A10 | Invariantes / ruptura | `invariantes` | C-N2.8, C-N2.8.9a | ¿Tipologías de ruptura distintas? | κ_Δ, κ_V, κ_P cualitativamente distintos | ✅ Tipología |
| A11 | Paredes de dominio | `paredes`, `colision` | C-N2.5.10 | ¿Fronteras nítidas? | Déficit ~3–5 pts, independiente ángulo | ❌ No robusto |
| A12 | Cuatro fuerces espontáneas | — | C-N2.7.6 | ¿Regímenes discretos emergen? | No como derivación espontánea | ❌ Fuera de CG002 |

---

## B. Arco constantes y κ_Δ (30-jun-2026)

| ID | Experimento | Script / artefacto | Nodo(s) | Pregunta | Resultado clave | Veredicto |
|----|-------------|-------------------|---------|----------|-----------------|-----------|
| B1 | 1000 cosmos | `cg002_constantes_1000.py` | C-N2.8.15 | ¿Constantes robustas? | f=0.506 CV 1.96%; \|orden\|=0.515 CV 1.86%; dir≈90° | ✅ Reproducido |
| B2 | Trivialidad hemisferio | análisis CC/Grok | C-N1.3 | ¿f≈½, orden≈½ son ley? | frac_pos≈1; nulo geom 0.5; exceso real ~0.016 | ⊘ Trivial + residuo |
| B3 | Exceso vs η,μ,S_BAND,d | `cg002_exceso_barrido.py` | C-N2.8.4, C-N2.8.14a | ¿Exceso aguanta barrido? | +0.015 estable a d=3; modulado por d | ✅ Invariante estructural |
| B4 | Nulo muestra finita | `cg002_exceso_caracteriza.py` | — | ¿Exceso es sesgo finito? | Sesgo 5–12%; 88–95% dinámica real | ✅ Real |
| B5 | Grain null combinatorio | `grain_null_model.py` | — | ¿Línea base 1/√N? | exp=−0.500; √(7/8N) a K=8,N=2000 | ✅ Nulo exacto |
| B6 | L2 dinámico vs null | `cg002_dynamic_l2_sweep.py` | C-N2.8.4 | ¿Motor se aparta del nulo? | L2≈0.354 plano; ratio 12× a N=2000 | ✅ Estructura κ_Δ |
| B7 | m_eff/K vs banda 6× | `cierre_kappaDelta_regla.py`, `engine_output.csv` | C-N2.5.10, C-N5.1 | ¿½ es derivación o calibración? | cambio 0.0000; K∈{6,8,12}; N-indep. | ✅ DERIVACIÓN |
| B8 | Grano 1/√K | reducción participación | C-N2.8.4 | ¿Forma del grano? | m_eff/K=½ ⟺ grano=1/√K | ✅ Identificación |

**Relectura B1–B2 (post-arco):** f≈½ y \|orden\|≈½ = geometría del hemisferio (trivial). Lo no-trivial = exceso dinámico (capa fina) + L2 estructural (capa gruesa).

---

## C. Motor producción y tests puntuales

| ID | Experimento | Script | Nodo(s) | Pregunta | Resultado clave | Veredicto |
|----|-------------|--------|---------|----------|-----------------|-----------|
| C1 | Acoplamiento originario v0.1c | `cg002_acoplamiento.py` | C-N2.5.7 | ¿B PASS dirección? | B PASS cualificado; tracking con caveat | ✅ Cualificado |
| C2 | Firma multicomponente v0.2 | `cg002_multicomponente.py` | C-N2.6 | ¿Rango = d? | d=3 → rango 3 intrínseco | ✅ Dimensión 3D |
| C3 | CG004 nube ΣU≈0 | (CC) | C-N2.7.4 | ¿Self-term universal? | G1 NO (tipo-carga); G2 SÍ (escala ΣS) | ❌ Gravedad fuera CG002 |
| C4 | Génesis visor | `cg002_genesis.html` | C-N2.6 | ¿Clumping emergente? | ~2.53 inhomogeneidad; sin órbitas (layout) | ✅ Clumping; ⊘ órbita |

---

## D. Síntesis por estatuto (arco κ_Δ cerrado)

| Capa | Observable | Escala | Estatuto | Veredicto |
|------|------------|--------|----------|-----------|
| **Gruesa** | L2 fase / m_eff/K = ½ | 1/√K, N-independiente | Derivada — C-N2.5.10 | 🔒 ✅ DERIVACIÓN |
| **Fina** | \|ū\| − nulo hemisferio | exceso ~0.015 a d=3; η,μ,S_BAND estables | Invariante estructural; magnitud dominio-específica (C-N2.8.14a) | 🔒 ✅ |
| **Trivial** | f≈½, \|orden\|≈½ | geometría hemisferio | C-N1.3 filtración | Documentado, no invariante independiente |
| **Historia** | Dirección flecha | ~90° entre cosmos | C-N3 contingente | ✅ |

---

## E. Alcance operativo (no epistemológico)

| Límite | Alcance medido | Estado |
|--------|----------------|--------|
| Régimen | θ_CP=0; S_BAND ×0.5…3.0; K∈{6,8,12}; N∈{250…4000} | Documentado |
| Reproducibilidad | CSV + scripts en `Cosmogenesis/` | ✅ |
| Identificación κ_Δ | κ_Δ ≡ 2π/K (revisable) | Operativo abierto |
| Próximo blanco | θ_CP≠0; regímenes no barridos | Programa siguiente |

---

*Reproducir arco A: `python3 cg002_experimentos_arco.py todos` · Arco B: ver tabla de archivos en `NOMENCLATURA_NODOS_CG002.md` y `CIERRE_ARCO_CG002_AUTORITATIVO.md`.*