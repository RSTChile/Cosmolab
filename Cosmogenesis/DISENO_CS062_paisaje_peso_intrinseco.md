# DISEÑO CS062 — EL PAISAJE CON GRAVEDAD ∝ PESO-INTRÍNSECO (no grado): ¿el 3D emerge más en todo el mapa cuando la gravedad se acopla a lo que físicamente le corresponde? (releer el negativo central del arco)

**Rama:** Cosmogénesis · **Nº:** CS062 (dimensión técnica: re-corrida del paisaje CS057 con acople gravitatorio corregido).
**Diseño:** CS · **Origen:** la grieta positiva de CS060-B + la observación de Alexis (gravedad-sin-masa).
**Fecha:** 5-jul-2026. **Estado:** DISEÑO, a codear por CC. **Prioridad: 1 (★, la barata y concreta).**
**Base:** cs057_paisaje_completo.py (la máquina completa, se REUSA) + adjudicacion_ARCO_CS058-061_CS.md +
AUDITORIA_gravedad_sin_masa_CS.md.

---

## 0. LA PREGUNTA, EN UNA LÍNEA
CS057 corrió la gravedad acoplada al GRADO del nodo (ρ=nº de vínculos) — un proxy que, como mostró CS060-B,
se AUTO-AMPLIFICA (los hubs atraen más → colapso a dimensión baja → sesgo ACTIVO contra el 3D). CS062
pregunta: **si la gravedad se acopla a un PESO INTRÍNSECO FIJO (como la masa real, no al grado que crece
sola), ¿el 3D-plano emerge más en TODO el paisaje — o el negativo de CS057 se sostiene igual?**

## 1. POR QUÉ ESTE EXPERIMENTO (y por qué es barato)
- CS060-B ya vio, en un zoom local, que gravedad∝peso da 3D/4D ~2× más viable que gravedad∝grado (viab 0.211
  vs 0.106). Pero fue LOCAL. CS062 lo lleva al MAPA COMPLETO: ¿es un efecto local o mueve el paisaje entero?
- **Reusa la máquina de CS057 casi intacta** — mismo Sobol, mismos ejes, mismo criterio ciego, mismos
  guardianes. El ÚNICO cambio es la línea del acople gravitatorio. Por eso es barato: días de cómputo ya
  probados, un cambio quirúrgico.
- Ataca el hallazgo MÁS concreto del arco de cierre: puede RELEER el negativo central (CS057) o CONFIRMARLO
  con la gravedad correcta — cualquiera es un resultado limpio.

## 2. EL CAMBIO EXACTO (quirúrgico, una función)
En cs057, la gravedad hace `rho = [len(a) for a in adj]` y pondera `rho[j]/d^α`. CS062:
- **Asignar a cada nodo un PESO INTRÍNSECO `m_i` al nacer** (una vez, fijo — NO el grado, NO se recalcula
  del nº de vínculos en cada paso). Distribución a elegir por CC (uniforme, o log-normal como las masas
  reales), DECLARADA.
- **La gravedad pondera por `m_i·m_j/d^α`** (masa×masa/distancia², la ley real de Newton) en vez de
  `ρ_j/d^α`. La atracción ya NO se auto-amplifica: un nodo no atrae más por haberse conectado más.
- **TODO lo demás de CS057 idéntico:** Sobol del hipercubo, los 6 pesos de fuerza, alcances, sync/async,
  criterio viable=estable∧expande ciego por tipos, punto físico marcado, sector oscuro emergente.

## 3. LOS BRAZOS (para que la diferencia sea legible)
- **Brazo PESO** (nuevo): gravedad ∝ m_i·m_j/d^α, peso intrínseco fijo.
- **Brazo GRADO** (= CS057, control directo): gravedad ∝ ρ. Correr AMBOS en el mismo arnés para que
  PESO−GRADO sea legible celda por celda.
- **Brazo NULL-PESO** (crítico, lección de CS060-B): pesos intrínsecos BARAJADOS entre nodos. Si el efecto
  del brazo PESO se sostiene bajo barajado, NO es la estructura del peso — es solo la independencia-del-grado
  (que ya sabíamos). Debe medirse cuánto del efecto es "peso real" vs "cualquier cosa que no sea grado".

## 4. GUARDIANES
1. **G-PESO-INTRÍNSECO-FIJO:** m_i se asigna al nacer y NO se recalcula del grado. Assert: la gravedad nunca
   lee len(adj) como masa.
2. **G-PESO-SEPARADO-DEL-GRADO (de CS060):** correlación(m_i, grado_i) reportada; deben ser independientes.
3. **G-MISMO-CRITERIO-QUE-CS057:** el criterio viable/estable/expande NO cambia respecto a CS057 (para que
  la comparación sea válida). Solo cambia el acople gravitatorio.
4. **G-NULL-PESO:** el brazo de pesos barajados separa "peso real" de "no-grado". Obligatorio.
5. **G-NO-FORZAR-3D + G-PREDICCIÓN-CIEGA:** predicción escrita antes; éxito ≠ "salió 3D", éxito = medir
   HONESTAMENTE cuánto mueve el 3D el acople correcto respecto al proxy.

## 5. LOS DESENLACES (pre-escritos)
- **(A) Con gravedad∝peso el 3D/4D emerge MÁS en todo el paisaje, y NO bajo NULL-peso → el negativo de CS057
  era en parte artefacto del proxy de grado.** Hallazgo mayor: la gravedad correcta es más amiga del 3D. Se
  relee el arco de fuerzas entero. (No prueba que el 3D se seleccione — pero mueve el mapa.)
- **(B) El 3D sube algo PERO también bajo NULL-peso → el efecto es independencia-del-grado, no el peso como
  tal.** Confirma CS060-B a escala: el proxy de grado sesgaba, pero el peso intrínseco no añade selección
  propia. Se corrige el registro (usar peso, no grado) sin reclamar más.
- **(C) El paisaje NO cambia (peso ≈ grado en el mapa completo) → el negativo de CS057 se SOSTIENE con la
  gravedad correcta.** El proxy no era el culpable; el 3D sigue sin emerger. Negativo reforzado, más limpio.

## 6. RESUMEN OPERATIVO PARA CC
- Clonar cs057_paisaje_completo.py; cambiar SOLO el acople gravitatorio a m_i·m_j/d^α con m_i intrínseco fijo.
- Tres brazos: PESO / GRADO(=CS057) / NULL-PESO. Mismo Sobol, mismos ejes, mismo criterio, mismos guardianes
  de CS057. Predicción ciega antes.
- Medir: fracción viable 3D y 4D por brazo, en el mapa completo Y en el punto físico. Reportar PESO−GRADO y
  PESO−NULL celda por celda. Correlación m_i↔grado (debe ser ~0).
- Entregar CSV + figuras (mapa 3D-viable PESO vs GRADO; barras 3D/4D por brazo real vs NULL) + informe.
  Traer a CS. Registrar CS062.

— Diseño CS062 por CS. La observación que lo obliga (gravedad sin masa = proxy de grado sesgado) es de
Alexis; el hallazgo local que lo motiva (CS060-B) es de CC; la estructura de re-corrida, los brazos y el
NULL-peso que separa peso-real de no-grado, míos. Puede releer el negativo central del arco o confirmarlo con
la gravedad correcta — cualquiera es limpio.
