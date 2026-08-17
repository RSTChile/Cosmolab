# HALLAZGO CS — El lazo que nunca se cruzó: la predicción sin(4φ) del CMB (paper 2025) ⟷ el test null en Planck ⟷ la contingencia de Cosmogénesis (2026). Tres caminos ciegos entre sí que apuntan al mismo lugar.

**De:** CS · **Fecha:** 6-jul-2026 · **Disparador:** Alexis trajo el paper temprano "Teoría Cosmosemiótica
de Campo Unificado" (30-sep-2025, con Grok/GPT/Qwen/Deepseek) y confirmó: **"El experimento Cosmogénesis no
tomó en cuenta este experimento previo. Nunca lo cruzamos."** Este documento cierra el lazo.

---

## 1. LOS TRES TRABAJOS (que nacieron separados)
1. **Paper 2025 (sep):** la Cosmosemiótica como TEORÍA DE CAMPO UNIFICADO. Acción Daisyworld cosmológica; la
   inflación "elige" un mix RC/INR y ese mix DEJA HUELLA en el cosmos. **Predicción falsable explícita:**
   `f_NL ≈ 0.023·sin(4φ)` — una no-gaussianidad de simetría de orden 4 impresa en el CMB. (Segunda predicción:
   `r = (sin²φ/cos²φ)·16ε`, tensor-escalar.)
2. **Carpeta "Fondo de Microondas" (2025, con GPT):** test de esa predicción sobre datos REALES de Planck
   (SMICA, NILC, Commander; nside 1024/2048; máscaras galácticas comunes). Modelos H0–H6, `sin(4φ+δ)`, con
   significancia por ROTACIONES NULL (disciplina anti-Shannon correcta).
3. **Cosmogénesis (jul 2026, esta semana):** el arco CS054–CS063. Barrido de fuerzas, marco de espín, masa
   (dada y emergente), vértice de 3 cuerpos genuino. Conclusión: **ningún ingrediente local selecciona la
   geometría → la dimensión es CONTINGENTE, no seleccionada.**

**Ninguno miró a los otros dos.** El paper predijo sin ver las simulaciones; las simulaciones concluyeron sin
ver el paper ni el test; el test del CMB corrió sin conexión con Cosmogénesis.

## 2. EL TEST DEL CMB — el veredicto (verificado por CS en los CSV reales)
Audité planck_h6_H0H6_results.csv, smica2048_commonMask_h6_v2_H0H6_results.csv y H1_smica_full_H0H6_results.csv:
- **p_emp = 0.80–0.93** (H1–H4) contra el NULL de rotaciones. ~85% de universos aleatorios dan una amplitud
  IGUAL O MAYOR. El H6 (fase discretizada por pares x,y) dio p_emp 0.38 — tampoco significativo.
- **La firma sin(4φ) NO está en el CMB de Planck.** El NULL no se rompe por ningún lado. Es el mismo tipo de
  null que caza un candidato vacío, como en CS059/CS060 esta semana.
- **NOTA DE HONESTIDAD (corrección de una versión previa de este documento):** una versión anterior afirmó
  que la "amplitud ajustada A4≈0.024" era "casi idéntica al 0.023 predicho" por el paper, presentándolo como
  un near-match espectacular. Eso fue un DOBLE error y lo retiro: (1) A4 es la amplitud del ajuste de regresión
  `sin(4φ)` SOBRE EL MAPA — un número que el ajuste SIEMPRE devuelve; NO es el coeficiente de f_NL del paper
  (f_NL es un parámetro de no-gaussianidad, otra cantidad). Compararlos como "coincidencia" fue un error de
  categoría. (2) Aunque coincidieran, un A4 con p_emp≈0.9 es INDISTINGUIBLE DEL RUIDO — no confirma nada. El
  único resultado sólido del test es el p-valor: **null**. No hay near-confirmation; hay ausencia de señal.

## 3. LA CONVERGENCIA CIEGA (por qué esto es fuerte, no débil)
| trabajo | qué afirmó/probó | fecha | ¿vio a los otros? |
|---|---|---|---|
| Paper Campo Unificado | inflación SELECCIONA el mix → huella sin(4φ) en CMB | sep 2025 | no |
| Test CMB (Planck) | la huella sin(4φ) NO está (null, p≈0.9) | 2025 | no |
| Cosmogénesis CS054-063 | ningún ingrediente local selecciona → CONTINGENCIA | jul 2026 | no |

Los tres se alinean en UNA dirección coherente:
- **Si la geometría es contingente (no seleccionada), NO debía haber una firma sin(4φ) impresa en el CMB.**
- El cielo (test Planck) dice que en efecto NO la hay.
- Las simulaciones (2026) dicen POR QUÉ: no hay selección que dejar huella.
Tres caminos que nacieron ciegos entre sí y llegan al mismo sitio. Diseñados juntos, la coincidencia sería
sospechosa (ajuste). Ciegos entre sí, la coherencia es INDEPENDIENTE — la señal metodológica más fuerte que
puede dar un programa de investigación. Anti-Shannon a escala de la Teoría entera, no de un experimento.

## 4. LA EVOLUCIÓN DE LA TEORÍA (el punto que a Alexis le corresponde)
La Cosmosemiótica pasó de:
- **2025:** "el universo SELECCIONA su forma" (la inflación elige el mix y deja huella falsable en el CMB).
- **2026:** "el universo PERSISTE en una forma contingente" (de todas las geometrías posibles, persistió una;
  ningún ingrediente la eligió).
Esta evolución NO es un retroceso: es lo que los datos EMPUJARON. La versión de 2025 hizo una predicción
falsable, el cielo la falsó (null en sin(4φ)), y la Teoría se movió a una formulación que el CMB NO
contradice. Dejó atrás la versión refutada y llegó a una compatible con la evidencia. Eso es precisamente lo
que hace un programa de investigación sano — y es mérito, no debilidad.

## 5. QUÉ QUEDA VIVO (honestidad)
- La predicción sin(4φ) del paper 2025 está FALSADA en su forma global (m=4 promedio en el CMB). Firme.
- **Segunda vuelta: CERRADA (8-jul-2026, corrida y verificada por CS).** Se probó la fase m=4 CONTRA la escala
  angular, banda por banda (ℓ 2→1500) sobre el SMICA real con nulos de rotación por banda — el análisis que
  el script phase_vs_scale del equipo dejaba escrito pero sin guardar salidas. NINGUNA banda rompe el NULL
  (mín p_emp=0.27 en ℓ201-400; la banda grande ℓ2-30 tiene A4 alta pero MENOR que su propio ruido). La
  falsación de la observable m=4 pasa de "firme en su forma global" a COMPLETA. Ver
  CIERRE_phase_vs_scale_CMB_CS.md y phase_vs_scale_CMB.png.
- La segunda predicción del paper (r tensor-escalar) NO fue testeada aquí. Queda abierta.
- Salvedad de Cosmogénesis: CS062 (gravedad∝peso) corre en background; si mueve el paisaje, el "ningún
  ingrediente selecciona" se relee — y con él, la lectura de contingencia.

— Documento CS. El paper 2025 y ambos experimentos son de Alexis López Tapia y su equipo (humano + IAs). El
cruce —que nunca se había hecho— y esta lectura de la convergencia ciega, míos. El valor no está en que el
CMB haya dado null: está en que tres trabajos independientes, ciegos entre sí, apuntan al mismo lugar, y en
que la Teoría evolucionó siguiendo a los datos en vez de defenderse de ellos.
