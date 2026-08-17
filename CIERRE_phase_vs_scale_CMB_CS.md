# CIERRE DEL CABO — Segunda vuelta del CMB: fase m=4 CONTRA escala angular. Ninguna banda rompe el NULL. La falsación de la firma sin(4φ) pasa de "global" a COMPLETA.

**De:** CS · **Fecha:** 8-jul-2026 · **Ejecutado y verificado por CS** (no delegado).
**Mapa:** COM_CMB_IQU-smica_2048_R3.00_full.fits (I), degradado a nside=1024. **Máscara:** común Planck R3
(77.9% de cielo válido). **Método:** por cada banda multipolar ℓ, se filtran los alm a la banda, se
reconstruye el mapa de banda, se ajusta A4=|(sin4φ,cos4φ)| y se obtiene p-valor por N=500 rotaciones SO(3)
de la coordenada φ (idéntico en espíritu al phase_vs_scale_h6.py del equipo, que estaba escrito pero cuyas
salidas NUNCA se habían guardado). Sub-muestreo de 200k píxeles válidos para el ajuste de los nulos.

## POR QUÉ ESTA VUELTA (lo que quedaba abierto)
La primera vuelta (2025) midió la firma sin(4φ) GLOBAL (promediada sobre todo el cielo) → null, p≈0.80–0.93.
Un promedio global puede LAVAR señal que viva en una escala angular concreta. La pregunta rigurosa: ¿hay
alguna banda de escala donde la fase m=4 SÍ supere al ruido? Esta es la respuesta.

## RESULTADO POR BANDA (verificado)
| banda ℓ    | A4 obs  | p_emp | veredicto |
|------------|---------|-------|-----------|
| 2–30       | 0.1068  | 0.944 | null (A4 < NULL mediana 0.140) |
| 31–80      | 0.0016  | 0.956 | null |
| 81–200     | 0.0009  | 0.499 | null |
| 201–400    | 0.0017  | 0.273 | null (mínimo de las seis) |
| 401–800    | 0.0047  | 0.956 | null |
| 801–1500   | 0.0023  | 0.493 | null |

**Mínimo p_emp = 0.27** (banda ℓ201-400). NINGUNA banda cerca del umbral 0.05. Sin corrección por
comparación múltiple (6 bandas) el resultado ya es null; con corrección lo sería aún más.

## EL DETALLE QUE IMPORTA (la banda grande engaña, como en el global)
La banda ℓ2-30 tiene la mayor amplitud (A4=0.107) — a simple vista parecería "señal". Pero su NULL mediana
es 0.140 y su p95 es 0.171: la amplitud observada es MENOR que la típica del ruido a esa escala. A escalas
grandes hay pocos modos, así que la amplitud fluctúa alto POR AZAR — y el NULL de rotación lo captura
exactamente. Es la misma lección de todo el arco: una amplitud grande no es señal si no le gana a su control.
(Mismo patrón que el A4≈0.024 del global que CS había malleído como near-match — corregido; aquí NO se
reincide.)

## VEREDICTO
La firma sin(4φ) predicha por el paper de campo unificado (2025) NO está en el CMB de Planck — **ni en el
promedio global (primera vuelta) NI en ninguna banda de escala angular (esta segunda vuelta)**. La falsación
pasa de "firme en su forma global" a **COMPLETA en la observable m=4**. El cabo que quedaba abierto en
HALLAZGO_lazo_CMB_Cosmogenesis_CS.md y en el capítulo de la Integrada queda CERRADO.

## LO QUE SIGUE EN PIE (honestidad, sin sobre-extender)
- Esto falsa la observable m=4 (no-gaussianidad de simetría 4) en fase, global y por escala. Es la predicción
  central y explícita del paper. Está cerrada.
- La SEGUNDA predicción del paper (relación tensor-escalar r) es OTRA observable, no testeada aquí. Sigue
  abierta — pero es un test distinto (modos B de polarización), no parte de este cabo.
- El cierre REFUERZA la lectura de contingencia: el cielo no muestra la huella de selección que la versión de
  2025 esperaba, consistente con el null de selección de Cosmogénesis (CS054-063).

— CS. Test corrido y verificado sobre datos reales de Planck. El script del equipo se reprodujo en su método;
sus salidas por banda, que nunca se habían guardado, quedan aquí por primera vez.
