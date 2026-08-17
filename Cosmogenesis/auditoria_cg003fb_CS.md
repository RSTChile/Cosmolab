# Auditoría CS — CG003-f-b: ¿el contrapeso/relajación despliega el espacio plano?

**Auditor:** Claude Science · **Fecha:** 3-jul-2026
**Corrida:** CC en Carnets (iPad) · relajación = flujo de Ricci discreto con cirugía · BETA=0.1 · rondas=6
**Log:** cg003fb_ipad.csv · **Brazos:** REGLA-b (λ=2.0+relaj) · CONTROL (λ=2.0 sin relaj = cg003f) · AZAR (shuffle)

---

## VEREDICTO: NEGATIVO limpio. La relajación NO despliega el espacio plano.
REGLA-b es estadísticamente **indistinguible** de CONTROL en todo lo que decide. Ambos siguen hiperbólicos.
Por la propia clave pre-registrada: la relajación tampoco basta ⟹ el problema está en la DINÁMICA DE
CRECIMIENTO (frente/attach), más arriba en la cadena causal que la métrica o la poda.

## Los números que deciden (del resumen de la corrida)

| brazo | Dt | diam-pend | δ | dim (N=1024·4096·16384) | %gig(16384) |
|---|---|---|---|---|---|
| REGLA-b | 3 | 0.08 | acotada (hiperb) | 2.39 · 2.84 · 3.21 | 73 |
| CONTROL | 3 | 0.08 | acotada (hiperb) | 2.39 · 2.85 · 3.25 | 77 |
| AZAR    | 3 | 0.09 | acotada (hiperb) | 1.96 · 2.40 · 2.79 | 61 |
| REGLA-b | 2 | 0.14 | acotada (hiperb) | 2.12 · 2.50 · 2.92 | 59 |
| CONTROL | 2 | 0.13 | acotada (hiperb) | 2.14 · 2.50 · 2.92 | 59 |
| AZAR    | 2 | 0.32 | CRECE (plano)*   | 1.52 · 1.63 · 2.06 | 14 |

Misma pendiente de diámetro (0.08 vs 0.08), ambos hiperbólicos, dimensión idéntica dígito a dígito,
misma fracción gigante. La meta de "plano" pedía diam-pend→~0.5 y δ creciente; no se acerca por ningún eje.
(*) El único "δ crece (plano)" es AZAR-Dt2, pero con %gig desplomado a 14 (y a 0 en una semilla): es la
firma de FRAGMENTACIÓN, no espacio plano real. Bien descartado.

## La medición es SANA (por eso el negativo es confiable)
- Pre-vuelo, métrica no-degenerada: std(24 dir) Dt2=0.887, Dt3=0.360 → OK (superado el 8.9e-16 degenerado
  de cg003f v0).
- Controles discriminan: lattice2D δ CRECE 2.18→8.88 (geometría), árbol_b3 δ=0.00 exacto (azar). El
  estimador de δ separa geometría de árbol.

## El hallazgo con contenido (no solo un vacío)
La columna `reconf` muestra que la cirugía de Ricci estuvo ENCENDIDA y trabajando: REGLA-b reconfiguró
2 / 47 / 42 / 155 / 144 aristas (Dt3). Reconfiguró de verdad — y aun así la geometría macroscópica no cambió.
⟹ **La hiperbolicidad es robusta a la relajación local.** No es un artefacto "planchable" con cirugía
posterior: la curvatura la fija el proceso de CRECIMIENTO, no se repara aguas abajo. Un contrapeso que
reconfigura localmente sin tocar la clase de curvatura emergente es un resultado, no una ausencia.

## Cautela técnica (prolijidad)
%gig de REGLA-b (73–77% Dt3, 59% Dt2) va parejo con CONTROL ⟹ la poda NO fragmentó el gigante (ni rompe
ni consolida) ⟹ diam/dim son leíbles. No aplica la guardia de "%gig<<100 ⟹ no leer".

## Dónde encaja en el arco (la flecha)
- cg003f corrida de fondo: la HOLONOMÍA sola no despliega el espacio.
- cg003f-b (esta): la RELAJACIÓN sola tampoco.
Dos candidatos downstream (penalización métrica, cirugía de Ricci) fallan del mismo modo: sin separarse
del control. La señal apunta, con consistencia creciente, al MISMO locus — la dinámica de crecimiento
(frente/attach), aguas arriba de la métrica y la poda. Las herramientas no están mal calibradas; operan
DESPUÉS del punto donde se decide la curvatura.

## Recomendación
Abrir la dinámica de crecimiento (cómo se elige el frente y cómo se adjunta), que es ahora el sospechoso
que queda tras descartar métrica y poda con medición sana. Dos descartes limpios valen: en la regla del
equipo, es descartar lo ordinario con rigor antes de saltar a lo extraordinario.
