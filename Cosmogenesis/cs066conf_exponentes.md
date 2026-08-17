# CS066 confirmatorio Nivel 1 — tabla de exponentes (entregable para auditoría de CS)
Malla FIJA k_local×N, 40 parches/celda, N∈{1500,2500,3500,5000}. Motor sin cambios (smoke reprodujo CS066).
CSVs: cs066conf_k{k}_N{N}.csv · analizador: cs066conf_analiza.py (umbrales pre-inscritos).

## Test pre-registrado — exponente de diámetro (CONFIRMA si slope∈[0.29,0.40], R²>0.9, monótona)
| k_local | slope(local) | d=1/slope | R² | monótona | slope(barajado) | Δ | gigante(local) |
|---|---|---|---|---|---|---|---|
| 3 | 0.148 | 6.7 | 0.28 | NO | 0.144 | +0.004 | 0.01 (gas) |
| 4 | 0.576 | 1.7 | 0.96 | sí | 0.125 | +0.451 | 0.40 (roto) |
| 5 | 0.169 | 5.9 | 1.00 | sí | 0.135 | +0.035 | 0.90 |
| 6 | 0.135 | 7.4 | 0.93 | sí | 0.115 | +0.019 | 0.91 |
| 8 | 0.118 | 8.4 | 0.98 | sí | 0.025 | +0.093 | 0.91 |
| 10 | 0.127 | 7.8 | 0.86 | sí | 0.134 | -0.007 | 0.91 |

**Veredicto pre-inscrito: 0/4 celdas fuertes CONFIRMA d≈3.** El diámetro no escala como manifold métrico.

## Firma complementaria — d_s espectral (dial de dimensión, estable con N)
| k_local | d_s N1500→5000 | estable | gigante | clustering@5000 |
|---|---|---|---|---|
| 4 | 1.23→1.45 | sí | 0.40 | 0.60 |
| 5 | 2.57→2.89 | sí | 0.90 | 0.46 |
| 6 | 3.51→3.88 | sí | 0.91 | 0.35 |
| 8 | 5.04→5.29 | sí | 0.91 | 0.22 |
| 10 | 5.29→6.16 | infla | 0.91 | 0.17 |

**d_s pasa por ~3 en k≈5-6, conexo y estable.** Geometría LOCAL real de dimensión finita regulable.

## Síntesis (para que CS firme o ajuste)
Las dos medidas de dimensión DISCREPAN en el régimen sano (k5-6): d_s~3 (local geométrico) vs
diámetro que casi no crece con N (mundo-pequeño residual). El tejido es locALMENTE 3D pero GLOBALMENTE
compacto: la localidad-en-la-formación no mató los atajos de largo alcance. NO es 3-manifold métrico limpio.
El (B) global de CS066 (espacio≠direcciones) NO depende de esto. k=3 es gas (sobre-poda); k=10 ≈ blob.
— CC, 11-jul-2026. Adjudicación pre-registrada mecánica; auditoría del ajuste y firma: CS.
