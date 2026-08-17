# Manifest — `images/` (export Meta completo)

**Ruta canónica de binarios:** `images/`  
**Captura:** 2026-07-21 (copia del usuario)  
**Totales:** 98 archivos · **58 PNG** · **29 JSON** · **9 PY** · **1 CSV** · **1 TXT**

Los 17 PNG de `assets/` (share) son un subconjunto; **preferir `images/`** de aquí en adelante.

---

## Por fase / tema

### A. Pregunta fundacional / viabilidad / ε

| Archivo | Notas |
|---------|--------|
| `curva_epsilon.png` | Curva ε |
| `cs074_eps_independencia.png` | Independencia en ε |
| `cs074_P_vs_T.png` | Persistencia vs T |
| `cs074_P_vs_t (1).png` | Persistencia vs t |
| `cs074_P_vs_H_cruce.png` | Cruce P vs H |
| `cs074_cruce_primaria_v6.png` | Cruce primaria v6 |
| `cs074_zoom_viabilidad.png` | Zoom viabilidad |
| `cs074_H_vs_T_viabilidad.png` | H vs T |
| `cs074_estabilidad_5000.png` | Estabilidad 5000 pasos |
| `descarga-CSV-del-barrido-completo.csv` | Barrido ε: columnas `epsilon,n_distintos,std_contraste,max_dens` (12 filas + header) |

### B. Clusters k, null, z, 1D/2D

| Archivo |
|---------|
| `cs074_k3_privilegiado_plot.png` |
| `cs074_k3_ratio.png` |
| `cs074_real_vs_null_k3.png` |
| `cs074_z_k3.png` |
| `cs074_grad_test.png` |
| `cs074_fino_z_vs_H.png` |
| `cs074_fino_z_vs_T.png` |
| `cs074_fino_H.png` |
| `cs074_fino_fraccion.png` |
| `cs074_k_conteo_vs_T.png` |
| `cs074_k_fraccion_vs_T.png` |
| `cs074_barrido_final_k3.png` |
| `cs074_barrido_final_total.png` |
| `cs074_2D_vs_1D.png` |
| `cs074_barrido_2D_Hc_Tc.png` |
| `cs074_fase_k3_dist.png` |

### C. Siembra / capacidad / emergencia

| Archivo |
|---------|
| `cs074_10entes_surv.png` |
| `cs074_10entes_vs_H.png` |
| `cs074_capacidad.png` |
| `cs074_capacidad_ratio.png` |
| `cs074_1000_surv.png` |
| `cs074_1000_ratio.png` |
| `cs074_comparacion_1000_vs_emerg2.png` |
| `cs074_emergencia_k3_k10_z.png` |
| `cs074_emergencia_k3_k10_real.png` |
| `cs074_emerg_total_5000.png` |
| `cs074_emerg_kdist_5000.png` |

### D. r_crit / N grande

| Archivo |
|---------|
| `cs074_rcrit_vs_k.png` |
| `cs074_rcrit_N20000_k3.png` |
| `cs074_rcrit_N20000_plot.png` |

### E. Fase 2 — carga, tensión, lógos, Z₃

| Archivo |
|---------|
| `cs074_paso1_rep2.png` |
| `cs074_k1_fase_repulsion2.png` |
| `cs074_tension_k3.png` |
| `cs074_paso3_logos.png` |
| `cs074_paso4_2D_Z3.png` |

### F. Fase 3 — f(H), J_c, K_c, λ/α

| Archivo | Tema |
|---------|------|
| `cs074_fase3A_fH.png` (+ `(1)`) | 3A masa / f(H) |
| `cs074_fase3B_Jc.png` (+ `(1)`) | 3B color / J_c |
| `cs074_fase3C_Kc.png` (+ `(1)`) | 3C K_c |
| `cs074_fase3C_fino_Kc_corr.png` | Fino K_c correlación |
| `cs074_fase3C_lam_alpha.png` (+ `(1)`) | λ / α |
| `cs074_fino_Kc_F.png` | K_c y F |

### G. Masas / jerarquía (Fase 4 / tanda)

| Archivo |
|---------|
| `cs074_jerarquia_masas_perim.png` |
| `cs074_50x50_masas.png` |
| `cs074_100x100_masas.png` (+ `(1)`) |

---

## JSON (29)

Incluye entre otros:  
`cs074_barrido_eps.json`, `cs074_barrido_fino_z.json`, `cs074_barrido_stab_5000.json`, `cs074_barrido_final_N20000.json`, `cs074_test_decisivo_null_grad.json`, `cs074_k3_privilegiado.json`, `cs074_10entes_barrido.json`, `cs074_capacidad_carga.json`, `cs074_1000entes.json`, `cs074_emergencia_*.json`, `cs074_rcrit_N20000_z.json`, `cs074_v6_resultado.json`, `cs074_v6_5000pasos.json`, `cs074_limites_barrido.json`, `Límites.json`, `JSON-fino.json`, etc.

Algunos nombres Meta traen tokens raros (`barrido-{{IE_50}}$N=20000$…`); son el mismo tipo de payload.

## Código / texto Meta (9+1)

| Archivo |
|---------|
| `Búsqueda-de-novedad-unificada.py` y variantes `(1)`…`(8)` |
| `Búsqueda-de-novedad-unificada.txt` |

---

## Duplicados `(1)`

Varias figuras tienen copia `(1)` — misma familia de plot (re-export Meta). En la galería se usa la versión **sin** sufijo salvo que difieran.

---

## Relación con `assets/`

| | `assets/` | `images/` |
|--|-----------|-----------|
| Origen | Share HTML | Export manual completo |
| PNG | 17 | **58** |
| JSON | 0 | **29** |
| **Usar como canónico** | no | **sí** |
