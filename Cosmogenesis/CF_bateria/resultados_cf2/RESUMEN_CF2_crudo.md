# CF-2 resultado crudo

**Protocolo:** `CF2_ESTIRAMIENTO_DENSIDAD_2026-07-23`  
**Pre-registro:** `../PROTOCOLO_CF2_estiramiento_densidad_PREREGISTRO.md` (antes de correr)

**Veredicto automático:** `CF2_PASS`

## Lectura cruda (sin adjudicación cosmológica)

- REAL: A_phys_ratio cae monótonamente al subir H_EXP (0.16 → 0.0003).  
- NULL_A_FIXED: A_phys_ratio≈0.26 (no stretch; rate_null_a_stretch=0).  
- NULL_RHO_FIXED: también cae A_phys (estiramiento geométrico) pero A_comov se separa de REAL (rho_sep=1.00 en todo H).  
- Variación entre semillas ≈ 0 (frente tanh casi determinista + ruido 1e-4). El multi-seed cumple T7; la física del sello es muy estable.

| H_EXP | a_final | rate_stretch | rate_rho_sep | rate_mono | pass_H |
|-------|---------|--------------|--------------|-----------|--------|
| 1.0 | 2.7 | 1.00 | 1.00 | 1.00 | True |
| 2.0 | 7.4 | 1.00 | 1.00 | 1.00 | True |
| 3.0 | 20.1 | 1.00 | 1.00 | 1.00 | True |
| 4.0 | 54.6 | 1.00 | 1.00 | 1.00 | True |
| 5.0 | 148.4 | 1.00 | 1.00 | 1.00 | True |
| 6.0 | 403.4 | 1.00 | 1.00 | 1.00 | True |
| 7.0 | 1096.6 | 1.00 | 1.00 | 1.00 | True |
| 8.0 | 2981.0 | 1.00 | 1.00 | 1.00 | True |

n_pass_H=8 (min 5); incluye a>50: True; a<20: True; null_a_ok: True
