# Manifest de attachments nombrados en el PDF

Lista de archivos `cs074_*` mencionados en el texto de la sesión (Meta `/mnt/data/…`).  
**Estado local:** solo los de la columna “En assets/” están en este repo.

## En `assets/` (17 — del share)

Ver [MANIFEST_ASSETS.md](MANIFEST_ASSETS.md).

## Nombrados en PDF — PNG no locales

| Archivo | Tema |
|---------|------|
| `cs074_P_vs_T.png` | Persistencia vs T |
| `cs074_P_vs_t.png` | Persistencia vs tiempo |
| `cs074_cruce_primaria_v6.png` | Cruce primaria v6 |
| `cs074_P_vs_H_cruce.png` | P vs H cruce |
| `cs074_zoom_viabilidad.png` | Zoom viabilidad |
| `cs074_H_vs_T_viabilidad.png` | H vs T viabilidad |
| `cs074_fino_fraccion.png` | Barrido fino fracción |
| `cs074_fino_H.png` | Barrido fino H |
| `cs074_rcrit_vs_k.png` | r_crit vs k |
| `cs074_rcrit_N20000_k3.png` | r_crit N=20000 k3 |
| `cs074_paso4_torsion_fast.png` | Torsión paso 4 |

## Nombrados en PDF — JSON / datos / código

| Archivo | Tema |
|---------|------|
| `cs074_persistencia_campo.py` | Instrumento (también en Cosmogenesis Mac) |
| `cs074_ARCHIVO_version_discreta_INCORRECTA.py` | Versión archivada incorrecta |
| `cs074_produccion_resultado_crudo.json` | Producción 64 filas |
| `cs074_produccion_meta.txt` | time/RAM |
| `cs074_v6_resultado.json` | v6 |
| `cs074_v6_5000pasos.json` | v6 largo |
| `cs074_limites_barrido.json` | Límites |
| `cs074_k_vs_T_sim.json` | k vs T |
| `cs074_barrido_eps.json` | Barrido ε |
| `cs074_barrido_stab_5000.json` | Estabilidad 5000 |
| `cs074_k3_privilegiado.json` | k3 privilegiado |
| `cs074_k3_5000.json` | k3 5000 |
| `cs074_test_decisivo_null_grad.json` | Null + gradiente |
| `cs074_barrido_fino_z.json` | Fino z |
| `cs074_10entes_barrido.json` | 10 entes |
| `cs074_capacidad_carga.json` | Capacidad |
| `cs074_emergencia_k4_k10.json` | Emergencia k |
| `cs074_1000entes.json` | 1000 entes |
| `cs074_emergencia_5000.json` | Emergencia 5000 |
| `cs074_emerg_vs_eps_10000.json` | Emerg vs ε |
| `cs074_N20000_H0005.json` | N=20000 |
| `cs074_barrido_final_N20000.json` | Barrido final N |
| `cs074_rcrit_N20000_z.json` | r_crit z |
| `cs074_logos_buffer.json` | Lógos buffer |

## Cómo recuperar faltantes

1. Desde el hilo Meta: descargar cada attachment y copiar a `assets/` o `data/`.  
2. Desde el Mac (si la corrida local existe):  
   `Cosmogenesis/cs074_produccion_resultado_crudo.json`  
   `Cosmogenesis/cs074_persistencia_campo.py`  
3. Re-export PDF **no** trae binarios de imagen (0 images en pdfimages).
