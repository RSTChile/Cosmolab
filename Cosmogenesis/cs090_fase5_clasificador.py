"""
CS090 — FASE V-A (piloto): CLASIFICADOR (aplica los umbrales YA FIJADOS en la especificación, §4)
====================================================================================================
QUIÉN SOY: toma las filas que produce cs090_fase5_motor.correr_regla() (una fila por tamaño N con
diám/giant/holonomía REAL + NULL_topo + NULL_valor emparejados) y decide en qué Clase I-IV cae la
regla, usando EXACTAMENTE los umbrales pre-registrados en
FASE5_especificacion_universalidad_CS.md §4 -- no se inventa ningún umbral nuevo acá:

  Clase I  — Disolución:              z<3 vs NULL, sin estructura persistente.
  Clase II — Mundo-pequeño congelado:  pendiente log(diám)-vs-log(N) entre ~0.35-0.45.
  Clase III— Geometría extensa:        pendiente sostenida >0.7-0.8, O separación z>3 sostenida vs NULL.
  Clase IV — Retroalimentación cerrada: además de III, holonomía ≥5x menor que NULL.

Se calcula, por regla: (a) pendiente REAL de log(diám) vs log(N) por regresión lineal sobre las 4
tallas del piloto (500/1000/1500/2000); (b) z-score REAL-vs-NULL_topo en CADA talla (z_N =
(diam_real-diam_null)/std_null) y si está "sostenido" (>3 en las 4 tallas, no sólo en promedio);
(c) razón de holonomía NULL_valor/REAL (¿≥5?).

Casos límite (pendiente entre 0.45 y 0.7, ni II ni III con los umbrales dados) se documentan como
"intermedio -- sin clase clara" en vez de forzarlos a una de las 4 cajas -- disciplina anti-Shannon del
proyecto (no se fabrica un veredicto que el propio umbral no sostiene).

No declara cierre ni veredicto sobre S>0 -- sólo aplica la regla de clasificación y reporta números.
"""
from __future__ import annotations
import numpy as np


def _pendiente_loglog(Ns, ys):
    """Pendiente de la regresión y=a*log(N)+b en log-log. Filtra NaN/valores no positivos."""
    Ns = np.asarray(Ns, dtype=float); ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ys) & (ys > 0) & (Ns > 0)
    if mask.sum() < 2:
        return float("nan")
    logN, logY = np.log(Ns[mask]), np.log(ys[mask])
    pendiente, _ = np.polyfit(logN, logY, 1)
    return float(pendiente)


def clasificar_regla(filas):
    """filas = lista de dicts de cs090_fase5_motor.correr_regla() para UNA regla (una fila por N).
    Devuelve dict con las métricas agregadas + la clase asignada + el detalle de qué umbral decidió."""
    Ns = [f["N"] for f in filas]
    diam_real = [f["diam_real"] for f in filas]
    diam_null = [f["diam_null_topo"] for f in filas]
    diam_null_std = [f["diam_null_topo_std"] for f in filas]
    holon_real = [f["holon_real"] for f in filas]
    holon_null = [f["holon_null_valor"] for f in filas]

    pendiente_real = _pendiente_loglog(Ns, diam_real)
    pendiente_null = _pendiente_loglog(Ns, diam_null)

    z_por_N = []
    for dr, dn, dstd in zip(diam_real, diam_null, diam_null_std):
        z_por_N.append(abs(dr - dn) / dstd if dstd > 1e-9 else 0.0)
    z_agg = float(np.mean(z_por_N))
    z_sostenido = all(z > 3.0 for z in z_por_N)

    holon_ratio_por_N = [ (hn / hr) if hr > 1e-9 else float("nan") for hn, hr in zip(holon_null, holon_real) ]
    holon_ratio = float(np.nanmean(holon_ratio_por_N))
    holon_ge5 = holon_ratio >= 5.0 if not np.isnan(holon_ratio) else False

    # ---------------- decisión, EN ORDEN, con los umbrales del §4 (no inventados acá) ----------------
    detalle = (f"pendiente_real={pendiente_real:.3f} pendiente_null={pendiente_null:.3f} "
               f"z_agg={z_agg:.2f} z_por_N={[round(z,1) for z in z_por_N]} z_sostenido={z_sostenido} "
               f"holon_ratio(null/real)={holon_ratio:.2f} holon_ge5={holon_ge5}")

    if z_agg < 3.0 and not z_sostenido and not (0.35 <= pendiente_real <= 0.45) and pendiente_real <= 0.7:
        clase = "I"
        motivo = "z<3 vs NULL, sin separación estructural sostenida -- indistinguible de ruido"
    elif 0.35 <= pendiente_real <= 0.45:
        clase = "II"
        motivo = f"pendiente log(diam)-vs-log(N)={pendiente_real:.3f} en rango mundo-pequeño 0.35-0.45"
    elif pendiente_real > 0.7 or z_sostenido:
        if holon_ge5:
            clase = "IV"
            motivo = (f"pendiente={pendiente_real:.3f}>0.7 (o z sostenido) Y holonomía NULL/REAL="
                       f"{holon_ratio:.2f}>=5 -- retroalimentación cerrada")
        else:
            clase = "III"
            motivo = f"pendiente={pendiente_real:.3f}>0.7 o z sostenido={z_sostenido} -- geometría extensa, sin cierre holonómico ≥5x"
    else:
        clase = "intermedio (sin clase clara)"
        motivo = (f"pendiente={pendiente_real:.3f} fuera de los rangos pre-registrados (ni ~0.35-0.45 "
                   f"ni >0.7) y z_agg={z_agg:.2f} no llega sostenido>3 -- NO se fuerza a I/II/III, se reporta tal cual")

    return dict(
        rule_id=filas[0]["rule_id"], clase=clase, motivo=motivo, detalle=detalle,
        pendiente_real=pendiente_real, pendiente_null=pendiente_null,
        z_agg=z_agg, z_sostenido=z_sostenido, holon_ratio=holon_ratio, holon_ge5=holon_ge5,
    )


if __name__ == "__main__":
    import cs090_fase5_generador as GEN
    import cs090_fase5_motor as MOT
    print("CS090 — clasificador: smoke test con N chico sobre las 5 combinaciones del piloto")
    combos = [("A0", "B0", "C0"), ("A1", "B0", "C0"), ("A2", "B1", "C1"), ("A1", "B1", "C0"), ("A2", "B0", "C2")]
    for (a, b, c) in combos:
        admitidas, _ = GEN.generar_reglas_clase(a, b, c, n_reglas=1, seed_base=7)
        p = admitidas[0]
        filas = MOT.correr_regla(p, Ns=(60, 120, 200, 300), n_sweeps=8, n_seeds_null_topo=2)
        r = clasificar_regla(filas)
        print(f"  {r['rule_id']}: clase={r['clase']}  ({r['motivo']})")
