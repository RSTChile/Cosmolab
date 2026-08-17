"""
cs090_fase6_o3c_mediacion.py — FASE VI, O3-C: analisis de la cadena
MECANISMO -> GEOMETRIA -> GRAVEDAD sobre el factorial 2x2 corrido en Phantom.
=========================================================================================================

QUE LEE
-------
`cs090_fase6_o3c_crudo.csv`, que escribe `cs090_fase6_o3c_factorial_mecanistico.py --etapa analizar`:
una fila por (regla x condicion), con la GEOMETRIA (`pendiente_corregida`, `clase_geom`) y la GRAVEDAD
(`fraccion_masa_en_sumideros`, `n_sumideros`, `kappa_v_agregado`, `t_primer_sumidero`) de la MISMA celda.

QUE CALCULA (en el orden en que se lee el informe)
---------------------------------------------------
  A. Descriptivos por condicion: media/mediana de pendiente y de fraccion de masa acretada.
  B. Factorial 2x2 sobre las dos variables de salida: efecto principal de RIGIDEZ, efecto principal de
     CRITERIO, e interaccion -- todo calculado como contrastes de medias de celda (con n=12 por celda no
     se ajusta ningun modelo sofisticado, se reportan los contrastes crudos).
  C. Comparaciones PAREADAS entre condiciones (misma regla en las dos condiciones): test de signos
     exacto (binomial) y Wilcoxon de rangos con signo, sobre pendiente y sobre fraccion de masa.
  D. La cadena de mediacion, con tres piezas y ninguna pretension de causalidad estadistica fina:
       a-path  condicion -> pendiente        (¿el mecanismo mueve la geometria?)
       b-path  pendiente -> masa | condicion (¿la geometria mueve la gravedad, a igual condicion?)
       c-path  condicion -> masa             (efecto TOTAL del mecanismo sobre la gravedad)
       c'-path condicion -> masa | pendiente (efecto DIRECTO, lo que queda al controlar la geometria)
     Se reportan: (1) correlacion de Pearson y Spearman condicion~masa ANTES y DESPUES de parcializar
     la pendiente (correlacion parcial por residuos), (2) los coeficientes de las dos regresiones OLS
     (masa~cond y masa~cond+pendiente) con la caida porcentual del coeficiente de la condicion, que es
     la lectura estandar de "cuanto del efecto pasa por el mediador", y (3) un bootstrap del efecto
     indirecto a*b (2000 remuestreos POR REGLA, para respetar el apareamiento) del que se reporta el
     intervalo percentil -- con n=12 reglas es un intervalo ancho por construccion, se reporta la
     DIRECCION, no una prueba.
  E. Version DENTRO DE REGLA (efectos fijos por genealogia): se le resta a cada variable la media de su
     regla y se repite la mediacion sobre los residuos. Esto quita cualquier diferencia entre
     genealogias (una regla que sea "acretiva" en las 4 condiciones no puede inflar nada) y deja
     unicamente la variacion que produjo el cambio de mecanismo.
  F. La prediccion fuerte del encargo, evaluada explicitamente y sin adornos:
       (i)  ¿la condicion 1 (rigido+soporte) domina tambien en gravedad?
       (ii) ¿cuando la geometria desaparece, desaparece el efecto gravitacional? -- se mira, dentro de
            las condiciones 2/3/4 (las que perdieron geometria), si la masa cae junto con la pendiente,
            y se compara la masa de las celdas de pendiente alta contra las de pendiente baja SIN
            importar la condicion (si la geometria es el canal, deberia ordenar mejor que la condicion).

No corre Phantom. No modifica nada. No declara cierre ni veredicto -- imprime numeros y escribe
`cs090_fase6_o3c_mediacion.csv` con la tabla de la cadena.
"""
from __future__ import annotations

import csv
import itertools
import sys

import numpy as np
from scipy import stats

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
RUTA_CRUDO = f"{HERE}/cs090_fase6_o3c_crudo.csv"
RUTA_SALIDA = f"{HERE}/cs090_fase6_o3c_mediacion.csv"

ORDEN_COND = ["c1-rigido-soporte", "c2-rigido-azar", "c3-elastico-soporte", "c4-elastico-azar"]
ETIQUETA = {
    "c1-rigido-soporte": "(1) rigido + soporte",
    "c2-rigido-azar": "(2) rigido + azar",
    "c3-elastico-soporte": "(3) elastico + soporte",
    "c4-elastico-azar": "(4) elastico + azar",
}
Y = "fraccion_masa_en_sumideros"      # la variable de gravedad
M = "pendiente_corregida"             # el mediador (geometria)


# ------------------------------------------------------------------------------------------------
def cargar():
    filas = []
    with open(RUTA_CRUDO) as f:
        for r in csv.DictReader(f):
            filas.append(dict(
                rule_id=r["rule_id"], cond=r["cond_id"], rigidez=r["rigidez"], criterio=r["criterio"],
                clase=r["clase_geom"],
                pendiente=float(r["pendiente_corregida"]),
                masa=float(r["fraccion_masa_en_sumideros"]) if r["fraccion_masa_en_sumideros"] else np.nan,
                n_sumideros=int(float(r["n_sumideros"])) if r["n_sumideros"] else 0,
                masa_sumideros=float(r["masa_sumideros_final"]) if r["masa_sumideros_final"] else np.nan,
                kappa_v=float(r["kappa_v_agregado"]) if r.get("kappa_v_agregado") else np.nan,
                t_primer=float(r["t_primer_sumidero"]) if r.get("t_primer_sumidero") else np.nan,
                diam_b1=float(r["diam_b1_corregido"]),
                grado_medio=float(r["grado_medio_grafo_final"]),
            ))
    # El diseño es PAREADO dentro de regla: una regla que no tenga las 4 condiciones corridas se
    # descarta entera, para que ninguna comparacion entre condiciones se apoye en reglas distintas.
    por_regla = {}
    for f in filas:
        por_regla.setdefault(f["rule_id"], set()).add(f["cond"])
    completas = {r for r, cs in por_regla.items() if len(cs) == len(ORDEN_COND)}
    descartadas = sorted(set(por_regla) - completas)
    if descartadas:
        print(f"[datos] reglas descartadas por no tener las 4 condiciones: {descartadas}")
    return [f for f in filas if f["rule_id"] in completas]


def _arr(filas, campo, cond=None):
    sel = [f for f in filas if cond is None or f["cond"] == cond]
    return np.array([f[campo] for f in sel], dtype=float)


def ols(X, y):
    """Minimos cuadrados con intercepto. X: (n,k) sin columna de unos. Devuelve (coefs, r2)."""
    A = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return beta, r2, resid


def corr_parcial(x, y, z):
    """Correlacion parcial de x con y controlando z: correlacion de los residuos de x~z e y~z."""
    _, _, rx = ols(z.reshape(-1, 1), x)
    _, _, ry = ols(z.reshape(-1, 1), y)
    return stats.pearsonr(rx, ry)


# ================================================================================================
def bloque_A(filas):
    print("\n" + "=" * 108)
    print("A. DESCRIPTIVOS POR CONDICION  (n = reglas por celda)")
    print("=" * 108)
    print(f"{'condicion':<26} {'n':>3} {'pend media':>11} {'pend med':>9} "
          f"{'masa media':>11} {'masa med':>10} {'sumid. medio':>13} {'%III':>6}")
    tabla = {}
    for c in ORDEN_COND:
        sub = [f for f in filas if f["cond"] == c]
        if not sub:
            continue
        pe = np.array([f["pendiente"] for f in sub])
        ma = np.array([f["masa"] for f in sub])
        su = np.array([f["n_sumideros"] for f in sub], dtype=float)
        p3 = 100.0 * sum(1 for f in sub if f["clase"] == "III") / len(sub)
        tabla[c] = dict(n=len(sub), pend_media=pe.mean(), pend_med=np.median(pe),
                        masa_media=np.nanmean(ma), masa_med=np.nanmedian(ma),
                        sumid=su.mean(), pct_III=p3)
        print(f"{ETIQUETA[c]:<26} {len(sub):>3} {pe.mean():>11.4f} {np.median(pe):>9.4f} "
              f"{np.nanmean(ma):>11.5f} {np.nanmedian(ma):>10.5f} {su.mean():>13.2f} {p3:>5.1f}%")
    return tabla


def bloque_B(tabla):
    print("\n" + "=" * 108)
    print("B. FACTORIAL 2x2 — contrastes de medias de celda (rigidez x criterio)")
    print("=" * 108)
    if len(tabla) < 4:
        print("   (faltan celdas, no se calcula)")
        return
    for nombre, clave in (("PENDIENTE (geometria)", "pend_media"), ("MASA ACRETADA (gravedad)", "masa_media")):
        c1, c2 = tabla["c1-rigido-soporte"][clave], tabla["c2-rigido-azar"][clave]
        c3, c4 = tabla["c3-elastico-soporte"][clave], tabla["c4-elastico-azar"][clave]
        ef_rig = ((c1 + c2) / 2) - ((c3 + c4) / 2)
        ef_cri = ((c1 + c3) / 2) - ((c2 + c4) / 2)
        inter = (c1 - c2) - (c3 - c4)
        print(f"\n  {nombre}")
        print(f"                       criterio=soporte   criterio=azar")
        print(f"    corte RIGIDO       {c1:>16.5f}   {c2:>13.5f}")
        print(f"    corte ELASTICO     {c3:>16.5f}   {c4:>13.5f}")
        print(f"    efecto RIGIDEZ (rigido - elastico), promediado sobre criterio : {ef_rig:+.5f}")
        print(f"    efecto CRITERIO (soporte - azar), promediado sobre rigidez    : {ef_cri:+.5f}")
        print(f"    INTERACCION (el criterio solo importa si el corte es rigido)  : {inter:+.5f}")


def bloque_C(filas):
    print("\n" + "=" * 108)
    print("C. COMPARACIONES PAREADAS ENTRE CONDICIONES (misma regla en las dos)")
    print("=" * 108)
    por_regla = {}
    for f in filas:
        por_regla.setdefault(f["rule_id"], {})[f["cond"]] = f
    print(f"{'par':<52} {'var':<10} {'gana A':>7} {'gana B':>7} {'medi.dif':>10} {'signos p':>9} {'wilcox p':>9}")
    filas_out = []
    for a, b in itertools.combinations(ORDEN_COND, 2):
        for var, nombre in ((M, "pendiente"), ("masa", "masa")):
            d = []
            for rid, dd in por_regla.items():
                if a in dd and b in dd:
                    va = dd[a]["pendiente"] if var == M else dd[a]["masa"]
                    vb = dd[b]["pendiente"] if var == M else dd[b]["masa"]
                    if not (np.isnan(va) or np.isnan(vb)):
                        d.append(va - vb)
            d = np.array(d)
            if len(d) < 3:
                continue
            gana_a, gana_b = int((d > 0).sum()), int((d < 0).sum())
            n_ef = gana_a + gana_b
            p_sig = stats.binomtest(gana_a, n_ef, 0.5).pvalue if n_ef else np.nan
            try:
                p_wil = stats.wilcoxon(d).pvalue
            except Exception:
                p_wil = np.nan
            etiqueta = f"{ETIQUETA[a]} vs {ETIQUETA[b]}"
            print(f"{etiqueta:<52} {nombre:<10} {gana_a:>7} {gana_b:>7} {np.median(d):>10.5f} "
                  f"{p_sig:>9.4f} {p_wil:>9.4f}")
            filas_out.append(dict(bloque="C_pareadas", par=f"{a}|{b}", variable=nombre, n=len(d),
                                  gana_A=gana_a, gana_B=gana_b, mediana_dif=float(np.median(d)),
                                  media_dif=float(np.mean(d)), p_signos=float(p_sig),
                                  p_wilcoxon=float(p_wil)))
    return filas_out


def _codigos(filas):
    """Tres codificaciones de la CONDICION, para no depender de una sola:
       cond_1vs_resto : 1 si es (1) rigido+soporte, 0 si no  -- la prediccion fuerte del encargo
       cond_rigidez   : 1 si el corte es rigido, 0 si elastico
       cond_criterio  : 1 si hay criterio de soporte, 0 si azar"""
    x1 = np.array([1.0 if f["cond"] == "c1-rigido-soporte" else 0.0 for f in filas])
    xr = np.array([1.0 if f["rigidez"] == "rigido" else 0.0 for f in filas])
    xc = np.array([1.0 if f["criterio"] == "soporte" else 0.0 for f in filas])
    return dict(cond_1vs_resto=x1, cond_rigidez=xr, cond_criterio=xc)


def mediacion(filas, x, nombre_x, etiqueta_bloque, filas_out):
    m = np.array([f["pendiente"] for f in filas])
    y = np.array([f["masa"] for f in filas])
    ok = ~np.isnan(y)
    x, m, y = x[ok], m[ok], y[ok]
    n = len(y)

    # a-path: condicion -> pendiente
    ba, r2a, _ = ols(x.reshape(-1, 1), m)
    ra = stats.pearsonr(x, m)
    # b-path y c'-path juntos: masa ~ cond + pendiente
    bcp, r2cp, _ = ols(np.column_stack([x, m]), y)
    # c-path: masa ~ cond
    bc, r2c, _ = ols(x.reshape(-1, 1), y)
    rc = stats.pearsonr(x, y)
    rc_s = stats.spearmanr(x, y)
    # correlacion parcial condicion~masa controlando pendiente
    rp = corr_parcial(x, y, m)
    # correlacion pendiente~masa (b bruta) y parcial controlando condicion
    rm = stats.pearsonr(m, y)
    rm_s = stats.spearmanr(m, y)
    rmp = corr_parcial(m, y, x)

    a, b_, c, cp = ba[1], bcp[2], bc[1], bcp[1]
    indirecto = a * b_
    prop = indirecto / c if abs(c) > 1e-12 else np.nan

    # bootstrap del efecto indirecto, remuestreando REGLAS enteras (respeta el apareamiento)
    reglas = sorted(set(f["rule_id"] for f in filas))
    idx_por_regla = {r: [i for i, f in enumerate(filas) if f["rule_id"] == r] for r in reglas}
    rng = np.random.default_rng(20260811)
    boot = []
    mm = np.array([f["pendiente"] for f in filas]); yy = np.array([f["masa"] for f in filas])
    xx = _codigos(filas)[nombre_x]
    for _ in range(2000):
        sel = []
        for r in rng.choice(reglas, size=len(reglas), replace=True):
            sel.extend(idx_por_regla[r])
        sel = np.array(sel)
        xs, ms_, ys = xx[sel], mm[sel], yy[sel]
        k = ~np.isnan(ys)
        xs, ms_, ys = xs[k], ms_[k], ys[k]
        if len(np.unique(xs)) < 2 or len(ys) < 6:
            continue
        try:
            bb_a, _, _ = ols(xs.reshape(-1, 1), ms_)
            bb_cp, _, _ = ols(np.column_stack([xs, ms_]), ys)
            boot.append(bb_a[1] * bb_cp[2])
        except Exception:
            continue
    boot = np.array(boot)
    ic = (np.percentile(boot, 2.5), np.percentile(boot, 97.5)) if len(boot) > 100 else (np.nan, np.nan)

    print(f"\n  --- {etiqueta_bloque}: X = {nombre_x} (n={n}) ---")
    print(f"    a-path  X -> pendiente          : a = {a:+.5f}   r={ra[0]:+.3f} (p={ra[1]:.4g})")
    print(f"    b-path  pendiente -> masa | X   : b = {b_:+.6f}  r_parcial={rmp[0]:+.3f} (p={rmp[1]:.4g})")
    print(f"            (pendiente~masa bruta)  : r={rm[0]:+.3f} (p={rm[1]:.4g})  "
          f"Spearman rho={rm_s.statistic:+.3f} (p={rm_s.pvalue:.4g})")
    print(f"    c-path  X -> masa  (TOTAL)      : c = {c:+.6f}  r={rc[0]:+.3f} (p={rc[1]:.4g})  "
          f"Spearman rho={rc_s.statistic:+.3f} (p={rc_s.pvalue:.4g})")
    print(f"    c'-path X -> masa | pendiente   : c'= {cp:+.6f}  r_parcial={rp[0]:+.3f} (p={rp[1]:.4g})")
    print(f"    efecto indirecto a*b = {indirecto:+.6f}   IC95% bootstrap-por-regla "
          f"[{ic[0]:+.6f}, {ic[1]:+.6f}]  (2000 remuestreos)")
    print(f"    proporcion mediada a*b/c = {prop:+.3f}   caida del coeficiente de X al meter la "
          f"pendiente: {100.0*(1-cp/c) if abs(c)>1e-12 else float('nan'):+.1f}%")

    filas_out.append(dict(bloque=etiqueta_bloque, par=nombre_x, variable="mediacion", n=n,
                          a=a, b=b_, c=c, c_prima=cp, indirecto=indirecto,
                          ic95_bajo=ic[0], ic95_alto=ic[1], proporcion_mediada=prop,
                          r_cond_masa=rc[0], p_cond_masa=rc[1],
                          r_cond_masa_parcial=rp[0], p_cond_masa_parcial=rp[1],
                          r_pend_masa=rm[0], p_pend_masa=rm[1],
                          r_pend_masa_parcial=rmp[0], p_pend_masa_parcial=rmp[1]))


def bloque_D(filas, filas_out):
    print("\n" + "=" * 108)
    print("D. CADENA DE MEDIACION — condicion -> pendiente (geometria) -> masa acretada (gravedad)")
    print("=" * 108)
    cods = _codigos(filas)
    for nombre in ("cond_1vs_resto", "cond_rigidez", "cond_criterio"):
        mediacion(filas, cods[nombre], nombre, "D_global", filas_out)


def bloque_E(filas, filas_out):
    print("\n" + "=" * 108)
    print("E. LA MISMA CADENA, DENTRO DE CADA REGLA (se le resta a cada variable la media de su regla)")
    print("=" * 108)
    print("   Quita cualquier diferencia entre genealogias: solo queda lo que movio el cambio de mecanismo.")
    por_regla = {}
    for f in filas:
        por_regla.setdefault(f["rule_id"], []).append(f)
    centradas = []
    for rid, sub in por_regla.items():
        pe = np.array([f["pendiente"] for f in sub])
        ma = np.array([f["masa"] for f in sub])
        for f, dp, dm in zip(sub, pe - np.nanmean(pe), ma - np.nanmean(ma)):
            g = dict(f); g["pendiente"] = dp; g["masa"] = dm
            centradas.append(g)
    cods = _codigos(centradas)
    for nombre in ("cond_1vs_resto", "cond_rigidez", "cond_criterio"):
        mediacion(centradas, cods[nombre], nombre, "E_dentro_de_regla", filas_out)


def bloque_F(filas, tabla, filas_out):
    print("\n" + "=" * 108)
    print("F. LA PREDICCION FUERTE DEL ENCARGO, EVALUADA")
    print("=" * 108)

    orden_pend = sorted(tabla.items(), key=lambda kv: -kv[1]["pend_media"])
    orden_masa = sorted(tabla.items(), key=lambda kv: -kv[1]["masa_media"])
    print("  (i) ¿la condicion 1 domina tambien en gravedad?")
    print(f"      ranking por GEOMETRIA (pendiente media): "
          f"{' > '.join(ETIQUETA[c] for c, _ in orden_pend)}")
    print(f"      ranking por GRAVEDAD  (masa media)     : "
          f"{' > '.join(ETIQUETA[c] for c, _ in orden_masa)}")
    print(f"      condicion 1 es la #1 en geometria: {orden_pend[0][0] == 'c1-rigido-soporte'}   "
          f"y en gravedad: {orden_masa[0][0] == 'c1-rigido-soporte'}")

    print("\n  (ii) ¿cuando la geometria desaparece, desaparece el efecto gravitacional?")
    pend = np.array([f["pendiente"] for f in filas])
    masa = np.array([f["masa"] for f in filas])
    ok = ~np.isnan(masa)
    corte = np.median(pend[ok])
    alto, bajo = masa[ok][pend[ok] > corte], masa[ok][pend[ok] <= corte]
    u = stats.mannwhitneyu(alto, bajo, alternative="two-sided")
    print(f"      celdas de pendiente ALTA (>{corte:.3f}, n={len(alto)}): masa media={alto.mean():.5f} "
          f"mediana={np.median(alto):.5f}")
    print(f"      celdas de pendiente BAJA (<={corte:.3f}, n={len(bajo)}): masa media={bajo.mean():.5f} "
          f"mediana={np.median(bajo):.5f}")
    print(f"      Mann-Whitney U p={u.pvalue:.4g}   (partiendo por la MEDIANA DE LA PENDIENTE, "
          f"ignorando de que condicion viene cada celda)")

    # el mismo corte pero por condicion 1 vs resto, para comparar cual ordena mejor
    es1 = np.array([f["cond"] == "c1-rigido-soporte" for f in filas])[ok]
    u2 = stats.mannwhitneyu(masa[ok][es1], masa[ok][~es1], alternative="two-sided")
    print(f"      comparacion: partir por CONDICION 1 vs resto -> masa media {masa[ok][es1].mean():.5f} "
          f"vs {masa[ok][~es1].mean():.5f}, Mann-Whitney p={u2.pvalue:.4g}")

    print("\n      dentro de las condiciones 2/3/4 (las que PERDIERON geometria):")
    sub = [f for f in filas if f["cond"] != "c1-rigido-soporte" and not np.isnan(f["masa"])]
    if len(sub) > 5:
        ps = np.array([f["pendiente"] for f in sub]); ms_ = np.array([f["masa"] for f in sub])
        rs = stats.spearmanr(ps, ms_)
        print(f"        Spearman pendiente~masa dentro de 2/3/4: rho={rs.statistic:+.3f} "
              f"(p={rs.pvalue:.4g}, n={len(sub)})")
        filas_out.append(dict(bloque="F", par="c234", variable="spearman_pend_masa", n=len(sub),
                              r_pend_masa=float(rs.statistic), p_pend_masa=float(rs.pvalue)))

    filas_out.append(dict(bloque="F", par="corte_mediana_pendiente", variable="masa", n=int(ok.sum()),
                          media_dif=float(alto.mean() - bajo.mean()), p_wilcoxon=float(u.pvalue)))
    filas_out.append(dict(bloque="F", par="cond1_vs_resto", variable="masa", n=int(ok.sum()),
                          media_dif=float(masa[ok][es1].mean() - masa[ok][~es1].mean()),
                          p_wilcoxon=float(u2.pvalue)))


def main():
    filas = cargar()
    print(f"[datos] {len(filas)} celdas (regla x condicion) desde {RUTA_CRUDO}")
    print(f"        reglas distintas: {len(set(f['rule_id'] for f in filas))}   "
          f"condiciones: {sorted(set(f['cond'] for f in filas))}")
    sin_masa = [f for f in filas if np.isnan(f["masa"])]
    if sin_masa:
        print(f"        AVISO: {len(sin_masa)} celdas sin fraccion de masa (¿dump faltante?): "
              f"{[f['rule_id']+'/'+f['cond'] for f in sin_masa]}")

    filas_out = []
    tabla = bloque_A(filas)
    bloque_B(tabla)
    filas_out += bloque_C(filas) or []
    bloque_D(filas, filas_out)
    bloque_E(filas, filas_out)
    bloque_F(filas, tabla, filas_out)

    campos = []
    for f in filas_out:
        for c in f:
            if c not in campos:
                campos.append(c)
    with open(RUTA_SALIDA, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas_out)
    print(f"\n[CSV] {len(filas_out)} filas -> {RUTA_SALIDA}")
    print("Sin cierre ni veredicto: solo numeros. La lectura final es de Alexis.")


if __name__ == "__main__":
    sys.exit(main())
