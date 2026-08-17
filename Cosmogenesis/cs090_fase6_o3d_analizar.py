"""
cs090_fase6_o3d_analizar.py — FASE VI, tarea O3-D: análisis del barrido de `kcap` en Phantom.

Lee `cs090_fase6_o3d_crudo.csv` (una fila por corrida, escrito por
`cs090_fase6_o3d_barrido_kcap.py analizar`) y responde con NÚMEROS — no con veredictos — las tres
preguntas de la tarea:

  1. MONOTONÍA. ¿La fracción de masa acretada varía monótonamente con kcap, como la geometría?
     -> medias/medianas por kcap, Spearman(kcap, masa), Kruskal-Wallis y η² de kcap sobre la masa.

  2. ¿DIRECTO O A TRAVÉS DE LA GEOMETRÍA? Se usa la PENDIENTE CONTINUA como variable, no las clases
     (consigna vigente: el escalón pierde contra la rampa, R²=0.663 vs 0.182).
     -> tres regresiones anidadas (masa~kcap, masa~pendiente, masa~kcap+pendiente), correlaciones
        parciales en ambos sentidos, y una descomposición de mediación a·b (efecto indirecto) vs c'
        (efecto directo) con intervalo de confianza por bootstrap de 10 000 remuestreos.
     -> se agrega el grado medio del grafo final como tercera variable, porque `kcap` LIMITA
        aritméticamente las aristas: cualquier "efecto directo de kcap" podría ser en realidad el
        número de aristas. Se reporta con y sin ese control (no se elige uno: se muestran los dos).

  3. EL EXTREMO kcap=7. ¿La masa acretada cae al nivel de un control sin estructura?
     -> Mann-Whitney y diferencia de medias contra los controles Erdős-Rényi emparejados en aristas,
        y contra los dos controles históricos de `bateria_grafo_random_masa_fija` (N=2000, 4945 aristas).

Eje secundario K: ANOVA de dos vías kcap × grupo_K (K bajo ≤5 / K alto ≥7), diseño balanceado 4×2 con
4 reglas por celda.

Escribe la figura `cs090_fase6_o3d_barrido.png` y el resumen por kcap
`cs090_fase6_o3d_resumen_por_kcap.csv`. No modifica nada. No declara cierre ni veredicto.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import warnings

import numpy as np
from scipy import stats

# Con la batería completa cada grupo de kcap tiene 8 reglas; los avisos de "grados de libertad <= 0"
# sólo aparecen si se corre el análisis a mitad de camino, con algún grupo de una sola regla.
warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = Path("/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis")
RUTA_CRUDO = _HERE / "cs090_fase6_o3d_crudo.csv"
RUTA_RESUMEN = _HERE / "cs090_fase6_o3d_resumen_por_kcap.csv"
RUTA_FIG = _HERE / "cs090_fase6_o3d_barrido.png"
RUTA_SELECCION = _HERE / "cs090_fase6_o3d_seleccion.json"
RUTA_CONTROL_HIST = _HERE / "cs090_fase6_o3d_control_historico.csv"


# =============================== carga ===============================
def cargar():
    filas = list(csv.DictReader(open(RUTA_CRUDO)))
    grupo_K = {s["rule_id"]: s["grupo_K"] for s in json.loads(RUTA_SELECCION.read_text())}
    reglas, controles = [], []
    for f in filas:
        d = dict(f)
        for k in ("fraccion_masa_en_sumideros", "kappa_v_agregado", "kappa_v_medio_valido",
                  "pendiente_corregida", "grado_medio_grafo_final", "masa_sumideros_final",
                  "t_primer_sumidero", "diam_grafo_final", "giant_grafo_final", "z_agg"):
            d[k] = float(d[k]) if d.get(k) not in (None, "", "None") else np.nan
        for k in ("kcap", "K", "n_sumideros", "n_aristas_grafo_final"):
            d[k] = int(float(d[k])) if d.get(k) not in (None, "", "None") else None
        d["grupo_K"] = grupo_K.get(d["rule_id"])
        (controles if d["clase"] == "CONTROL_ER" else reglas).append(d)
    return reglas, controles


# =============================== utilidades estadísticas ===============================
def ols(y, X, nombres):
    """Mínimos cuadrados con intercepto. Devuelve coeficientes, errores estándar, t, p y R²."""
    X1 = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    gl = len(y) - X1.shape[1]
    s2 = float(resid @ resid) / gl
    cov = s2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), gl)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / sst if sst > 0 else np.nan
    r2adj = 1 - (1 - r2) * (len(y) - 1) / gl
    return dict(nombres=["intercepto"] + nombres, beta=beta, se=se, t=t, p=p, r2=r2, r2adj=r2adj,
                resid=resid, gl=gl)


def imprimir_ols(titulo, m):
    print(f"\n  {titulo}   R²={m['r2']:.3f}  R²aj={m['r2adj']:.3f}  gl={m['gl']}")
    for n, b, se, t, p in zip(m["nombres"], m["beta"], m["se"], m["t"], m["p"]):
        print(f"      {n:<26} β={b:+.5f}  ee={se:.5f}  t={t:+.2f}  p={p:.4f}")


def corr_parcial(x, y, z):
    """Correlación de Pearson entre x e y quitando de ambos lo que z explica linealmente."""
    z1 = np.column_stack([np.ones(len(z))] + [z] if np.ndim(z) == 1 else [np.ones(len(z))] + list(z))
    rx = x - z1 @ np.linalg.lstsq(z1, x, rcond=None)[0]
    ry = y - z1 @ np.linalg.lstsq(z1, y, rcond=None)[0]
    r, p = stats.pearsonr(rx, ry)
    return r, p


def eta2(grupos):
    """η² de una vía: fracción de la varianza total explicada por la pertenencia al grupo."""
    todos = np.concatenate(grupos)
    gm = todos.mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in grupos)
    ss_t = float(((todos - gm) ** 2).sum())
    return ss_b / ss_t if ss_t > 0 else np.nan


def anova2(y, a, b):
    """ANOVA de dos vías balanceada (suma de cuadrados tipo I sobre diseño balanceado = tipo III)."""
    gm = y.mean()
    ss_t = float(((y - gm) ** 2).sum())
    niv_a, niv_b = sorted(set(a)), sorted(set(b))
    ss_a = sum(len(y[a == la]) * (y[a == la].mean() - gm) ** 2 for la in niv_a)
    ss_b = sum(len(y[b == lb]) * (y[b == lb].mean() - gm) ** 2 for lb in niv_b)
    ss_ab = 0.0
    for la in niv_a:
        for lb in niv_b:
            sel = (a == la) & (b == lb)
            if sel.sum() == 0:
                continue
            ss_ab += sel.sum() * (y[sel].mean() - y[a == la].mean() - y[b == lb].mean() + gm) ** 2
    # La suma de cuadrados del error se calcula DIRECTAMENTE como la variación dentro de cada celda
    # (no por resta ss_t - ss_a - ss_b - ss_ab): la resta sólo es válida en diseños perfectamente
    # balanceados y da negativos —y F=nan— si alguna celda quedó incompleta.
    ss_e = 0.0
    for la in niv_a:
        for lb in niv_b:
            sel = (a == la) & (b == lb)
            if sel.sum() > 0:
                ss_e += float(((y[sel] - y[sel].mean()) ** 2).sum())
    gl_a, gl_b = len(niv_a) - 1, len(niv_b) - 1
    gl_ab = gl_a * gl_b
    n_celdas = sum(1 for la in niv_a for lb in niv_b if ((a == la) & (b == lb)).sum() > 0)
    gl_e = len(y) - n_celdas
    out = {}
    for nom, ss, gl in (("kcap", ss_a, gl_a), ("grupo_K", ss_b, gl_b), ("interacción", ss_ab, gl_ab)):
        F = (ss / gl) / (ss_e / gl_e) if gl > 0 and ss_e > 0 else np.nan
        out[nom] = dict(ss=ss, gl=gl, F=F, p=float(stats.f.sf(F, gl, gl_e)) if np.isfinite(F) else np.nan,
                        eta2=ss / ss_t)
    out["error"] = dict(ss=ss_e, gl=gl_e)
    return out


def bootstrap_mediacion(kcap, pend, masa, n_boot=10000, semilla=20260811):
    """Efecto indirecto a·b (kcap -> pendiente -> masa) y directo c' con IC percentil por bootstrap
    de reglas completas (se remuestrean filas, no residuos)."""
    rng = np.random.default_rng(semilla)
    n = len(masa)
    ind, dire = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        k, p_, m_ = kcap[i], pend[i], masa[i]
        if np.std(k) < 1e-12 or np.std(p_) < 1e-12:
            continue
        a = np.polyfit(k, p_, 1)[0]
        X = np.column_stack([np.ones(n), k, p_])
        try:
            beta = np.linalg.lstsq(X, m_, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        ind.append(a * beta[2]); dire.append(beta[1])
    return np.array(ind), np.array(dire)


# =============================== análisis ===============================
def main():
    reglas, controles = cargar()
    print("=" * 100)
    print(f"O3-D — barrido de kcap en Phantom: {len(reglas)} reglas A2-B0-C2 + {len(controles)} "
          f"controles Erdős-Rényi emparejados en aristas")
    print("=" * 100)

    kcap = np.array([r["kcap"] for r in reglas], float)
    K = np.array([r["K"] for r in reglas], float)
    masa = np.array([r["fraccion_masa_en_sumideros"] for r in reglas])
    pend = np.array([r["pendiente_corregida"] for r in reglas])
    kv = np.array([r["kappa_v_agregado"] for r in reglas])
    gmed = np.array([r["grado_medio_grafo_final"] for r in reglas])
    grupoK = np.array([r["grupo_K"] for r in reglas])

    # ---------------- 1. monotonía ----------------
    print("\n### 1. ¿Varía la masa acretada monótonamente con kcap?\n")
    print(f"  {'kcap':>5} {'n':>3} {'frac_masa media':>16} {'±ee':>7} {'mediana':>8} "
          f"{'pend media':>11} {'kappaV media':>13} {'grado medio':>12} {'n_sinks':>8}")
    filas_resumen = []
    for k in sorted(set(kcap)):
        s = kcap == k
        fila = dict(kcap=int(k), n=int(s.sum()),
                    frac_masa_media=float(masa[s].mean()), frac_masa_ee=float(masa[s].std(ddof=1) / np.sqrt(s.sum())),
                    frac_masa_mediana=float(np.median(masa[s])),
                    frac_masa_min=float(masa[s].min()), frac_masa_max=float(masa[s].max()),
                    pendiente_media=float(pend[s].mean()), pendiente_ee=float(pend[s].std(ddof=1) / np.sqrt(s.sum())),
                    kappa_v_media=float(np.nanmean(kv[s])), grado_medio=float(gmed[s].mean()),
                    n_sumideros_medio=float(np.mean([r["n_sumideros"] for r in reglas if r["kcap"] == k])))
        filas_resumen.append(fila)
        print(f"  {int(k):>5} {int(s.sum()):>3} {fila['frac_masa_media']:>16.4f} "
              f"{fila['frac_masa_ee']:>7.4f} {fila['frac_masa_mediana']:>8.4f} "
              f"{fila['pendiente_media']:>11.3f} {fila['kappa_v_media']:>13.3f} "
              f"{fila['grado_medio']:>12.2f} {fila['n_sumideros_medio']:>8.2f}")

    if controles:
        cm = np.array([c["fraccion_masa_en_sumideros"] for c in controles])
        ck = np.array([c["kappa_v_agregado"] for c in controles])
        cg = np.array([c["grado_medio_grafo_final"] for c in controles])
        print(f"  {'ER':>5} {len(cm):>3} {cm.mean():>16.4f} "
              f"{cm.std(ddof=1)/np.sqrt(len(cm)) if len(cm)>1 else float('nan'):>7.4f} "
              f"{np.median(cm):>8.4f} {'—':>11} {np.nanmean(ck):>13.3f} {cg.mean():>12.2f}")

    with open(RUTA_RESUMEN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_resumen[0].keys()))
        w.writeheader(); w.writerows(filas_resumen)

    for nombre, v in (("fracción de masa", masa), ("pendiente corregida", pend),
                      ("κ_V agregado", kv), ("grado medio del grafo", gmed)):
        ok = np.isfinite(v)
        rho, p_rho = stats.spearmanr(kcap[ok], v[ok])
        H, p_h = stats.kruskal(*[v[(kcap == k) & ok] for k in sorted(set(kcap))])
        e2 = eta2([v[(kcap == k) & ok] for k in sorted(set(kcap))])
        print(f"\n  {nombre:<24} Spearman(kcap,·) ρ={rho:+.3f} p={p_rho:.5f} | "
              f"Kruskal-Wallis H={H:.2f} p={p_h:.5f} | η²(kcap)={e2:.3f}")

    # ---------------- 2. ¿pasa por la geometría? ----------------
    print("\n\n### 2. ¿El efecto de kcap sobre la masa pasa a través de la geometría (pendiente continua)?\n")
    r_pm, p_pm = stats.pearsonr(pend, masa)
    r_km, p_km = stats.pearsonr(kcap, masa)
    r_kp, p_kp = stats.pearsonr(kcap, pend)
    print(f"  correlaciones simples:  r(pendiente, masa)={r_pm:+.3f} (p={p_pm:.5f})   "
          f"r(kcap, masa)={r_km:+.3f} (p={p_km:.5f})   r(kcap, pendiente)={r_kp:+.3f} (p={p_kp:.5f})")

    m_k = ols(masa, [kcap], ["kcap"])
    m_p = ols(masa, [pend], ["pendiente"])
    m_kp = ols(masa, [kcap, pend], ["kcap", "pendiente"])
    m_kpg = ols(masa, [kcap, pend, gmed], ["kcap", "pendiente", "grado_medio"])
    imprimir_ols("masa ~ kcap", m_k)
    imprimir_ols("masa ~ pendiente", m_p)
    imprimir_ols("masa ~ kcap + pendiente", m_kp)
    imprimir_ols("masa ~ kcap + pendiente + grado_medio (control por aristas)", m_kpg)

    # Diagnóstico de colinealidad: kcap, aristas y pendiente pueden ser casi la misma variable
    # (kcap LIMITA aritméticamente las aristas, y las aristas condicionan la forma). Si lo son,
    # "separar" sus efectos por regresión múltiple es pedirle a los datos algo que no contienen —
    # se reporta el número en vez de callarlo.
    print("\n  colinealidad entre los tres candidatos a causa:")
    for n1, v1, n2, v2 in (("kcap", kcap, "pendiente", pend), ("kcap", kcap, "grado_medio", gmed),
                           ("pendiente", pend, "grado_medio", gmed)):
        rr, pp = stats.pearsonr(v1, v2)
        print(f"      r({n1}, {n2}) = {rr:+.3f} (p={pp:.5f})")
    for nom, v, otras in (("kcap", kcap, [pend, gmed]), ("pendiente", pend, [kcap, gmed]),
                          ("grado_medio", gmed, [kcap, pend])):
        r2_aux = ols(v, otras, ["a", "b"])["r2"]
        print(f"      VIF({nom}) = {1/(1-r2_aux):.1f}   (R² de explicar {nom} con las otras dos = {r2_aux:.3f})")

    r1, p1 = corr_parcial(kcap, masa, pend)
    r2, p2 = corr_parcial(pend, masa, kcap)
    print(f"\n  parcial r(kcap, masa | pendiente) = {r1:+.3f} (p={p1:.4f})")
    print(f"  parcial r(pendiente, masa | kcap) = {r2:+.3f} (p={p2:.4f})")

    a = np.polyfit(kcap, pend, 1)[0]
    b = m_kp["beta"][2]
    c = m_k["beta"][1]
    c_prima = m_kp["beta"][1]
    ind, dire = bootstrap_mediacion(kcap, pend, masa)
    ic_ind = np.percentile(ind, [2.5, 97.5]) if len(ind) else (np.nan, np.nan)
    ic_dir = np.percentile(dire, [2.5, 97.5]) if len(dire) else (np.nan, np.nan)
    prop = (a * b) / c if abs(c) > 1e-12 else np.nan
    print(f"\n  mediación:  a=kcap->pendiente {a:+.4f} | b=pendiente->masa|kcap {b:+.5f} | "
          f"c=total {c:+.5f} | c'=directo {c_prima:+.5f}")
    print(f"    efecto INDIRECTO a·b = {a*b:+.5f}  IC95% bootstrap [{ic_ind[0]:+.5f}, {ic_ind[1]:+.5f}]")
    print(f"    efecto DIRECTO   c'  = {c_prima:+.5f}  IC95% bootstrap [{ic_dir[0]:+.5f}, {ic_dir[1]:+.5f}]")
    print(f"    proporción mediada por la geometría = {prop*100:.1f}% del efecto total")

    # eje secundario K
    print("\n  ANOVA de dos vías (kcap × grupo_K, diseño balanceado 4 por celda) sobre la masa:")
    tab = anova2(masa, kcap, grupoK)
    for nom in ("kcap", "grupo_K", "interacción"):
        d = tab[nom]
        print(f"      {nom:<14} gl={d['gl']}  F={d['F']:.2f}  p={d['p']:.4f}  η²={d['eta2']:.3f}")
    print(f"      error          gl={tab['error']['gl']}")
    for g in sorted(set(grupoK)):
        s = grupoK == g
        print(f"      {g}: masa media={masa[s].mean():.4f} (n={s.sum()}), pendiente media={pend[s].mean():.3f}")

    # ---------------- 3. el extremo kcap=7 vs el control ----------------
    print("\n\n### 3. kcap=7 (0 % de geometría extendida en el grafo) vs control sin estructura\n")
    m7 = masa[kcap == 7]
    m4 = masa[kcap == 4]
    if controles:
        cm = np.array([c["fraccion_masa_en_sumideros"] for c in controles])
        u, pu = stats.mannwhitneyu(m7, cm, alternative="two-sided")
        t7, pt = stats.ttest_ind(m7, cm, equal_var=False)
        print(f"  kcap=7: media={m7.mean():.4f} (n={len(m7)}, rango {m7.min():.4f}-{m7.max():.4f})")
        print(f"  control ER emparejado en aristas: media={cm.mean():.4f} (n={len(cm)}, "
              f"rango {cm.min():.4f}-{cm.max():.4f})")
        print(f"  diferencia kcap7 − control = {m7.mean()-cm.mean():+.4f}  "
              f"Mann-Whitney U={u:.1f} p={pu:.4f} | Welch t={t7:+.2f} p={pt:.4f}")
        u4, pu4 = stats.mannwhitneyu(m4, cm, alternative="two-sided")
        print(f"  (referencia) kcap=4: media={m4.mean():.4f}; diferencia vs control "
              f"{m4.mean()-cm.mean():+.4f}  Mann-Whitney U={u4:.1f} p={pu4:.4f}")
        u47, pu47 = stats.mannwhitneyu(m4, m7, alternative="two-sided")
        print(f"  kcap=4 vs kcap=7: diferencia {m4.mean()-m7.mean():+.4f}  U={u47:.1f} p={pu47:.4f}")

        # controles con aristas emparejadas a cada extremo, por separado (no promediados entre sí)
        for etiqueta, objetivo in (("~2321 aristas (espejo de kcap=4)", 2321),
                                   ("~4608 aristas (espejo de kcap=7)", 4608)):
            sub = np.array([c["fraccion_masa_en_sumideros"] for c in controles
                            if c["n_aristas_grafo_final"] == objetivo])
            if len(sub):
                print(f"    control ER {etiqueta}: media={sub.mean():.4f} n={len(sub)} "
                      f"rango {sub.min():.4f}-{sub.max():.4f}")

    if RUTA_CONTROL_HIST.exists():
        hist = list(csv.DictReader(open(RUTA_CONTROL_HIST)))
        hm = np.array([float(h["fraccion_masa_en_sumideros"]) for h in hist])
        hk = np.array([float(h["kappa_v_agregado"]) for h in hist])
        print(f"\n  controles HISTÓRICOS del proyecto (bateria_grafo_random_masa_fija, N=2000, "
              f"4945 aristas, corridos hace meses con los mismos parámetros): "
              f"frac_masa={[round(float(v),4) for v in hm]} (media {hm.mean():.4f}), "
              f"κ_V={[round(float(v),3) for v in hk]}")

    ruta_pc = _HERE / "cs090_fase6_o3d_pendiente_controles.csv"
    if ruta_pc.exists():
        pc = list(csv.DictReader(open(ruta_pc)))
        print("\n  los controles ER medidos con LA MISMA vara de pendiente (mismo coarse-graining, "
              "mismo diam_gigante):")
        for r in pc:
            print(f"      {r['rule_id']}: aristas={r['n_aristas']} pendiente={float(r['pendiente_corregida']):.3f} "
                  f"frac_masa={float(r['fraccion_masa_en_sumideros']):.4f} "
                  f"κ_V={float(r['kappa_v_agregado']):.3f}")
        # ¿caen sobre la recta masa-vs-pendiente ajustada con las 32 reglas?
        bb = np.polyfit(pend, masa, 1)
        pp = np.array([float(r["pendiente_corregida"]) for r in pc])
        mm = np.array([float(r["fraccion_masa_en_sumideros"]) for r in pc])
        resid_ctrl = mm - np.polyval(bb, pp)
        resid_regl = masa - np.polyval(bb, pend)
        print(f"    residuo de los controles respecto de la recta de las reglas: "
              f"media {resid_ctrl.mean():+.4f}, rango [{resid_ctrl.min():+.4f}, {resid_ctrl.max():+.4f}]")
        print(f"    (para comparar, residuo de las 32 reglas: desvío estándar {resid_regl.std(ddof=1):.4f}, "
              f"rango [{resid_regl.min():+.4f}, {resid_regl.max():+.4f}])")
        print(f"    pendiente media del NULL_topo de las propias reglas (grafo ER de la misma densidad, "
              f"medido dentro del motor) por kcap:")
        for k in sorted(set(kcap)):
            s = kcap == k
            pn = np.array([r["pendiente_null"] for r in reglas], float)[s]
            za = np.array([r["z_agg"] for r in reglas], float)[s]
            print(f"        kcap={int(k)}: pendiente REAL={pend[s].mean():.3f} vs "
                  f"NULL_topo={pn.mean():.3f}  (z_agg medio={za.mean():.2f})")

    figura(reglas, controles)
    print(f"\n[figura] {RUTA_FIG}")
    print(f"[resumen] {RUTA_RESUMEN}")


# =============================== figura ===============================
def figura(reglas, controles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kcap = np.array([r["kcap"] for r in reglas], float)
    masa = np.array([r["fraccion_masa_en_sumideros"] for r in reglas])
    pend = np.array([r["pendiente_corregida"] for r in reglas])
    kv = np.array([r["kappa_v_agregado"] for r in reglas])
    grupoK = np.array([r["grupo_K"] for r in reglas])
    cm = np.array([c["fraccion_masa_en_sumideros"] for c in controles]) if controles else np.array([])

    colores = {4: "#1b6ca8", 5: "#2e9e63", 6: "#e0a020", 7: "#c0392b"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    for k in sorted(set(kcap)):
        s = kcap == k
        ax.scatter(kcap[s] + np.random.default_rng(int(k)).uniform(-.12, .12, s.sum()), masa[s],
                   s=42, color=colores[int(k)], alpha=.8, edgecolor="white", linewidth=.6)
        ax.errorbar(k, masa[s].mean(), yerr=masa[s].std(ddof=1) / np.sqrt(s.sum()), fmt="_",
                    ms=26, mew=2.6, color=colores[int(k)], capsize=5, zorder=5)
    # Los controles NO se dibujan como una banda única: cada uno está emparejado en aristas con UN
    # extremo del barrido (2321 ↔ kcap=4, 4608 ↔ kcap=7), y mezclarlos borraría justo lo que se quiere
    # ver. Van como cruces grises EN la posición del kcap que espejan.
    for objetivo, x_esp in ((2321, 4), (4608, 7)):
        sub = np.array([c["fraccion_masa_en_sumideros"] for c in controles
                        if c["n_aristas_grafo_final"] == objetivo])
        if len(sub):
            ax.errorbar(x_esp + 0.30, sub.mean(), yerr=sub.std(ddof=1) / np.sqrt(len(sub)),
                        fmt="X", ms=13, color="#444", capsize=5, zorder=7,
                        label="control ER sin estructura,\nmismas aristas" if objetivo == 2321 else None)
            ax.scatter(np.full(len(sub), x_esp + 0.30), sub, s=22, color="#888", zorder=6)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("kcap (tope duro de vecinos por nodo)")
    ax.set_ylabel("fracción de masa en sumideros")
    ax.set_title("Pregunta 1 — masa acretada vs kcap (raya = media ± ee)")
    ax.set_xticks(sorted(set(kcap.astype(int))))

    ax = axes[0, 1]
    for k in sorted(set(kcap)):
        s = kcap == k
        ax.scatter(kcap[s] + np.random.default_rng(int(k)).uniform(-.12, .12, s.sum()), pend[s],
                   s=42, color=colores[int(k)], alpha=.8, edgecolor="white", linewidth=.6)
        ax.errorbar(k, pend[s].mean(), yerr=pend[s].std(ddof=1) / np.sqrt(s.sum()), fmt="_",
                    ms=26, mew=2.6, color=colores[int(k)], capsize=5, zorder=5)
    ax.axhline(0.7, color="#888", ls=":", lw=1.2)
    ax.text(6.9, 0.71, "umbral Clase III (0.7)", ha="right", fontsize=8, color="#666")
    ax.set_xlabel("kcap"); ax.set_ylabel("pendiente corregida log(diám)-vs-log(N_cajas)")
    ax.set_title("La geometría medida en el grafo (variable continua)")
    ax.set_xticks(sorted(set(kcap.astype(int))))

    ax = axes[1, 0]
    for k in sorted(set(kcap)):
        s = kcap == k
        marcador = {"K_bajo": "o", "K_alto": "^", "K_medio": "s"}
        for g in sorted(set(grupoK[s])):
            ss = s & (grupoK == g)
            ax.scatter(pend[ss], masa[ss], s=52, color=colores[int(k)], marker=marcador.get(g, "o"),
                       alpha=.85, edgecolor="white", linewidth=.6,
                       label=f"kcap={int(k)} {g}" if g == "K_bajo" else None)
    if len(pend) > 2:
        xs = np.linspace(pend.min(), pend.max(), 50)
        bb = np.polyfit(pend, masa, 1)
        ax.plot(xs, np.polyval(bb, xs), color="#333", lw=1.4, ls="-",
                label=f"ajuste global (r={stats.pearsonr(pend, masa)[0]:+.2f})")
    # Los controles Erdős-Rényi, medidos con LA MISMA vara de pendiente (ver
    # cs090_fase6_o3d_pendiente_controles.py): si caen sobre la recta, la geometría medida no distingue
    # una regla de un grafo al azar de la misma densidad.
    ruta_pc = _HERE / "cs090_fase6_o3d_pendiente_controles.csv"
    if ruta_pc.exists():
        pc = list(csv.DictReader(open(ruta_pc)))
        ax.scatter([float(r["pendiente_corregida"]) for r in pc],
                   [float(r["fraccion_masa_en_sumideros"]) for r in pc],
                   s=110, marker="X", color="#444", edgecolor="white", linewidth=1.1, zorder=6,
                   label="control ER (sin estructura), misma vara")
    ax.set_xlabel("pendiente corregida (geometría)"); ax.set_ylabel("fracción de masa en sumideros")
    ax.set_title("Pregunta 2 — ¿la masa sigue a la geometría? (color=kcap, △=K alto)")
    ax.legend(fontsize=7.5, loc="best")

    ax = axes[1, 1]
    for k in sorted(set(kcap)):
        s = kcap == k
        ax.scatter(kcap[s] + np.random.default_rng(int(k) + 7).uniform(-.12, .12, s.sum()), kv[s],
                   s=42, color=colores[int(k)], alpha=.8, edgecolor="white", linewidth=.6)
        ax.errorbar(k, np.nanmean(kv[s]), yerr=np.nanstd(kv[s], ddof=1) / np.sqrt(s.sum()), fmt="_",
                    ms=26, mew=2.6, color=colores[int(k)], capsize=5, zorder=5)
    for objetivo, x_esp in ((2321, 4), (4608, 7)):
        sub = np.array([c["kappa_v_agregado"] for c in controles
                        if c["n_aristas_grafo_final"] == objetivo])
        if len(sub):
            ax.errorbar(x_esp + 0.30, np.nanmean(sub), yerr=np.nanstd(sub, ddof=1) / np.sqrt(len(sub)),
                        fmt="X", ms=13, color="#444", capsize=5, zorder=7,
                        label="control ER, mismas aristas" if objetivo == 2321 else None)
            ax.scatter(np.full(len(sub), x_esp + 0.30), sub, s=22, color="#888", zorder=6)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("kcap"); ax.set_ylabel("κ_V agregado (último tercio / primer tercio)")
    ax.set_title("κ_V: cómo se reparte la acreción en el tiempo")
    ax.set_xticks(sorted(set(kcap.astype(int))))

    fig.suptitle("O3-D — barrido de kcap directamente en Phantom (A2-B0-C2, N=2000, masa fija 18800)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(RUTA_FIG, dpi=140)


if __name__ == "__main__":
    main()
