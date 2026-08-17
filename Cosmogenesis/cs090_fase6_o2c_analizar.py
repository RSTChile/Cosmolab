"""
CS090 — FASE VI / O2-C: ANÁLISIS de las tres pruebas de capacidad finita
=====================================================================================================
QUIÉN SOY: lee los 3 CSV crudos que produce `cs090_fase6_o2c_capacidad_finita.py` y responde, con
números, las tres preguntas de la tarea. El observable es SIEMPRE la pendiente continua corregida
(log diámetro-gigante vs log N_cajas del coarse-graining); las clases I-IV se cuentan sólo como dato
secundario, por pedido explícito del equipo (el escalón pierde contra la rampa: R²=0.663 vs 0.182,
`FASE6_reanalisis_azar_continuo_CS.md`).

Todas las comparaciones entre condiciones son PAREADAS por regla (mismo `rule_id`, mismo N): cada
regla es su propio control, así la varianza entre reglas (que es grande) no ensucia la comparación.
Se usa Wilcoxon de rangos con signo (no paramétrico, n chico) y además el conteo crudo de signos, que
es el resumen más difícil de sobre-interpretar.

No declara cierre ni veredicto — imprime números y arma un PNG.
"""
from __future__ import annotations
import csv, sys
import numpy as np
from collections import defaultdict

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from scipy import stats as SP
except Exception:
    SP = None


def cargar(ruta):
    try:
        with open(ruta) as fh:
            filas = list(csv.DictReader(fh))
    except FileNotFoundError:
        print(f"(falta {ruta})")
        return []
    for f in filas:
        for k in ("pendiente", "pendiente_null", "z_agg", "kcap_media_emp", "kcap_cv", "grado_medio_b1",
                  "diam_b1", "giant_b1", "holon_ratio", "dt", "kcap_sd", "grado_inicial_medio"):
            if k in f:
                try:
                    f[k] = float(f[k])
                except (TypeError, ValueError):
                    f[k] = float("nan")
        for k in ("N_sistema", "kcap_base", "kcap_min", "kcap_max", "n_aristas_b1", "seed"):
            if k in f:
                f[k] = int(float(f[k]))
    return filas


def _agrupar(filas, *claves):
    d = defaultdict(list)
    for f in filas:
        d[tuple(f[k] for k in claves)].append(f)
    return d


def _res(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return dict(n=0, media=float("nan"), mediana=float("nan"), sd=float("nan"), ee=float("nan"))
    return dict(n=len(v), media=float(v.mean()), mediana=float(np.median(v)), sd=float(v.std(ddof=1) if len(v) > 1 else 0.0),
                ee=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def pareado(filas_a, filas_b, clave=("rule_id", "N_sistema")):
    """Diferencia pareada A-B de la pendiente, emparejando por (regla, N)."""
    ia = {tuple(f[k] for k in clave): f["pendiente"] for f in filas_a}
    ib = {tuple(f[k] for k in clave): f["pendiente"] for f in filas_b}
    comunes = sorted(set(ia) & set(ib), key=str)
    d = np.array([ia[c] - ib[c] for c in comunes if np.isfinite(ia[c]) and np.isfinite(ib[c])])
    if len(d) == 0:
        return None
    p = float("nan")
    if SP is not None and len(d) >= 5 and np.any(d != 0):
        try:
            p = float(SP.wilcoxon(d).pvalue)
        except Exception:
            p = float("nan")
    return dict(n=len(d), media=float(d.mean()), mediana=float(np.median(d)),
                gana_A=int((d > 0).sum()), gana_B=int((d < 0).sum()), empate=int((d == 0).sum()), p=p)


def pendiente_de_pendiente(filas, clave_x="N_sistema"):
    """Cuánto cambia la pendiente al cambiar el tamaño: regresión de `pendiente` contra log(N).
    Es "la deriva con la escala": si vale ~0, la familia de geometrías se conserva al crecer N."""
    xs, ys = [], []
    for f in filas:
        if np.isfinite(f["pendiente"]) and f[clave_x] > 0:
            xs.append(np.log(f[clave_x])); ys.append(f["pendiente"])
    if len(set(xs)) < 2:
        return dict(b=float("nan"), r=float("nan"), n=len(xs))
    b, a = np.polyfit(xs, ys, 1)
    r = float(np.corrcoef(xs, ys)[0, 1])
    return dict(b=float(b), a=float(a), r=r, n=len(xs))


# =========================================================================================
def analizar_p1(filas):
    print("\n" + "=" * 108)
    print("PRUEBA 1 — kcap ABSOLUTO fijo, N variando (¿la geometría depende del kcap absoluto o del relativo?)")
    print("=" * 108)
    if not filas:
        return
    kcaps = sorted({f["kcap_base"] for f in filas})
    Ns = sorted({f["N_sistema"] for f in filas})
    print(f"\n  Pendiente continua corregida — media ± error estándar (n reglas por celda)")
    print(f"  {'kcap':<6} " + "".join(f"{'N='+str(N):>20}" for N in Ns) + "   deriva b(pend~logN)   r")
    g = _agrupar(filas, "kcap_base", "N_sistema")
    for kc in kcaps:
        linea = f"  {kc:<6} "
        for N in Ns:
            r = _res([f["pendiente"] for f in g.get((kc, N), [])])
            linea += f"{r['media']:>12.3f}±{r['ee']:.3f}({r['n']:>2})" if r["n"] else f"{'—':>20}"
        dd = pendiente_de_pendiente([f for f in filas if f["kcap_base"] == kc])
        linea += f"   {dd['b']:+.3f}   r={dd['r']:+.2f}"
        print(linea)

    print(f"\n  Grado medio FINAL alcanzado (b=1) — cuánto del cupo se llena de verdad")
    print(f"  {'kcap':<6} " + "".join(f"{'N='+str(N):>14}" for N in Ns))
    for kc in kcaps:
        print(f"  {kc:<6} " + "".join(f"{_res([f['grado_medio_b1'] for f in g.get((kc,N),[])])['media']:>14.2f}" for N in Ns))

    print(f"\n  Contraste ENTRE kcap, pareado por (regla, N) — ¿importa el número absoluto?")
    for i in range(len(kcaps)):
        for j in range(i + 1, len(kcaps)):
            a, b = kcaps[i], kcaps[j]
            r = pareado([f for f in filas if f["kcap_base"] == a], [f for f in filas if f["kcap_base"] == b])
            if r:
                print(f"    kcap={a} vs kcap={b}:  n={r['n']:<3} media_dif={r['media']:+.3f} "
                      f"mediana_dif={r['mediana']:+.3f}  gana_{a}={r['gana_A']:<3} gana_{b}={r['gana_B']:<3} "
                      f"Wilcoxon p={r['p']:.2g}")

    print(f"\n  Contraste ENTRE N, pareado por (regla, kcap) — ¿deriva con el tamaño a cupo fijo?")
    for i in range(len(Ns) - 1):
        a, b = Ns[i], Ns[-1]
        ia = {(f["rule_id"], f["kcap_base"]): f["pendiente"] for f in filas if f["N_sistema"] == a}
        ib = {(f["rule_id"], f["kcap_base"]): f["pendiente"] for f in filas if f["N_sistema"] == b}
        com = sorted(set(ia) & set(ib), key=str)
        d = np.array([ib[c] - ia[c] for c in com])
        p = float(SP.wilcoxon(d).pvalue) if (SP is not None and len(d) >= 5 and np.any(d != 0)) else float("nan")
        print(f"    N={b} menos N={a}:  n={len(d):<3} media_dif={d.mean():+.3f} mediana_dif={np.median(d):+.3f} "
              f"sube={int((d>0).sum())} baja={int((d<0).sum())} Wilcoxon p={p:.2g}")

    print(f"\n  Modelo descriptivo conjunto  pendiente ~ a + b1*log(kcap) + b2*log(N)")
    X, y = [], []
    for f in filas:
        if np.isfinite(f["pendiente"]):
            X.append([1.0, np.log(f["kcap_base"]), np.log(f["N_sistema"])]); y.append(f["pendiente"])
    if len(y) > 5:
        X, y = np.array(X), np.array(y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        R2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"    a={coef[0]:+.3f}   b1(log kcap)={coef[1]:+.3f}   b2(log N)={coef[2]:+.3f}   R²={R2:.3f}  n={len(y)}")
        print(f"    lectura: si |b1| >> |b2|, manda el NÚMERO ABSOLUTO de cupo; si son comparables y de signo")
        print(f"             opuesto, lo que manda es la RAZÓN capacidad/tamaño.")
        if abs(coef[2]) > 1e-9:
            print(f"    razón b1/b2 = {coef[1]/coef[2]:+.2f}  (=-1 exacto sería 'sólo importa kcap/N^1'; "
                  f"0 sería 'no importa N')")


def curva_kcap(p1, p2):
    """La curva que unifica P1 y P2: pendiente en función del cupo ABSOLUTO, una serie por tamaño.
    Si las series de distintos N se superponen, la pendiente es función de kcap SOLO — y entonces no
    hay nada especial en ningún valor concreto: hay una rampa continua."""
    print("\n" + "=" * 108)
    print("CURVA UNIFICADA (P1 + P2 juntas) — pendiente en función del cupo ABSOLUTO, por tamaño")
    print("=" * 108)
    todas = list(p1) + list(p2)
    Ns = sorted({f["N_sistema"] for f in todas})
    kcaps = sorted({f["kcap_base"] for f in todas})
    g = _agrupar(todas, "kcap_base", "N_sistema")
    print(f"  {'':<8}" + "".join(f"{'N='+str(N):>16}" for N in Ns) + f"{'TODOS los N':>18}")
    for kc in kcaps:
        linea = f"  kcap={kc:<3}"
        for N in Ns:
            r = _res([f["pendiente"] for f in g.get((kc, N), [])])
            linea += f"{r['media']:>10.3f}({r['n']:>2})" if r["n"] else f"{'—':>16}"
        rt = _res([f["pendiente"] for f in todas if f["kcap_base"] == kc])
        linea += f"{rt['media']:>12.3f}({rt['n']:>3})"
        print(linea)
    x = np.array([np.log(f["kcap_base"]) for f in todas], float)
    y = np.array([f["pendiente"] for f in todas], float)
    m = np.isfinite(x) & np.isfinite(y)
    b, a = np.polyfit(x[m], y[m], 1)
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    print(f"\n  Ajuste global  pendiente = {a:+.3f} {b:+.3f}·log(kcap)   r={r:+.3f}  R²={r*r:.3f}  n={int(m.sum())}")
    print(f"  ¿Escalón en algún kcap? diferencias entre cupos consecutivos (medias sobre todos los N):")
    medias = {kc: _res([f["pendiente"] for f in todas if f["kcap_base"] == kc])["media"] for kc in kcaps}
    for i in range(len(kcaps) - 1):
        print(f"    kcap {kcaps[i]}→{kcaps[i+1]}:  Δpendiente = {medias[kcaps[i+1]]-medias[kcaps[i]]:+.3f}")


def analizar_p2(filas_p2, filas_p1):
    print("\n" + "=" * 108)
    print("PRUEBA 2 — capacidad NORMALIZADA al tamaño (¿alguna normalización conserva la familia de geometrías?)")
    print("=" * 108)
    if not filas_p2:
        return
    # la referencia ABS es la rama kcap=6 de la Prueba 1 (mismo anclaje, mismas reglas)
    abs6 = [dict(f, condicion="ABS(kcap=6)") for f in filas_p1 if f["kcap_base"] == 6]
    todas = abs6 + filas_p2
    Ns = sorted({f["N_sistema"] for f in todas})
    conds = ["ABS(kcap=6)"] + sorted({f["condicion"] for f in filas_p2})
    g = _agrupar(todas, "condicion", "N_sistema")
    print(f"\n  Pendiente continua — media ± ee, y el kcap que cada normalización impone a cada N")
    print(f"  {'normalización':<14} " + "".join(f"{'N='+str(N):>20}" for N in Ns) + "   deriva b(pend~logN)   r")
    for c in conds:
        linea = f"  {c:<14} "
        for N in Ns:
            fs = g.get((c, N), [])
            r = _res([f["pendiente"] for f in fs])
            kc = fs[0]["kcap_base"] if fs else "—"
            linea += f"{r['media']:>8.3f}±{r['ee']:.3f}[k={kc}]" if r["n"] else f"{'—':>20}"
        dd = pendiente_de_pendiente([f for f in todas if f["condicion"] == c])
        linea += f"   {dd['b']:+.3f}   r={dd['r']:+.2f}"
        print(linea)
    print(f"\n  Dispersión de la pendiente ENTRE tamaños (desviación de las 4 medias por N): cuanto MENOR,")
    print(f"  más se conserva la familia de geometrías a través de escalas.")
    for c in conds:
        medias = [_res([f["pendiente"] for f in g.get((c, N), [])])["media"] for N in Ns]
        medias = [m for m in medias if np.isfinite(m)]
        if len(medias) >= 2:
            print(f"    {c:<14} medias por N = {['%.3f'%m for m in medias]}   sd_entre_N={np.std(medias, ddof=1):.3f}  "
                  f"rango={max(medias)-min(medias):.3f}")


def analizar_p3(filas):
    print("\n" + "=" * 108)
    print("PRUEBA 3 — capacidad HETEROGÉNEA entre nodos, misma media (¿hace falta homogeneidad?)")
    print("=" * 108)
    if not filas:
        return
    Ns = sorted({f["N_sistema"] for f in filas})
    conds = sorted({f["condicion"] for f in filas}, key=lambda c: _res([f["kcap_cv"] for f in filas if f["condicion"] == c])["media"])
    g = _agrupar(filas, "condicion", "N_sistema")
    print(f"\n  Pendiente continua — media ± ee por condición y tamaño; CV = dispersión del cupo entre nodos")
    print(f"  {'condición':<14} {'CV':>6} {'kcap_med':>9} {'kcap_max':>9}  " + "".join(f"{'N='+str(N):>18}" for N in Ns) + f"{'TODOS':>18}")
    for c in conds:
        fc = [f for f in filas if f["condicion"] == c]
        cv = _res([f["kcap_cv"] for f in fc])["media"]
        km = _res([f["kcap_media_emp"] for f in fc])["media"]
        kx = _res([float(f["kcap_max"]) for f in fc])["media"]
        linea = f"  {c:<14} {cv:>6.2f} {km:>9.2f} {kx:>9.1f}  "
        for N in Ns:
            r = _res([f["pendiente"] for f in g.get((c, N), [])])
            linea += f"{r['media']:>11.3f}±{r['ee']:.3f}" if r["n"] else f"{'—':>18}"
        rt = _res([f["pendiente"] for f in fc])
        linea += f"{rt['media']:>11.3f}±{rt['ee']:.3f}"
        print(linea)

    print(f"\n  Grado medio FINAL (b=1) por condición — control de que la 'masa de aristas' es comparable")
    for c in conds:
        fc = [f for f in filas if f["condicion"] == c]
        print(f"    {c:<14} grado_medio={_res([f['grado_medio_b1'] for f in fc])['media']:.2f}  "
              f"aristas_b1={_res([float(f['n_aristas_b1']) for f in fc])['media']:.0f}  "
              f"diam_b1={_res([f['diam_b1'] for f in fc])['media']:.1f}  "
              f"giant_b1={_res([f['giant_b1'] for f in fc])['media']:.3f}")

    print(f"\n  Contraste PAREADO contra el caso homogéneo UNIF (misma regla, mismo N)")
    base = [f for f in filas if f["condicion"] == "UNIF"]
    for c in conds:
        if c == "UNIF":
            continue
        r = pareado([f for f in filas if f["condicion"] == c], base)
        if r:
            print(f"    {c:<14} vs UNIF:  n={r['n']:<3} media_dif={r['media']:+.3f} mediana_dif={r['mediana']:+.3f}  "
                  f"sube={r['gana_A']:<3} baja={r['gana_B']:<3} Wilcoxon p={r['p']:.2g}")

    print(f"\n  ¿Dosis-respuesta? correlación pendiente vs dispersión del cupo (CV), sobre TODAS las corridas")
    x = np.array([f["kcap_cv"] for f in filas]); y = np.array([f["pendiente"] for f in filas])
    m = np.isfinite(x) & np.isfinite(y)
    if SP is not None and m.sum() > 10:
        rho, pr = SP.spearmanr(x[m], y[m])
        pe, pp = SP.pearsonr(x[m], y[m])
        print(f"    Spearman rho={rho:+.3f} (p={pr:.2g})   Pearson r={pe:+.3f} (p={pp:.2g})   n={int(m.sum())}")


def colapso_por_grado(p1, p2, p3):
    """EL CONTROL QUE DECIDE LA PRUEBA 3.

    Un cupo es un TECHO, no una cuota: subirle el cupo a un nodo que sólo tiene 6 vecinos disponibles no
    le agrega ninguno, pero bajárselo sí le corta. Por eso, aunque todas las distribuciones heterogéneas
    tengan la MISMA media de cupo, no tienen por qué terminar con el mismo número de aristas: el efecto
    del techo es asimétrico y la heterogeneidad, a media constante, tiende a BAJAR el grado realmente
    alcanzado. Analogía: si a la mitad del pueblo le decís 'máximo 2 amigos' y a la otra mitad 'máximo
    10', el promedio del permiso es 6 — pero los de 10 igual no encuentran 10 personas con quién ser
    amigos, así que el pueblo termina con menos amistades que si a todos les hubieras dicho 'máximo 6'.

    Este bloque separa las dos explicaciones posibles del efecto de la heterogeneidad:
      (a) la FORMA del reparto importa por sí misma, o
      (b) la heterogeneidad sólo actúa BAJANDO la saturación efectiva, y una vez que se controla por el
          grado realmente alcanzado, un sistema heterogéneo es indistinguible de uno homogéneo con ese
          mismo grado.
    Se ajusta la curva pendiente ~ a + b·log(grado medio alcanzado) SÓLO con las corridas HOMOGÉNEAS
    (P1+P2, cupo uniforme) y se mira el residuo de las heterogéneas contra esa curva. Residuo ≈ 0 apoya
    (b); residuo sistemáticamente distinto de 0 apoya (a)."""
    print("\n" + "=" * 108)
    print("CONTROL DECISIVO — ¿la heterogeneidad importa por su FORMA, o sólo por bajar la saturación efectiva?")
    print("=" * 108)
    homog = [f for f in list(p1) + list(p2) if np.isfinite(f["pendiente"]) and f["grado_medio_b1"] > 0]
    if len(homog) < 10 or not p3:
        print("  (faltan datos)")
        return
    x = np.log([f["grado_medio_b1"] for f in homog]); y = np.array([f["pendiente"] for f in homog])
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"  Curva de referencia, ajustada SÓLO con cupo homogéneo (P1+P2, n={len(y)}):")
    print(f"     pendiente = {a:+.3f} {b:+.3f}·log(grado medio alcanzado)     r={r:+.3f}  R²={r*r:.3f}")
    print(f"\n  Residuo de cada condición de la Prueba 3 contra esa curva (0 = cae exactamente encima):")
    print(f"  {'condición':<14} {'CV cupo':>8} {'grado real':>11} {'pend. obs':>10} {'pend. predicha':>15} {'residuo':>9}")
    for c in sorted({f["condicion"] for f in p3},
                    key=lambda c: _res([f["kcap_cv"] for f in p3 if f["condicion"] == c])["media"]):
        fc = [f for f in p3 if f["condicion"] == c and np.isfinite(f["pendiente"]) and f["grado_medio_b1"] > 0]
        if not fc:
            continue
        cv = _res([f["kcap_cv"] for f in fc])["media"]
        gr = _res([f["grado_medio_b1"] for f in fc])["media"]
        obs = np.array([f["pendiente"] for f in fc])
        pred = a + b * np.log([f["grado_medio_b1"] for f in fc])
        res = obs - pred
        pw = float(SP.wilcoxon(res).pvalue) if (SP is not None and len(res) >= 5) else float("nan")
        print(f"  {c:<14} {cv:>8.2f} {gr:>11.2f} {obs.mean():>10.3f} {pred.mean():>15.3f} "
              f"{res.mean():>+9.3f}  (ee={res.std(ddof=1)/np.sqrt(len(res)):.3f}, Wilcoxon vs 0 p={pw:.2g})")


def cruce_datos_archivados():
    """CRUCE INDEPENDIENTE, sin correr nada nuevo: las 20 reglas de C2-hard de F5-C2-C3 tenían `kcap`
    sampleado al azar en 4-7 por el generador, y sus pendientes ya fueron re-medidas con el diámetro
    corregido en la tarea O1-B (`cs090_fase6_remedicion_mecanismo.csv`). Si la rampa pendiente-vs-kcap
    que sale de esta tarea es real, tiene que estar YA presente, sin que nadie la hubiera mirado, en
    ese archivo histórico. Este chequeo no genera datos: sólo cruza dos CSV que ya existen."""
    print("\n" + "=" * 108)
    print("CRUCE con datos YA ARCHIVADOS (F5-C2-C3 'C2-hard' re-medido en O1-B) — ¿la rampa ya estaba ahí?")
    print("=" * 108)
    try:
        rem = [r for r in csv.DictReader(open(f"{_HERE}/cs090_fase6_remedicion_mecanismo.csv"))
               if r["tarea"] == "mecanismo_aislado" and r["brazo"] == "C2-hard"]
        res = [r for r in csv.DictReader(open(f"{_HERE}/cs090_fase5_mecanismo_aislado_resumen.csv"))
               if r["brazo"] == "C2-hard"]
    except FileNotFoundError as e:
        print(f"  (no disponible: {e})")
        return
    kc = {r["rule_id"]: int(r["kcap"]) for r in res}
    x = np.array([kc[r["rule_id"]] for r in rem], float)
    y = np.array([float(r["pendiente_corregida"]) for r in rem], float)
    for k in sorted(set(x)):
        print(f"    kcap={int(k)}  n={int((x==k).sum()):<3} pendiente media (N=2000, archivo) = {y[x==k].mean():.3f}")
    if SP is not None and len(x) > 5:
        rho, p = SP.spearmanr(x, y)
        print(f"    Spearman kcap vs pendiente = {rho:+.3f} (p={p:.2g}), n={len(x)} reglas archivadas")


def figura(p1, p2, p3, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(sin figura: {e})")
        return
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # panel 1 — pendiente vs N para cada kcap absoluto
    g = _agrupar(p1, "kcap_base", "N_sistema")
    Ns = sorted({f["N_sistema"] for f in p1})
    for kc in sorted({f["kcap_base"] for f in p1}):
        m = [_res([f["pendiente"] for f in g.get((kc, N), [])]) for N in Ns]
        ax[0].errorbar(Ns, [x["media"] for x in m], yerr=[x["ee"] for x in m], marker="o",
                       capsize=3, label=f"kcap={kc}")
    ax[0].axhspan(0.35, 0.45, color="0.85", zorder=0)
    ax[0].axhline(0.7, ls=":", c="0.4")
    ax[0].set_xscale("log"); ax[0].set_xlabel("N del sistema"); ax[0].set_ylabel("pendiente log(diám)-log(N_cajas)")
    ax[0].set_title("P1 — kcap absoluto fijo\n(banda gris: mundo-pequeño 0.35-0.45; línea: 0.7)")
    ax[0].legend(fontsize=8)

    # panel 2 — normalizaciones
    abs6 = [dict(f, condicion="ABS(kcap=6)") for f in p1 if f["kcap_base"] == 6]
    todas = abs6 + p2
    g2 = _agrupar(todas, "condicion", "N_sistema")
    Ns2 = sorted({f["N_sistema"] for f in todas})
    for c in sorted({f["condicion"] for f in todas}):
        m = [_res([f["pendiente"] for f in g2.get((c, N), [])]) for N in Ns2]
        ax[1].errorbar(Ns2, [x["media"] for x in m], yerr=[x["ee"] for x in m], marker="s", capsize=3, label=c)
    ax[1].axhspan(0.35, 0.45, color="0.85", zorder=0)
    ax[1].set_xscale("log"); ax[1].set_xlabel("N del sistema"); ax[1].set_ylabel("pendiente")
    ax[1].set_title("P2 — capacidad normalizada al tamaño")
    ax[1].legend(fontsize=8)

    # panel 3 — heterogeneidad: pendiente vs CV del cupo
    conds = sorted({f["condicion"] for f in p3},
                   key=lambda c: _res([f["kcap_cv"] for f in p3 if f["condicion"] == c])["media"])
    xs = [_res([f["kcap_cv"] for f in p3 if f["condicion"] == c])["media"] for c in conds]
    for N in sorted({f["N_sistema"] for f in p3}):
        ys = [_res([f["pendiente"] for f in p3 if f["condicion"] == c and f["N_sistema"] == N]) for c in conds]
        ax[2].errorbar(xs, [y["media"] for y in ys], yerr=[y["ee"] for y in ys], marker="o", capsize=3, label=f"N={N}")
    for x, c in zip(xs, conds):
        ax[2].annotate(c, (x, ax[2].get_ylim()[0]), rotation=90, fontsize=7, va="bottom", ha="center", color="0.3")
    ax[2].axhspan(0.35, 0.45, color="0.85", zorder=0)
    ax[2].set_xlabel("CV del cupo entre nodos (0 = todos igual)"); ax[2].set_ylabel("pendiente")
    ax[2].set_title("P3 — capacidad heterogénea, MISMA media")
    ax[2].legend(fontsize=8)

    fig.suptitle("O2-C — ¿es kcap=6 un número especial, o basta con que la capacidad de relación sea finita?", y=1.0)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    print(f"\nPNG: {ruta}")


if __name__ == "__main__":
    suf = sys.argv[1] if len(sys.argv) > 1 else ""
    p1 = cargar(f"{_HERE}/cs090_fase6_o2c_p1_kcap_absoluto{suf}.csv")
    p2 = cargar(f"{_HERE}/cs090_fase6_o2c_p2_normalizado{suf}.csv")
    p3 = cargar(f"{_HERE}/cs090_fase6_o2c_p3_heterogeneo{suf}.csv")
    print(f"filas cargadas: P1={len(p1)}  P2={len(p2)}  P3={len(p3)}")
    analizar_p1(p1)
    curva_kcap(p1, p2)
    analizar_p2(p2, p1)
    cruce_datos_archivados()
    analizar_p3(p3)
    colapso_por_grado(p1, p2, p3)

    # dato SECUNDARIO (continuidad con informes anteriores): reparto de clases
    print("\n" + "=" * 108)
    print("DATO SECUNDARIO (no es el endpoint) — reparto de clases I-IV por condición")
    print("=" * 108)
    from collections import Counter
    for nombre, fs in (("P1", p1), ("P2", p2), ("P3", p3)):
        for c in sorted({f["condicion"] for f in fs}):
            sub = [f for f in fs if f["condicion"] == c]
            cnt = Counter(f["clase"][:12] for f in sub)
            n3 = sum(v for k, v in cnt.items() if k.startswith("III") or k.startswith("IV"))
            print(f"  {nombre} {c:<14} n={len(sub):<4} %III+IV={100*n3/len(sub):5.1f}%   {dict(cnt)}")
    if p1:
        figura(p1, p2, p3, f"{_HERE}/cs090_fase6_o2c_capacidad_finita.png")
    print("\nSin cierre ni veredicto: números para la lectura de Alexis.")
