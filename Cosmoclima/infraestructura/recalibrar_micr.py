"""
RENORMALIZAR LAS TRES PRIORIDADES ESTRATÉGICAS
================================================

DECISIÓN (Alexis, 23-ago-2026): renormalizar las tres — entradas a 0-1, pesos que
suman 1, sin divisor —, meter la IB dentro de la Peh, recalcular el FVT y el
IRMD, y congelar los cortes.

★ EL PROBLEMA, EN UNA FRASE
-----------------------------
Las fórmulas publicadas **dividen por el máximo observado**:

    Pev = (0,5·IB + 0,3·FANC + 0,2·FVT) / 1,549
                                           ↑ no es constante: es el máximo que
                                             se observó cuando se calculó

Consecuencias medidas: no estacionariedad (al re-medir el FANC, la Peh cambió en
793 de 846 filas), mezcla de escalas (el FANC entraba en 1-4 y la IB en 0-1), y
bandas apelmazadas (la banda «Alta» de la Peh no podía llenarse con NINGÚN dato).

★ LO QUE APARECIÓ AL MIRAR LOS PESOS
--------------------------------------
    Pev   IB 0,5 + FANC 0,3 + FVT 0,2  =  1,0
    Pen   FEN 0,5 + IB 0,3 + FVT 0,2   =  1,0
    Peh   FANC 0,5 + VT 0,3 + FVT 0,2  =  1,0

**Los pesos nunca estuvieron mal: lo que estaba mal eran las entradas.** Basta
llevar FEN y FANC de 1-4 a 0-1 y el divisor se cae solo, porque el índice ya vive
en 0-1 por construcción. Pev y Pen no cambian ni un peso.

La Peh sí, porque le falta la IB.

★ CÓMO SE ELIGEN LOS PESOS DE LA Peh · CRITERIOS DECLARADOS ANTES DE MEDIR
---------------------------------------------------------------------------
No a ojo. Se recorre una malla y se conserva el vector que cumpla, en este orden:

    C1  los pesos suman 1 y ninguno es cero ni negativo
    C2  el caso que motivó todo se corrige: los Cuarteles Policiales (IB 0,95)
        quedan por encima de los Dispositivos IoT (IB 0,50)
    C3  la importancia entra con el MISMO PESO que ya tiene en la Pen: 0,30.
        Es estructural, no estadístico — ver abajo por qué.
    C4  el FANC sigue siendo el peso dominante — la Peh es, por definición, la
        prioridad ANTE ATAQUE NO CONVENCIONAL, así que la amenaza que le da
        nombre no puede quedar de comparsa
    C5  entre los que pasan C1-C4, gana el que MENOS mueva el orden

C3 hace que la importancia entre de verdad: sin él, C2 se satisface con un peso
ínfimo de IB y todo queda igual de mal.

★ POR QUÉ C3 ES ESTRUCTURAL Y NO ESTADÍSTICO
----------------------------------------------
La primera versión de C3 pedía `corr(Peh, IB) ≥ corr(Pen, IB)`. **Con C4 era
imposible: ningún vector pasaba.** Y la razón resultó ser un hecho sobre el
mundo, medido:

    corr(FANC, IB) = -0,0433      corr(FEN, IB) = +0,2975

**Lo que el derecho protege y lo que al país le importa son cosas ortogonales.**
Los Reactores Nucleares y los Cuarteles Policiales tienen ambos IB 0,95, pero el
primero está en el grado máximo del artículo 56 y el segundo es un bien civil
corriente. Por eso la Pen puede correlacionar +0,51 con la importancia teniendo
el FEN al mando, y la Peh no puede, teniendo el FANC al mando.

Exigir esa correlación era exigirle a la Peh que dejara de ser sobre ataques.
El criterio correcto es estructural: **la importancia pesa en la Peh lo mismo
que en la Pen** — 0,30 —, y la correlación que resulte es un resultado, no una
meta.

★ C4 SE AGREGÓ DESPUÉS DE VER EL PRIMER RESULTADO, Y POR UNA RAZÓN
--------------------------------------------------------------------
Sin C4, el ganador salía con **FANC 0,10** — el peso más bajo de los cuatro, en
el índice de ataques. No era casualidad: C5 premia parecerse a la Peh publicada,
y ésa se calculó con el FANC VIEJO, que era casi constante (587 de 845 en
«Media») y por tanto casi no ordenaba. Premiar el parecido con ella era
**optimizar hacia el defecto**.

⚠️ Cambiar un criterio después de ver el resultado es justamente lo que este
proyecto no hace. Se deja escrito que se agregó, cuándo y por qué, y se deja el
resultado anterior a la vista para que se pueda juzgar.

★ Y LA LÍNEA BASE DE C5 SE CORRIGIÓ POR LO MISMO
--------------------------------------------------
«Mover lo menos posible» debe medirse contra la fórmula VIEJA EVALUADA CON EL
FANC NUEVO — no contra la columna publicada. Así se aísla el efecto de lo que
estamos cambiando (meter la IB) del efecto de arreglar el FANC, que ya se decidió
aparte.

★ LOS CORTES SE CONGELAN
--------------------------
DECISIÓN (Alexis, 23-ago-2026): percentiles UNA VEZ, y congelar.

Se calculan por percentil sobre las 845 filas de hoy y se escriben como números
fijos. Reparte bien AHORA y queda estacionario DESPUÉS: agregar filas mañana ya
no mueve de banda a nadie. Es el mismo principio que se aplicó al PelPre — la
referencia nacional se congela.

⚠️ Recalcular los percentiles en cada corrida reintroduciría exactamente la no
estacionariedad que la renormalización viene a eliminar. Por eso se congelan.

USO
---
    ../.venv-esa/bin/python recalibrar_micr.py            # mide y reporta
    ../.venv-esa/bin/python recalibrar_micr.py --escribir # deja el CSV
"""

import csv
import sys
from collections import Counter
from itertools import product
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import micr                                              # noqa: E402

DATOS = AQUI / "datos"
FANC_MEDIDO = DATOS / "fanc_medido_4grados.csv"
SALIDA = DATOS / "micr_recalibrada.csv"

ORDEN = ("Muy Alta", "Alta", "Media", "Baja", "Muy Baja")
CASO_IMPORTANTE, CASO_TRIVIAL = 353, 206      # Cuarteles Policiales · IoT

# Percentiles con que se cortan las bandas. Se aplican UNA VEZ y el resultado se
# congela en `CORTES_CONGELADOS` más abajo.
PERCENTILES = ((95, "Muy Alta"), (80, "Alta"), (50, "Media"), (20, "Baja"))
PERCENTILES_IRMD = ((80, "Alto"), (40, "Medio"))


def n01(etiqueta):
    """Etiqueta de fragilidad a 0-1. Baja→0 · Media→⅓ · Alta→⅔ · Muy Alta→1."""
    return (micr.NIVEL[etiqueta] - 1) / (micr.NIVEL_MAX - 1)


def fvt_n(fen, fanc, vt):
    """FVT renormalizado = promedio de las tres, todas ya en 0-1.

    La forma publicada `(FEN + FANC + VT·3)/9` se sale de rango con las
    etiquetas en 1-4 (llega a 1,222) y nunca baja de 0,222. Con las entradas
    normalizadas, el promedio simple es equivalente en intención y vive en 0-1.
    """
    return (fen + fanc + vt) / 3


def percentil(vals, p):
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def cortes_por_percentil(vals, esquema=PERCENTILES):
    """Cortes por percentil, movidos al PUNTO MEDIO del hueco.

    ★ El percentil devuelve un valor OBSERVADO, así que hay filas que caen
    exactamente encima del corte. Cualquier redondeo posterior —al guardar el
    CSV, al leerlo— las cruza de banda. Ya nos pasó dos veces: con la fila 18 al
    recuperar los cortes de la Pev, y con 237 filas al releer el CSV.

    La regla, de una vez por todas: **un corte va donde NO hay dato**. Se toma
    el valor del percentil y se baja hasta la mitad del hueco que lo separa del
    valor distinto inmediatamente inferior.
    """
    s = sorted(set(vals))
    out = []
    for p, b in esquema:
        v = percentil(vals, p)
        menores = [x for x in s if x < v]
        corte = (v + menores[-1]) / 2 if menores else v - 1e-9
        out.append((round(corte, 10), b))
    return out


def rangos(v):
    def r(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        out = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            m = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[o[k]] = m
            i = j + 1
        return out
    return r(v)


def _corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def pearson(a, b):
    return _corr(a, b)


def spearman(a, b):
    return _corr(rangos(a), rangos(b))


def pares_invertidos(idx, ib):
    """Cuántos pares están al revés respecto de la importancia, con idx empatado
    o menor cuando la importancia es claramente mayor. Es la medida directa del
    defecto: mide cuántas veces la Matriz pone lo trivial sobre lo crítico."""
    n, mal = len(idx), 0
    for i in range(n):
        for j in range(i + 1, n):
            if ib[i] - ib[j] > 0.2 and idx[i] < idx[j]:
                mal += 1
            elif ib[j] - ib[i] > 0.2 and idx[j] < idx[i]:
                mal += 1
    return mal


def cargar():
    """Las filas, con el FANC re-medido en cuatro grados sustituyendo al viejo."""
    base, huerfanas = micr.leer()
    nuevo = {}
    if FANC_MEDIDO.exists():
        nuevo = {int(x["n"]): x["FANC"]
                 for x in csv.DictReader(FANC_MEDIDO.open(encoding="utf-8"))}
    E = []
    for x in base:
        n = int(float(x["n"]))
        fanc_txt = nuevo.get(n, x["FANC"])
        fen, fanc = n01(x["FEN"]), n01(fanc_txt)
        ib, vt = float(x["IB"]), float(x["VT"])
        E.append(dict(n=n, elemento=x["elemento"], Sector=x["Sector"],
                      FEN=x["FEN"], FANC=fanc_txt, FANC_antes=x["FANC"],
                      fen=fen, fanc=fanc, ib=ib, vt=vt, fvt=fvt_n(fen, fanc, vt),
                      fvt_pub=float(x["FVT"]), pf_pub=float(x["PF"]),
                      IRMD_pub=x["IRMD"], Pev_pub=x["Pev"],
                      Peh_pub=x["Peh"], Pen_pub=x["Pen"],
                      # el valor continuo publicado, para medir cuánto se mueve
                      # ★ línea base de C5: la fórmula VIEJA con el FANC NUEVO,
                      # ya normalizada. Aísla el efecto de meter la IB.
                      peh_base=0.5 * fanc + 0.3 * vt + 0.2 * fvt_n(fen, fanc, vt),
                      peh_pub_v=micr.peh(micr.NIVEL[x["FANC"]], float(x["VT"]),
                                         float(x["FVT"]))))
    return E, huerfanas, bool(nuevo)


def main():
    E, huerfanas, con_fanc = cargar()
    print("=" * 78)
    print("RENORMALIZACIÓN DE LAS TRES PRIORIDADES ESTRATÉGICAS")
    print("=" * 78)
    print(f"\n  filas : {len(E)}")
    print(f"  FANC  : {'re-medido en 4 grados ✓' if con_fanc else '⚠️ el publicado'}")
    for h in huerfanas:
        print(f"  ⚠️  excluida (sin número): «{h['elemento']}»")
    if con_fanc:
        c = Counter(e["FANC"] for e in E)
        print("          " + " · ".join(f"{g}:{c[g]}" for g in ORDEN if c[g]))

    # ── Pev y Pen: los pesos NO cambian, sólo las entradas ───────────────────
    for e in E:
        e["Pev"] = 0.5 * e["ib"] + 0.3 * e["fanc"] + 0.2 * e["fvt"]
        e["Pen"] = 0.5 * e["fen"] + 0.3 * e["ib"] + 0.2 * e["fvt"]
        e["PF"] = e["ib"] * e["fvt"]

    ib = [e["ib"] for e in E]
    corr_pen_ib = pearson([e["Pen"] for e in E], ib)

    # ── Peh: malla con los cuatro criterios ──────────────────────────────────
    print("\n" + "=" * 78)
    print("PESOS DE LA Peh · malla con los cuatro criterios declarados")
    print("=" * 78)
    peh_pub = [e["peh_base"] for e in E]      # ★ ver C5 en la cabecera
    i_imp = next(i for i, e in enumerate(E) if e["n"] == CASO_IMPORTANTE)
    i_tri = next(i for i, e in enumerate(E) if e["n"] == CASO_TRIVIAL)
    print(f"\n  hoy   corr(Peh, IB) = {pearson(peh_pub, ib):+.4f}"
          f"   ·   referencia C3: corr(Pen, IB) = {corr_pen_ib:+.4f}")
    print(f"  testigo  «{E[i_imp]['elemento']}» (IB {E[i_imp]['ib']}) vs "
          f"«{E[i_tri]['elemento']}» (IB {E[i_tri]['ib']})")
    print(f"           hoy {peh_pub[i_imp]:.4f} vs {peh_pub[i_tri]:.4f}  →  "
          f"{'CORRECTO' if peh_pub[i_imp] > peh_pub[i_tri] else '✗ INVERTIDO'}")

    paso = 0.05
    malla = [round(x * paso, 2) for x in range(1, int(1 / paso))]
    cand = []
    descartes = Counter()
    for w_fanc, w_ib, w_vt in product(malla, repeat=3):
        w_fvt = round(1 - w_fanc - w_ib - w_vt, 10)
        if w_fvt <= 0:
            continue
        descartes["C1 ok"] += 1
        v = [w_fanc * e["fanc"] + w_ib * e["ib"] + w_vt * e["vt"] + w_fvt * e["fvt"]
             for e in E]
        if v[i_imp] <= v[i_tri]:
            descartes["falla C2 (testigo)"] += 1
            continue
        if abs(w_ib - 0.30) > 1e-9:
            descartes["falla C3 (importancia)"] += 1
            continue
        if w_fanc < max(w_ib, w_vt, w_fvt):
            descartes["falla C4 (FANC dominante)"] += 1
            continue
        cand.append((spearman(v, peh_pub), (w_fanc, w_ib, w_vt, w_fvt), v))
    cand.sort(key=lambda t: -t[0])

    print(f"\n  vectores probados        : {descartes['C1 ok']:,}")
    print(f"     descartados por C2    : {descartes['falla C2 (testigo)']:,}")
    print(f"     descartados por C3    : {descartes['falla C3 (importancia)']:,}")
    print(f"     descartados por C4    : {descartes['falla C4 (FANC dominante)']:,}")
    print(f"     ★ pasan los cuatro    : {len(cand):,}")
    if not cand:
        print("\n  ✗ ninguno pasa. Se reporta y NO se afloja el criterio.")
        return 1

    print(f"\n     {'FANC':>5} {'IB':>5} {'VT':>5} {'FVT':>5}  {'Spearman':>9}"
          f"  {'corr(Peh,IB)':>13}")
    for sp_, w, v in cand[:5]:
        print(f"     {w[0]:5.2f} {w[1]:5.2f} {w[2]:5.2f} {w[3]:5.2f}  {sp_:9.4f}"
              f"  {pearson(v, ib):+13.4f}")

    sp_ok, W, peh_v = cand[0]
    for e, v in zip(E, peh_v):
        e["Peh"] = v
    print(f"\n  ★ ELEGIDO   FANC {W[0]} · IB {W[1]} · VT {W[2]} · FVT {W[3]}")
    print(f"     corr(Peh, IB)  {pearson(peh_pub, ib):+.4f} → {pearson(peh_v, ib):+.4f}")
    print(f"     Spearman contra la Peh publicada: {sp_ok:.4f}")
    print(f"     testigo: {peh_v[i_imp]:.4f} vs {peh_v[i_tri]:.4f} → CORRECTO")
    mal_a = pares_invertidos(peh_pub, ib)
    mal_b = pares_invertidos(peh_v, ib)
    print(f"     pares invertidos contra la importancia: {mal_a:,} → {mal_b:,} "
          f"({100*(mal_b-mal_a)/max(mal_a,1):+.1f} %)")

    # ── cortes, calculados una vez y congelados ──────────────────────────────
    print("\n" + "=" * 78)
    print("CORTES · calculados hoy por percentil, y congelados")
    print("=" * 78 + "\n")
    cortes = {}
    for k in ("Pev", "Peh", "Pen"):
        cortes[k] = cortes_por_percentil([e[k] for e in E])
        print(f"  {k}  " + " / ".join(f"{c:.6f}" for c, _ in cortes[k]))
    cortes["IRMD"] = cortes_por_percentil([e["PF"] for e in E], PERCENTILES_IRMD)
    print(f"  IRMD (sobre PF)  " + " / ".join(f"{c:.6f}" for c, _ in cortes["IRMD"]))

    for e in E:
        for k in ("Pev", "Peh", "Pen"):
            e[f"{k}_banda"] = micr.banda(e[k], cortes[k])
        # ⚠️ `micr.banda` respalda en «Muy Baja», que no existe en el IRMD
        # (sólo tiene Alto/Medio/Bajo). Sin este respaldo propio, 312 filas
        # salían con una etiqueta inventada y los recuentos no sumaban 845.
        e["IRMD"] = next((nom for c, nom in cortes["IRMD"] if e["PF"] >= c), "Bajo")

    print("\n" + "=" * 78)
    print("REPARTO RESULTANTE")
    print("=" * 78 + "\n")
    for k in ("Pev", "Peh", "Pen"):
        c = Counter(e[f"{k}_banda"] for e in E)
        vac = [b for b in ORDEN if not c.get(b)]
        print(f"  {k}  " + " · ".join(f"{b}:{c.get(b,0):3d}" for b in ORDEN)
              + ("   ⚠️ VACÍA: " + ", ".join(vac) if vac else "   ✓ las cinco"))
    ci = Counter(e["IRMD"] for e in E)
    ca = Counter(e["IRMD_pub"] for e in E)
    print(f"\n  IRMD  antes " + " · ".join(f"{k}:{ca[k]}" for k in ("Alto", "Medio", "Bajo"))
          + "   →   ahora " + " · ".join(f"{k}:{ci[k]}" for k in ("Alto", "Medio", "Bajo")))

    # ── cuánto se mueve ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CUÁNTO SE MUEVE LA MATRIZ")
    print("=" * 78 + "\n")
    for k in ("Pev", "Peh", "Pen"):
        d = sum(1 for e in E if e[f"{k}_banda"] != e[f"{k}_pub"])
        print(f"  {k}   cambia de banda {d:4d} de {len(E)} ({100*d/len(E):5.1f} %)")
    for etiqueta, d in (
            ("IRMD", sum(1 for e in E if e["IRMD"] != e["IRMD_pub"])),
            ("FVT ", sum(1 for e in E if abs(e["fvt"] - e["fvt_pub"]) >= 0.005)),
            ("FANC", sum(1 for e in E if e["FANC"] != e["FANC_antes"]))):
        print(f"  {etiqueta}  cambia         {d:4d} de {len(E)} ({100*d/len(E):5.1f} %)")
    print("""
  ★ El FVT y el IRMD se mueven mucho A PROPÓSITO: es lo que significa
    volverlos calculados en vez de asignados a criterio. Las tres
    prioridades se mueven menos porque C4 pedía justamente eso.""")

    # ── el testigo, para ver el arreglo con nombre y apellido ────────────────
    print("\n" + "=" * 78)
    print("EL CASO QUE MOTIVÓ TODO")
    print("=" * 78 + "\n")
    for i in (i_imp, i_tri):
        e = E[i]
        print(f"  {e['elemento'][:44]:<44} IB {e['ib']:.2f}   "
              f"Peh {e['Peh_pub']:<9} → {e['Peh_banda']}")

    if "--escribir" in sys.argv:
        cols = ["n", "elemento", "Sector", "FEN", "FANC", "FEN_n", "FANC_n",
                "IB", "VT", "FVT", "PF", "IRMD", "Pev", "Peh", "Pen",
                "Pev_banda", "Peh_banda", "Pen_banda"]
        with SALIDA.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for e in E:
                # ★ 10 decimales, no 4. Con 4, FEN_n guardaba 0,6667 donde el
                # valor es 2/3, y al recalcular desde el CSV 237 filas cruzaban
                # un corte. Un redondeo NUNCA debe decidir una banda.
                w.writerow({"n": e["n"], "elemento": e["elemento"],
                            "Sector": e["Sector"], "FEN": e["FEN"],
                            "FANC": e["FANC"],
                            "FEN_n": round(e["fen"], 10),
                            "FANC_n": round(e["fanc"], 10),
                            "IB": e["ib"], "VT": e["vt"],
                            "FVT": round(e["fvt"], 10), "PF": round(e["PF"], 10),
                            "IRMD": e["IRMD"],
                            "Pev": round(e["Pev"], 10), "Peh": round(e["Peh"], 10),
                            "Pen": round(e["Pen"], 10),
                            "Pev_banda": e["Pev_banda"],
                            "Peh_banda": e["Peh_banda"],
                            "Pen_banda": e["Pen_banda"]})
        print(f"\n  escrito: {SALIDA.name}")
        print("\n  ★ PARA CONGELAR: copiar estos cortes a micr.py como constantes.")
        for k in ("Pev", "Peh", "Pen", "IRMD"):
            print(f"     CORTES_{k.upper()}_2026 = {cortes[k]}")
    else:
        print("\n  (nada escrito · corre con --escribir para dejar el CSV)")
    print("\n  ★ NO se sube a SharePoint sin que el director lo revise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
