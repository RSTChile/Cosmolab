"""
CS090 — FASE VI / O2-D: ANÁLISIS Y FIGURAS DEL CAMPO CONTINUO 2D
====================================================================================================
QUIÉN SOY: segundo archivo de O2-D (el primero, `cs090_fase6_o2d_campo2d.py`, genera y mide). Acá se
responde con números la pregunta central — **¿el campo continuo 2D muestra estructura persistente SIN
ningún grafo?** — y se producen las figuras.

Qué hace, en orden:
 1) CONTROL NULL PRIMERO (punto 3 de la tarea): para cada métrica, ¿distingue el campo REAL del mismo
    campo BARAJADO? Test de Wilcoxon pareado (cada corrida es su propio control) + conteo de signos +
    tamaño de efecto (delta de Cliff). Si una métrica no separa, NO se interpreta después: se dice.
 2) La pregunta central, con los 4 criterios pre-declarados en la tarea: pico nítido en P(k), entropía
    baja, percolación no trivial, dimensión no entera. Cada criterio con su número y su umbral.
 3) Persistencia temporal: la entropía de bloques, ¿baja (se estructura) o sube (se homogeniza)?
 4) Figuras: campos REAL vs BARAJADO, P(k) real vs null, curvas de percolación, y el resumen de las
    métricas por familia.

No declara cierre ni veredicto. Reporta números.
"""
from __future__ import annotations
import sys, csv, json
import numpy as np
from scipy import stats

_HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, _HERE)


def cargar(prefijo="cs090_fase6_o2d_2d"):
    with open(f"{_HERE}/{prefijo}_raw.csv") as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        for k, v in list(f.items()):
            if k in ("cfg_id", "familia", "entropia_traj"):
                continue
            try:
                f[k] = float(v)
            except (ValueError, TypeError):
                pass
    return filas


def cliff_delta(a, b):
    """Delta de Cliff: probabilidad de que un valor de `a` supere a uno de `b`, menos la inversa.
    Va de -1 a +1; |δ|≥0.474 se lee como efecto grande, 0.33 mediano, 0.147 chico. Se usa porque no
    supone normalidad ni escala — sólo orden — y acá las métricas tienen distribuciones muy distintas."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mayor = np.sum(a[:, None] > b[None, :])
    menor = np.sum(a[:, None] < b[None, :])
    return float((mayor - menor) / (len(a) * len(b)))


# métricas que tienen versión REAL y NULL, con la dirección que ESPERA la hipótesis de "hay estructura"
# (+1 = estructura debería dar un valor MAYOR que el barajado; -1 = MENOR)
METRICAS_PAREADAS = [
    ("prominencia_pico",     +1, "altura del pico de P(k) sobre el fondo"),
    ("entropia_espectral",   -1, "reparto de potencia entre modos (1=ruido blanco)"),
    ("nitidez_pico",         -1, "ancho relativo del pico (chico=afilado)"),
    ("exponente_ley_potencias", -1, "pendiente de log P(k) vs log k"),
    ("anisotropia",          +1, "dirección privilegiada en el espectro angular"),
    ("boxdim_occ5",          -1, "dimensión box-counting del 5% más alto"),
    ("boxdim_occ10",         -1, "dimensión box-counting del 10% más alto"),
    ("boxdim_occ25",         -1, "dimensión box-counting del 25% más alto"),
    ("boxdim_occ50",         -1, "dimensión box-counting de la mitad más alta"),
    ("entropia_bloques",     -1, "entropía de configuración espacial (1=desorden)"),
    ("masa_radio_alfa",      -1, "exponente masa-radio (2=homogéneo en 2D)"),
    ("xi",                   +1, "longitud de correlación"),
    ("std_valores",           0, "desvío de los valores (CONTROL: el NULL lo conserva exacto)"),
]


def control_null(filas, salida):
    """Punto 3 de la tarea, hecho ANTES de interpretar nada: ¿las métricas distinguen el campo real
    del mismo campo con las posiciones barajadas? Wilcoxon PAREADO (cada corrida contra su propio
    barajado — el emparejamiento es exacto, mismos valores, misma corrida) + conteo de signos +
    delta de Cliff. `std_valores` va como control negativo: el barajado conserva la distribución de
    valores por construcción, así que ahí NO debe haber diferencia — si la hubiera, el NULL estaría
    mal implementado."""
    p(salida, "=" * 100)
    p(salida, "1) CONTROL NULL — ¿las métricas distinguen el campo REAL del MISMO campo BARAJADO?")
    p(salida, "   (Wilcoxon pareado sobre las corridas admitidas; delta de Cliff como tamaño de efecto)")
    p(salida, "=" * 100)
    p(salida, f"{'métrica':<26}{'REAL media':>13}{'NULL media':>13}{'REAL>NULL':>12}{'Wilcoxon p':>13}{'Cliff δ':>10}  separa")
    resultados = {}
    for nombre, direccion, _desc in METRICAS_PAREADAS:
        r = np.array([f[f"{nombre}_real"] for f in filas], float)
        n = np.array([f[f"{nombre}_null"] for f in filas], float)
        ok = np.isfinite(r) & np.isfinite(n)
        r, n = r[ok], n[ok]
        if len(r) < 5:
            continue
        n_mayor = int(np.sum(r > n))
        try:
            _, pw = stats.wilcoxon(r, n)
        except ValueError:
            pw = 1.0
        d = cliff_delta(r, n)
        separa = (pw < 0.01) and (abs(d) >= 0.33) and (np.sign(d) == direccion or direccion == 0)
        marca = "SÍ" if separa else ("no" if direccion != 0 else "(control: debe ser 'no')")
        resultados[nombre] = dict(media_real=float(np.mean(r)), media_null=float(np.mean(n)),
                                  n_mayor=n_mayor, n=len(r), p=float(pw), delta=d, separa=bool(separa))
        p(salida, f"{nombre:<26}{np.mean(r):>13.4f}{np.mean(n):>13.4f}{f'{n_mayor}/{len(r)}':>12}"
                  f"{pw:>13.2e}{d:>10.3f}  {marca}")
    return resultados


def percolacion_y_criterios(filas, salida):
    """Punto 4 de la tarea: los 4 criterios pre-declarados de 'hay estructura persistente sin grafo'.
    Cada uno con su umbral explícito, evaluado corrida por corrida y resumido por familia."""
    p(salida, "")
    p(salida, "=" * 100)
    p(salida, "2) LOS 4 CRITERIOS PRE-DECLARADOS, corrida por corrida (resumen por familia)")
    p(salida, "=" * 100)
    dim_libre = float(filas[0]["ndim"])
    p(salida, "   pico nítido      : prominencia_pico_real > 10x la del barajado")
    p(salida, "   entropía baja    : entropia_bloques_real < 0.90 (el barajado da ~1.00)")
    p(salida, "   percolación no trivial: se declaró de antemano 'q_c real < 0.45' PENSANDO en un campo")
    p(salida, "     agrupado, que percola ANTES que el azar. Se reporta ese criterio tal como se")
    p(salida, "     pre-declaró, y ADEMÁS uno simétrico |q_c real − q_c null| ≥ 0.10, porque los datos")
    p(salida, "     mostraron desviación en la dirección CONTRARIA en una de las familias (manchas")
    p(salida, "     aisladas ⇒ el conjunto de nivel está FRAGMENTADO y percola DESPUÉS que el azar).")
    p(salida, "     Las dos direcciones son igual de no-triviales; sólo una estaba anticipada.")
    p(salida, f"   dimensión no entera: boxdim del 5% más alto < {dim_libre - 0.20:.2f} "
              f"(campo homogéneo en {int(dim_libre)}D daría ≈{dim_libre:.1f})")
    conteo = {}
    for f in filas:
        fam = f["familia"]
        c = conteo.setdefault(fam, dict(n=0, pico=0, entropia=0, perc_pre=0, perc_sim=0, dim=0,
                                        los4_pre=0, los4_sim=0))
        c["n"] += 1
        pico = f["prominencia_pico_real"] > 10 * f["prominencia_pico_null"]
        ent = f["entropia_bloques_real"] < 0.90
        perc_pre = np.isfinite(f["qc_real"]) and f["qc_real"] < 0.45
        perc_sim = (np.isfinite(f["qc_real"]) and np.isfinite(f["qc_null"])
                    and abs(f["qc_real"] - f["qc_null"]) >= 0.10)
        dim = np.isfinite(f["boxdim_occ5_real"]) and f["boxdim_occ5_real"] < dim_libre - 0.20
        for nombre, v in (("pico", pico), ("entropia", ent), ("perc_pre", perc_pre),
                          ("perc_sim", perc_sim), ("dim", dim)):
            c[nombre] += int(v)
        c["los4_pre"] += int(pico and ent and perc_pre and dim)
        c["los4_sim"] += int(pico and ent and perc_sim and dim)
    p(salida, "")
    p(salida, f"{'familia':<8}{'n':>4}{'pico':>7}{'H baja':>8}{'perc(pre)':>11}{'perc(sim)':>11}"
              f"{'dim':>6}{'4 (pre)':>9}{'4 (sim)':>9}")
    for fam, c in sorted(conteo.items()):
        p(salida, f"{fam:<8}{c['n']:>4}{c['pico']:>7}{c['entropia']:>8}{c['perc_pre']:>11}"
                  f"{c['perc_sim']:>11}{c['dim']:>6}{c['los4_pre']:>9}{c['los4_sim']:>9}")
    p(salida, "")
    teorico = 0.593 if dim_libre == 2 else 0.312
    p(salida, "   q_c del BARAJADO (control de que la referencia es la correcta): "
              f"media={np.nanmean([f['qc_null'] for f in filas]):.3f}, "
              f"rango=[{np.nanmin([f['qc_null'] for f in filas]):.2f}, "
              f"{np.nanmax([f['qc_null'] for f in filas]):.2f}]  "
              f"(percolación de sitios al azar en {int(dim_libre)}D: {teorico:.3f})")
    for fam in sorted(conteo):
        sub = [f for f in filas if f["familia"] == fam]
        p(salida, f"   q_c REAL {fam}: media={np.nanmean([f['qc_real'] for f in sub]):.3f}  "
                  f"| separación media REAL−barajado en la curva completa: "
                  f"{np.mean([f['max_dif_gigante'] for f in sub]):+.3f}")
    return conteo


def persistencia_temporal(filas, salida):
    """Punto 'el campo se estructura o se homogeniza con el tiempo': se mira la trayectoria de la
    entropía de bloques desde el 2% del tiempo hasta el 100%. Delta negativo = se estructura."""
    p(salida, "")
    p(salida, "=" * 100)
    p(salida, "3) PERSISTENCIA TEMPORAL — ¿el campo se estructura o se homogeniza?")
    p(salida, "   (entropía de configuración espacial en 5 instantes; delta<0 = se estructura)")
    p(salida, "=" * 100)
    p(salida, f"{'cfg':<10}{'H(2%)':>9}{'H(10%)':>9}{'H(25%)':>9}{'H(50%)':>9}{'H(100%)':>10}{'delta':>10}  tendencia")
    por_cfg = {}
    for f in filas:
        por_cfg.setdefault(f["cfg_id"], []).append(json.loads(f["entropia_traj"]))
    for cid, trs in sorted(por_cfg.items()):
        claves = sorted(trs[0], key=int)
        medias = [float(np.mean([t[k] for t in trs])) for k in claves]
        delta = medias[-1] - medias[0]
        tend = "SE ESTRUCTURA" if delta < -0.01 else ("se homogeniza" if delta > 0.01 else "estable")
        p(salida, f"{cid:<10}" + "".join(f"{m:>9.3f}" if i < 4 else f"{m:>10.3f}"
                                          for i, m in enumerate(medias[:5])) + f"{delta:>10.3f}  {tend}")


def figuras(filas, prefijo="cs090_fase6_o2d_2d"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    campos = np.load(f"{_HERE}/{prefijo}_campos.npy", allow_pickle=True).item()
    with open(f"{_HERE}/{prefijo}_espectros.csv") as fh:
        esp = list(csv.DictReader(fh))
    with open(f"{_HERE}/{prefijo}_percolacion.csv") as fh:
        perc = list(csv.DictReader(fh))

    cids = sorted(campos)
    # ---------------- FIGURA 1: campos REAL vs BARAJADO ----------------
    n = len(cids)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 4.0))
    for j, cid in enumerate(cids):
        for i, clave in enumerate(("campo", "campo_null")):
            ax = axes[i, j]
            ax.imshow(campos[cid][clave], cmap="magma", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cid, fontsize=7)
            if j == 0:
                ax.set_ylabel("REAL" if i == 0 else "BARAJADO", fontsize=8)
    fig.suptitle("O2-D · campo continuo 2D (arriba) y el MISMO campo con posiciones barajadas (abajo)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{_HERE}/{prefijo}_campos.png", dpi=130)
    plt.close(fig)

    # ---------------- FIGURA 2: espectros P(k) real vs null ----------------
    ncol = 5
    nrow = int(np.ceil(len(cids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow), squeeze=False)
    for idx, cid in enumerate(cids):
        ax = axes[idx // ncol][idx % ncol]
        sub = [e for e in esp if e["cfg_id"] == cid]
        ks = [float(e["k"]) for e in sub]
        ax.loglog(ks, [max(float(e["Pk_real"]), 1e-16) for e in sub], lw=1.2, label="REAL")
        ax.loglog(ks, [max(float(e["Pk_null"]), 1e-16) for e in sub], lw=1.0, ls="--", label="barajado")
        ax.set_title(cid, fontsize=8)
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=6)
    for idx in range(len(cids), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    fig.suptitle("O2-D · espectro de potencia radial P(k): pico nítido = escala característica; "
                 "plano = ruido blanco", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{_HERE}/{prefijo}_espectros.png", dpi=130)
    plt.close(fig)

    # ---------------- FIGURA 3: percolación + resumen de métricas ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for cid in cids:
        sub = [q for q in perc if q["cfg_id"] == cid]
        qs = [float(q["q"]) for q in sub]
        color = "tab:blue" if cid.startswith("FASE") else "tab:red"
        ax.plot(qs, [float(q["gigante_real"]) for q in sub], color=color, alpha=0.55, lw=1.0)
        ax.plot(qs, [float(q["gigante_null"]) for q in sub], color="0.6", alpha=0.35, lw=0.8, ls="--")
    ax.axvline(0.593, color="k", ls=":", lw=1.0)
    ax.text(0.60, 0.08, "q$_c$ teórico\nsitios al azar\n(0.593)", fontsize=7)
    ax.set_xlabel("ocupación q del conjunto de nivel"); ax.set_ylabel("componente mayor / sitios encendidos")
    ax.set_title("Percolación de conjuntos de nivel\n(azul=FASE, rojo=RD, gris punteado=barajado)", fontsize=9)

    ax = axes[1]
    for fam, color in (("FASE", "tab:blue"), ("RD", "tab:red")):
        sub = [f for f in filas if f["familia"] == fam]
        ax.scatter([f["entropia_bloques_real"] for f in sub], [f["boxdim_occ5_real"] for f in sub],
                   s=22, color=color, label=f"{fam} REAL", alpha=0.8)
        ax.scatter([f["entropia_bloques_null"] for f in sub], [f["boxdim_occ5_null"] for f in sub],
                   s=14, facecolors="none", edgecolors=color, label=f"{fam} barajado", alpha=0.6)
    ax.set_xlabel("entropía de configuración espacial"); ax.set_ylabel("dimensión box-counting (5% más alto)")
    ax.set_title("Entropía espacial vs. dimensión\n(lleno=real, hueco=barajado)", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[2]
    for fam, color in (("FASE", "tab:blue"), ("RD", "tab:red")):
        sub = [f for f in filas if f["familia"] == fam]
        ax.scatter([f["k_pico_real"] for f in sub],
                   [max(f["prominencia_pico_real"], 1e-3) for f in sub], s=22, color=color, label=f"{fam} REAL")
        ax.scatter([f["k_pico_null"] for f in sub],
                   [max(f["prominencia_pico_null"], 1e-3) for f in sub], s=14, facecolors="none",
                   edgecolors=color, label=f"{fam} barajado")
    ax.set_yscale("log"); ax.set_xlabel("k del pico de P(k)"); ax.set_ylabel("prominencia del pico (log)")
    ax.set_title("¿Hay una escala característica?\n(prominencia = pico / fondo)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{_HERE}/{prefijo}_metricas.png", dpi=130)
    plt.close(fig)
    print(f"  -> {prefijo}_campos.png, {prefijo}_espectros.png, {prefijo}_metricas.png")


def p(salida, texto):
    print(texto)
    salida.append(texto)


def main(prefijo="cs090_fase6_o2d_2d"):
    filas = cargar(prefijo)
    salida = []
    p(salida, f"O2-D — análisis de {len(filas)} corridas ({prefijo})")
    res_null = control_null(filas, salida)
    percolacion_y_criterios(filas, salida)
    persistencia_temporal(filas, salida)

    with open(f"{_HERE}/{prefijo}_tests.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metrica", "media_real", "media_null", "n_mayor", "n",
                                            "p", "delta", "separa"])
        w.writeheader()
        for k, v in res_null.items():
            w.writerow(dict(metrica=k, **v))
    print(f"\n  -> {prefijo}_tests.csv")
    figuras(filas, prefijo)
    return filas


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cs090_fase6_o2d_2d")
