"""
cs090_fase6_reanalisis_40pares_corregido.py — FASE V-B re-analizada con las ETIQUETAS corregidas.
==================================================================================================

DE QUÉ SE TRATA
---------------
Los 40 pares de Fase V-B (`cs090_fase5b_TOTAL_40pares.csv`) contrastan, dentro de cada par, una regla
etiquetada **Clase I** contra una etiquetada **Clase III**, emparejadas por K y kcap, y comparan lo que
hizo Phantom con cada una (fracción de masa en sumideros y κ_V). El resultado publicado
(`FASE5B_escala_40pares_CS.md`) es 31/40 signos a favor de Clase III, p=0.00068 por test de signos y
p=0.00001 por Wilcoxon.

La adopción del diámetro corregido (`cs090_diam_corregido.py`) puede cambiar la ETIQUETA de algunas de
esas 80 reglas. Si en un par la regla que se creía "Clase I" resulta ser Clase III con la medición
corregida, ese par deja de ser un contraste I-vs-III: pasa a ser III-vs-III, y no puede contarse como
evidencia a favor ni en contra. **La física no cambió** (los sumideros de Phantom son los mismos y no se
vuelve a correr nada): lo único que cambia es qué está comparando cada par.

QUÉ HACE ESTE SCRIPT
--------------------
1. Lee las clases corregidas de `cs090_fase6_remedicion_430.csv` (una fila por seed).
2. Re-etiqueta las 80 filas de `cs090_fase5b_TOTAL_40pares.csv` por `seed`.
3. Clasifica cada par en: contraste VÁLIDO (una punta Clase I, la otra Clase III, con la etiqueta
   corregida), contraste ROTO (las dos puntas quedan en la misma clase), o contraste INVERTIDO (la que
   hacía de "I" pasó a ser la de clase más alta y viceversa — se documenta aparte, no se re-orienta a
   mano).
4. Rehace los tests de signos y Wilcoxon con LOS MISMOS métodos de
   `cs090_fase5b_estadistica_40pares.py` (se importan sus funciones, no se re-escriben) sobre:
   (a) los 40 pares tal como se publicaron, (b) sólo los pares que siguen siendo contraste válido.

No corre Phantom. No toca ningún script anterior. No declara veredicto.
"""
from __future__ import annotations
import csv
import sys
import collections

HERE = "/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0, HERE)

from cs090_fase5b_estadistica_40pares import test_signos, test_wilcoxon

TOTAL40 = f"{HERE}/cs090_fase5b_TOTAL_40pares.csv"
REMEDIDO = f"{HERE}/cs090_fase6_remedicion_430.csv"


def orden_clase(c):
    """Orden 'de menos a más estructura' para poder decir si un par sigue siendo un contraste."""
    return {"I": 0, "intermedio (sin clase clara)": 1, "II": 2, "III": 3, "IV": 4}.get(c, -1)


def main():
    corr = {}
    with open(REMEDIDO) as f:
        for r in csv.DictReader(f):
            corr[int(r["seed"])] = r

    filas = list(csv.DictReader(open(TOTAL40)))
    por_par = collections.defaultdict(dict)
    for r in filas:
        por_par[r["par"]][r["rol"]] = r

    sin_remedir = [r["seed"] for r in filas if int(r["seed"]) not in corr]
    print(f"[cobertura] filas de los 40 pares: {len(filas)}; seeds sin re-medir: {len(sin_remedir)}")

    resumen, cambios = [], []
    for par, roles in sorted(por_par.items()):
        if "I" not in roles or "III" not in roles:
            continue
        fI, fIII = roles["I"], roles["III"]
        sI, sIII = int(fI["seed"]), int(fIII["seed"])
        cI_v, cIII_v = fI["clase"], fIII["clase"]
        cI_c = corr[sI]["clase_corregida"] if sI in corr else cI_v
        cIII_c = corr[sIII]["clase_corregida"] if sIII in corr else cIII_v
        if cI_c != cI_v:
            cambios.append((par, "brazo I", fI["rule_id"], cI_v, cI_c))
        if cIII_c != cIII_v:
            cambios.append((par, "brazo III", fIII["rule_id"], cIII_v, cIII_c))

        if orden_clase(cI_c) < orden_clase(cIII_c):
            estado = "valido"
        elif orden_clase(cI_c) == orden_clase(cIII_c):
            estado = "roto_misma_clase"
        else:
            estado = "invertido"

        d_frac = float(fIII["fraccion_masa_en_sumideros"]) - float(fI["fraccion_masa_en_sumideros"])
        kvI, kvIII = fI["kappa_v_agregado"], fIII["kappa_v_agregado"]
        d_kv = (float(kvIII) - float(kvI)) if kvI not in ("", "None") and kvIII not in ("", "None") else None
        resumen.append(dict(par=par, rule_I=fI["rule_id"], rule_III=fIII["rule_id"],
                            seed_I=sI, seed_III=sIII,
                            clase_I_vieja=cI_v, clase_I_corregida=cI_c,
                            clase_III_vieja=cIII_v, clase_III_corregida=cIII_c,
                            estado_contraste=estado, match_exacto=fI["match_exacto_K_kcap"],
                            d_frac=d_frac, d_kv=d_kv))

    print(f"\n[re-etiquetado] reglas de los 40 pares que cambian de clase: {len(cambios)}")
    for c in cambios:
        print(f"   {c[0]:<32} {c[1]:<10} {c[2]:<26} {c[3]}  ->  {c[4]}")

    cnt = collections.Counter(r["estado_contraste"] for r in resumen)
    print(f"\n[estado de los pares tras re-etiquetar] {dict(cnt)}")
    for r in resumen:
        if r["estado_contraste"] != "valido":
            print(f"   {r['estado_contraste']:<18} {r['par']:<32} "
                  f"I:{r['clase_I_vieja']}->{r['clase_I_corregida']}   "
                  f"III:{r['clase_III_vieja']}->{r['clase_III_corregida']}   d_frac={r['d_frac']:+.4f}")

    validos = [r for r in resumen if r["estado_contraste"] == "valido"]

    print("\n" + "=" * 92)
    print("A) COMO SE PUBLICÓ — los 40 pares con las etiquetas VIEJAS")
    print("=" * 92)
    test_signos([r["d_frac"] for r in resumen], "fracción de masa, 40 pares")
    test_wilcoxon([r["d_frac"] for r in resumen], "fracción de masa, 40 pares")
    dkv = [r["d_kv"] for r in resumen if r["d_kv"] is not None]
    test_signos(dkv, "kappa_V, 40 pares")
    test_wilcoxon(dkv, "kappa_V, 40 pares")

    print("\n" + "=" * 92)
    print(f"B) SÓLO LOS PARES QUE SIGUEN SIENDO CONTRASTE I-vs-III CON LA ETIQUETA CORREGIDA (n={len(validos)})")
    print("=" * 92)
    if validos:
        test_signos([r["d_frac"] for r in validos], "fracción de masa, pares válidos")
        test_wilcoxon([r["d_frac"] for r in validos], "fracción de masa, pares válidos")
        dkv2 = [r["d_kv"] for r in validos if r["d_kv"] is not None]
        test_signos(dkv2, "kappa_V, pares válidos")
        test_wilcoxon(dkv2, "kappa_V, pares válidos")

    with open(f"{HERE}/cs090_fase6_reanalisis_40pares_corregido.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        for r in resumen:
            w.writerow(r)
    print(f"\n[csv] cs090_fase6_reanalisis_40pares_corregido.csv ({len(resumen)} pares)")


if __name__ == "__main__":
    main()
