"""
ALINEAR LOS ÍTEMS AGRÍCOLAS CON EL CATASTRO QUE EXISTE
========================================================

INSTRUCCIÓN (Alexis, 26-ago-2026): «los campos de cultivo son importantes, lo
que yo haría es cambiar "Campos de Cultivo (Granos)" por "Terreno de Uso
Agrícola" [...] alineados con el catastro, donde entrarían los polígonos de
CONAF. Nada dice que no podamos adaptar la Matriz a las categorías existentes.»

★ EL PRINCIPIO, QUE ES EL DEL PROYECTO
----------------------------------------
La Matriz separa «Campos de Cultivo (Granos)» de «Campos de Cultivo (Frutas y
Verduras)». El catastro nacional de CONAF no usa esa división: usa
**Rotación Cultivo-Pradera** y **Terreno de Uso Agrícola**. Por eso sus 74.981
polígonos no podían entrar en ningún ítem.

Adaptar el ítem al catastro —en vez de forzar el catastro al ítem— es
exactamente lo que ya se hizo en Industrial con las divisiones CIIU.

★★ UNA CORRECCIÓN A LA PROPUESTA, Y POR QUÉ
---------------------------------------------
La instrucción pedía además renombrar el 398 («Frutas y Verduras») como
«Rotación Cultivo-Pradera». Eso **no se hace**, porque el 398 ya está poblado
con los **94.731 predios del Catastro Frutícola de CIREN**, que son literalmente
frutales: renombrarlo dejaría 94.731 predios de manzanos y viñas bajo una
etiqueta que dice otra cosa.

Se consigue lo mismo sin ese efecto:

    397  «Campos de Cultivo (Granos)»            → «Terreno de Uso Agrícola»
         renombrado. Recibe los 28.131 polígonos de esa clase en CONAF.

    398  «Campos de Cultivo (Frutas y Verduras)» → SE MANTIENE
         ya tiene los 94.731 predios frutícolas de CIREN, que sí son frutas.

    852  ★ ÍTEM NUEVO · «Rotación Cultivo-Pradera»
         recibe los 46.850 polígonos de esa clase en CONAF.

Así entran los 74.981 completos, cada uno bajo el nombre que le da el catastro.

⚠️ El nombre original del 397 queda registrado en el propio ítem, en
`elemento_original`, para que el cambio sea reversible y auditable contra el
Word oficial.

USO
---
    ../.venv-esa/bin/python alinear_items_agricolas.py
"""

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import micr  # noqa: E402

MATRIZ = AQUI / "web" / "publico" / "datos" / "matriz.json"

NUEVO_397 = "Terreno de Uso Agrícola"
ITEM_852 = (852, "Rotación Cultivo-Pradera")


def main():
    m = json.loads(MATRIZ.read_text(encoding="utf-8"))
    por_n = {int(i["n"]): i for i in m["items"]}
    cortes = m.get("cortes", {})

    # ── 1 · renombrar el 397 ────────────────────────────────────────────────
    i397 = por_n.get(397)
    if not i397:
        print("  ⚠️ no existe el ítem 397")
        return 1
    if i397["elemento"] != NUEVO_397:
        i397["elemento_original"] = i397.get("elemento_original", i397["elemento"])
        i397["elemento"] = NUEVO_397
        i397["motivo"] = (
            "renombrado 26-ago-2026 para alinearlo con el catastro nacional de "
            f"usos de tierra de CONAF. Nombre en el Word oficial: "
            f"«{i397['elemento_original']}». El catastro no distingue granos "
            "de otros cultivos, así que el ítem original no era poblable.")
        print(f"  ★ 397 renombrado")
        print(f"      antes : {i397['elemento_original']}")
        print(f"      ahora : {i397['elemento']}")
    else:
        print("  397 ya estaba renombrado")

    # ── 2 · el 398 se deja intacto, y se dice por qué ───────────────────────
    i398 = por_n.get(398)
    print(f"\n  · 398 «{i398['elemento']}» SE MANTIENE")
    print(f"      ya tiene 94.731 predios del Catastro Frutícola de CIREN, que")
    print(f"      son frutales: renombrarlo los dejaría mal etiquetados.")

    # ── 3 · crear el 852 ────────────────────────────────────────────────────
    n, elemento = ITEM_852
    if n in por_n:
        print(f"\n  {n} ya existe, se omite")
    else:
        # Hereda de su hermano 397: es la misma clase de activo —suelo agrícola
        # a la intemperie— y no hay razón medida para distinguirlos.
        fen, fanc = i397["FEN"], i397["FANC"]
        ib, vt = i397["IB"], i397["VT"]
        f = micr.fvt(micr.fen_num(fen), micr.fanc_num(fanc), vt)
        pf = micr.pf(ib, f)
        nuevo = {
            "n": n, "elemento": elemento, "sector": "Alimentario",
            "FEN": fen, "FANC": fanc, "IB": ib, "VT": vt,
            "FEN_n": round(micr.n01(fen), 6), "FANC_n": round(micr.n01(fanc), 6),
            "FVT": round(f, 6), "PF": round(pf, 6), "IRMD": micr.irmd(pf),
            "Pev": round(micr.pev(ib, micr.fanc_num(fanc), f), 6),
            "Peh": round(micr.peh(micr.fanc_num(fanc), vt, f), 6),
            "Pen": round(micr.pen(micr.fen_num(fen), ib, f), 6),
            "art": i397.get("art", "52"),
            "motivo": ("ítem nuevo 26-ago-2026 · clase del catastro nacional de "
                       "usos de tierra de CONAF que la Matriz no contemplaba. "
                       "Hereda FEN/FANC/IB/VT del 397: mismo tipo de activo, "
                       "suelo agrícola a la intemperie."),
        }
        for k, campo in (("Pev", "Pev_b"), ("Peh", "Peh_b"), ("Pen", "Pen_b")):
            nuevo[campo] = (micr.banda(nuevo[k], cortes[k])
                            if cortes.get(k) else i397.get(campo))
        m["items"].append(nuevo)
        print(f"\n  ★ {n} · {elemento}")
        print(f"      FEN {fen} · FANC {fanc} · IB {ib} · VT {vt} "
              f"(heredados del 397)")
        print(f"      → FVT {f:.4f} · PF {pf:.4f} · IRMD {micr.irmd(pf)}")

    m["items"].sort(key=lambda i: int(i["n"]))
    MATRIZ.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    print(f"\n  la Matriz queda con {len(m['items'])} ítems")
    return 0


if __name__ == "__main__":
    print("=" * 78)
    print("ÍTEMS AGRÍCOLAS · alinearlos con el catastro de CONAF")
    print("=" * 78)
    sys.exit(main())
