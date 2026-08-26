"""
CINCO ÍTEMS NUEVOS PARA INDUSTRIAL · el sector que se quedó sin categorías
===========================================================================

INSTRUCCIÓN (Alexis, 25-ago-2026): «para Industrial crea las categorías
generales que sean necesarias».

★ EL PROBLEMA, QUE ES AL REVÉS DEL HABITUAL
---------------------------------------------
Hasta ahora este proyecto venía encontrando que la MICR **separa más fino de lo
que registra cualquier catastro**: pide carbón/gas/diésel y la fuente no lo dice;
pide pozo artesiano o no y la DGA no lo anota.

En Industrial pasa lo contrario. Sus 44 ítems son tan específicos —Maquinaria,
Vehículos, Componentes Electrónicos, Equipos Médicos, Armamento— que **la
industria real del país no tiene dónde entrar**. Hay 1.514 establecimientos
manufactureros con coordenada en el RETC y un aserradero, una cementera o una
planta de celulosa no caben en ningún ítem.

★★ LAS CATEGORÍAS SALEN DEL CIIU, NO DE MI CRITERIO
-----------------------------------------------------
Inventar cinco nombres a ojo sería exactamente lo que este trabajo no hace. Las
categorías son **divisiones de la CIIU rev.4**, la clasificación industrial
internacional que el propio RETC usa para tipificar cada establecimiento. Así la
frontera entre un ítem y otro no la decide nadie de este lado: viene con el dato.

Medido sobre los 1.514 manufactureros del RETC (sección C):

    div 10-12  alimentos, bebidas, tabaco   714  → NO se crea ítem: van al
                                                   sector Alimentario (407/408/409/440)
    div 20-21  químicos y farmacéuticos     142  → NO se crea: sector Químico (660)
    div 19     refinación de petróleo        16  → NO se crea: Energía (111)
    div 23     minerales no metálicos       147  → ★ ÍTEM NUEVO 847
    div 16     madera y aserraderos         102  → ★ ÍTEM NUEVO 848
    div 24-25  metalurgia y metalmecánica    92  → ★ ÍTEM NUEVO 849
    div 17     papel y celulosa              54  → ★ ÍTEM NUEVO 850
    el resto   plástico, textil, muebles…   247  → ★ ÍTEM NUEVO 851 (general)

★★★ POR QUÉ CADA UNA ES UN ÍTEM Y NO UNA SOLA CATEGORÍA GENÉRICA
------------------------------------------------------------------
Porque **ceden por cosas distintas**, que es el criterio que este proyecto ya
aplicó al separar la captación del estanque en el agua potable rural:

    aserradero          combustible acumulado a la intemperie → arde
    cementera           polvo y áridos → el agua lo arrastra y lo solidifica
    papel y celulosa    depende de agua de proceso en volumen → la sequía la para
    metalurgia          hornos y energía continua → el corte eléctrico la daña
    manufactura general sin modo de falla dominante declarable

El ítem 851 es un **cajón declarado**, no un descuido: agrupa lo que no tiene
volumen suficiente para sostener una categoría propia, y se dice que lo es.

★ LOS VALORES
---------------
FEN, FANC, IB y VT se heredan del ítem 704 (Fábricas de Maquinaria) salvo donde
hay razón escrita para apartarse. FVT, PF, IRMD, Pev, Peh y Pen NO se asignan:
los calcula `micr.py` con las fórmulas del Word oficial.

USO
---
    ../.venv-esa/bin/python crear_items_industrial.py
"""

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import micr  # noqa: E402

MATRIZ = AQUI / "web" / "publico" / "datos" / "matriz.json"

# (n, elemento, FEN, FANC, IB, VT, motivo de apartarse del 704, divisiones CIIU)
NUEVOS = [
    (847, "Plantas de Materiales de Construcción (Cemento, Cal y Áridos)",
     "Alta", "Baja", 0.90, 0.75,
     "IB 0,90 y VT 0,75: menos crítica que la maquinaria para la continuidad "
     "del país, pero su material a la intemperie es arrastrable por agua",
     ("23",)),
    (848, "Aserraderos y Plantas de Elaboración de Madera",
     "Alta", "Baja", 0.85, 0.65,
     "VT 0,65: es el ítem industrial con más combustible acumulado a la "
     "intemperie, y los incendios forestales son el evento que menos se absorbe "
     "del país (19,2 % medido sobre SENAPRED)",
     ("16",)),
    (849, "Plantas Metalúrgicas y Metalmecánicas",
     "Alta", "Baja", 0.90, 0.80,
     "hereda del 704: hornos y proceso continuo, depende de energía sin corte",
     ("24", "25")),
    (850, "Plantas de Papel y Celulosa",
     "Alta", "Baja", 0.90, 0.70,
     "VT 0,70: su insumo crítico es agua de proceso en volumen, así que la "
     "amenaza que la detiene es la sequía y no sólo el temporal",
     ("17",)),
    (851, "Plantas Manufactureras (General)",
     "Media", "Baja", 0.85, 0.75,
     "FEN Media: es un cajón declarado —plástico, textil, muebles, imprenta— "
     "sin un modo de falla dominante que justifique subirlo a Alta",
     ("13", "14", "15", "18", "22", "26", "27", "28", "29", "30", "31", "32", "33")),
]


def calculados(fen, fanc, ib, vt):
    f = micr.fvt(micr.fen_num(fen), micr.fanc_num(fanc), vt)
    p = micr.pf(ib, f)
    return {
        "FEN_n": round(micr.n01(fen), 6), "FANC_n": round(micr.n01(fanc), 6),
        "FVT": round(f, 6), "PF": round(p, 6), "IRMD": micr.irmd(p),
        "Pev": round(micr.pev(ib, micr.fanc_num(fanc), f), 6),
        "Peh": round(micr.peh(micr.fanc_num(fanc), vt, f), 6),
        "Pen": round(micr.pen(micr.fen_num(fen), ib, f), 6),
    }


def main():
    m = json.loads(MATRIZ.read_text(encoding="utf-8"))
    existentes = {int(i["n"]) for i in m["items"]}
    cortes = m.get("cortes", {})

    print(f"  ítems en la Matriz: {len(m['items'])}")
    creados = 0
    for n, elemento, fen, fanc, ib, vt, motivo, divs in NUEVOS:
        if n in existentes:
            print(f"  {n} ya existe, se omite")
            continue
        v = calculados(fen, fanc, ib, vt)
        item = {
            "n": n, "elemento": elemento, "sector": "Industrial",
            "FEN": fen, "FANC": fanc, "IB": ib, "VT": vt,
            **v,
            "Pev_b": micr.banda(v["Pev"], cortes.get("Pev")) if cortes.get("Pev") else None,
            "Peh_b": micr.banda(v["Peh"], cortes.get("Peh")) if cortes.get("Peh") else None,
            "Pen_b": micr.banda(v["Pen"], cortes.get("Pen")) if cortes.get("Pen") else None,
            "art": "52",
            "motivo": f"ítem nuevo 25-ago-2026 · CIIU rev.4 div {'+'.join(divs)} · {motivo}",
        }
        m["items"].append(item)
        creados += 1
        print(f"\n  ★ {n} · {elemento}")
        print(f"      CIIU div {'+'.join(divs)} · FEN {fen} · FANC {fanc} · "
              f"IB {ib} · VT {vt}")
        print(f"      → FVT {v['FVT']:.4f} · PF {v['PF']:.4f} · IRMD {v['IRMD']}")
        print(f"      {motivo}")

    m["items"].sort(key=lambda i: int(i["n"]))
    MATRIZ.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    print(f"\n  creados: {creados} · la Matriz queda con {len(m['items'])} ítems")
    print("  ⚠️ Son ítems NUEVOS, no del Word oficial. Van marcados con su motivo")
    print("     y su división CIIU para que se puedan discutir uno por uno.")
    return 0


if __name__ == "__main__":
    print("=" * 78)
    print("INDUSTRIAL · cinco categorías nuevas, derivadas de la CIIU rev.4")
    print("=" * 78)
    sys.exit(main())
