"""
MATRIZ · los 846 ítems, con su procedencia
============================================

Empaqueta la MICR recalibrada para el navegador. Es el artefacto más pequeño y
el más consultado: alimenta la pestaña de consultas y da los valores de FEN con
que se calcula la fragilidad efectiva en la proyección.

★ SE INCLUYE DE DÓNDE SALE CADA VALOR, NO SÓLO EL VALOR
---------------------------------------------------------
El FANC no viaja como «Alta» a secas sino con el artículo del Protocolo I que lo
justifica, y el motivo en texto. La aplicación puede así responder «¿por qué este
ítem está en este nivel?» sin que nadie tenga que abrir el código.

Es la diferencia entre un número y un número defendible.

USO
---
    ../../.venv-esa/bin/python construir/matriz.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ))
import micr                                              # noqa: E402

DATOS = AQUI.parent / "publico" / "datos"
RECAL = RAIZ / "datos" / "micr_recalibrada.csv"
FANC = RAIZ / "datos" / "fanc_medido_4grados.csv"
SP = RAIZ / "datos" / "micr_sharepoint_ultimo.csv"
SALIDA = DATOS / "matriz.json"


def construir():
    porque = {}
    if FANC.exists():
        porque = {int(x["n"]): (x["articulo"], x["motivo"])
                  for x in csv.DictReader(FANC.open(encoding="utf-8"))}

    filas = []
    for x in csv.DictReader(RECAL.open(encoding="utf-8")):
        n = int(x["n"])
        art, mot = porque.get(n, ("", ""))
        filas.append({
            "n": n, "elemento": x["elemento"], "sector": x["Sector"],
            "FEN": x["FEN"], "FANC": x["FANC"],
            "FEN_n": round(float(x["FEN_n"]), 6),
            "FANC_n": round(float(x["FANC_n"]), 6),
            "IB": float(x["IB"]), "VT": float(x["VT"]),
            "FVT": round(float(x["FVT"]), 6), "PF": round(float(x["PF"]), 6),
            "IRMD": x["IRMD"],
            "Pev": round(float(x["Pev"]), 6), "Peh": round(float(x["Peh"]), 6),
            "Pen": round(float(x["Pen"]), 6),
            "Pev_b": x["Pev_banda"], "Peh_b": x["Peh_banda"], "Pen_b": x["Pen_banda"],
            "art": art, "motivo": mot,
        })

    # ★ El ítem 846 nació después del recalibrado y vive sólo en SharePoint.
    # Se trae de allí y se le calculan las derivadas con el MISMO micr.py, para
    # que no haya dos caminos de cálculo.
    if SP.exists() and not any(f["n"] == 846 for f in filas):
        for x in csv.DictReader(SP.open(encoding="utf-8")):
            if x["n"] and int(float(x["n"])) == 846:
                fen, fanc = micr.n01(x["FEN"]), micr.n01(x["FANC"])
                ib, vt = float(x["IB"]), float(x["VT"])
                fvt = micr.fvt_n(fen, fanc, vt)
                pf = ib * fvt
                pev, peh, pen = (micr.pev_n(ib, fanc, fvt),
                                 micr.peh_n(fanc, ib, vt, fvt),
                                 micr.pen_n(fen, ib, fvt))
                filas.append({
                    "n": 846, "elemento": x["elemento"], "sector": x["Sector"],
                    "FEN": x["FEN"], "FANC": x["FANC"],
                    "FEN_n": round(fen, 6), "FANC_n": round(fanc, 6),
                    "IB": ib, "VT": vt, "FVT": round(fvt, 6), "PF": round(pf, 6),
                    "IRMD": micr.irmd_n(pf),
                    "Pev": round(pev, 6), "Peh": round(peh, 6), "Pen": round(pen, 6),
                    "Pev_b": micr.banda(pev, micr.CORTES_PEV_2026),
                    "Peh_b": micr.banda(peh, micr.CORTES_PEH_2026),
                    "Pen_b": micr.banda(pen, micr.CORTES_PEN_2026),
                    "art": "52", "motivo": "bien civil · componente de planta CSP",
                })

    filas.sort(key=lambda f: f["n"])
    DATOS.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps({
        "items": filas,
        "cortes": {"Pev": micr.CORTES_PEV_2026, "Peh": micr.CORTES_PEH_2026,
                   "Pen": micr.CORTES_PEN_2026, "IRMD": micr.CORTES_IRMD_2026},
        "pesos": micr.PESOS,
        "nivel": micr.NIVEL,
    }, ensure_ascii=False), encoding="utf-8")

    print(f"  ítems  : {len(filas)}")
    print(f"  sectores: {len({f['sector'] for f in filas})}")
    for k in ("Pev_b", "Peh_b", "Pen_b"):
        c = Counter(f[k] for f in filas)
        print(f"  {k[:3]}  " + " · ".join(f"{b}:{c.get(b,0)}" for b in
              ("Muy Alta", "Alta", "Media", "Baja", "Muy Baja")))
    print(f"  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return {"items": len(filas)}


if __name__ == "__main__":
    print("=" * 70); print("MATRIZ · 846 ítems"); print("=" * 70)
    construir(); sys.exit(0)
