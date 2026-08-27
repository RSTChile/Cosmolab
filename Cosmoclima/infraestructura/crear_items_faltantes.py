"""
ONCE ÍTEMS PARA LOS CATASTROS QUE NO TENÍAN DÓNDE ENTRAR
==========================================================

INSTRUCCIÓN (Alexis, 26-ago-2026): «crea las categorías respectivas y los
agregamos».

★ EL PROBLEMA QUE CIERRAN
---------------------------
Al integrar el barrido quedaron casi 22.000 activos verificados y
georreferenciados sin poder entrar, **no por falta de dato sino porque la Matriz
no contempla el activo**:

    16.126  jardines infantiles     MINEDUC + JUNJI + Integra
     4.503  farmacias comunitarias  el ítem 272 es «Farmacias HOSPITALARIAS»
     1.219  salud, tipo sin ítem    dentales, diálisis, vacunatorios, COSAM…
        82  recintos penitenciarios
        47  Servicio Médico Legal

★★ EL DE LOS JARDINES ES EL MÁS GRAVE, Y CONVIENE DECIR POR QUÉ
-----------------------------------------------------------------
Educación enumera escuelas primarias, secundarias, universidades, institutos y
centros de formación técnica, y no tenía nivel parvulario. Son 16.126
establecimientos con niños de 0 a 6 años —los únicos usuarios de infraestructura
crítica que **no pueden evacuar solos**— invisibles para el instrumento.

Por eso el ítem 853 se aparta de su hermano el 441 en un punto: VT sube de 0,50
a 0,70. No es un juicio estético: la sala cuna tiene una restricción operativa
que la escuela básica no tiene.

★ Y DOS QUE NO NECESITAN ÍTEM NUEVO
-------------------------------------
    Consultorio General Rural (CGR)     → 266 · Clínicas Rurales
    Centro de Referencia de Salud (CRS) → 267 · Centros de Atención Primaria
Ya existen y describen exactamente eso. Crear ítems nuevos ahí sería duplicar.

⚠️ El 863 es un CAJÓN DECLARADO —unidades móviles, policlínicos de funcionarios,
PRAIS, direcciones de servicio— y se dice que lo es. Lo que no se hace es
repartir esos 153 a ojo entre los ítems nuevos para que el número quede redondo.

USO
---
    ../.venv-esa/bin/python crear_items_faltantes.py
"""

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import micr  # noqa: E402

MATRIZ = AQUI / "web" / "publico" / "datos" / "matriz.json"

# (n, elemento, sector, hereda_de, {campo: valor} sólo donde se aparta, motivo)
NUEVOS = [
    (853, "Jardines Infantiles y Salas Cuna", "Educación", 441, {"VT": 0.70},
     "VT 0,70 frente al 0,50 de la escuela básica: los niños de 0 a 6 años no "
     "pueden evacuar solos, que es una restricción operativa real y no una "
     "diferencia de importancia"),
    (854, "Farmacias Comunitarias", "Salud", 272, {},
     "hereda de Farmacias Hospitalarias: mismo activo, distinta red"),
    (855, "Clínicas Dentales", "Salud", 267, {"FEN": "Media"},
     "FEN Media: atención programable, su interrupción no es inmediata"),
    (856, "Clínicas y Centros Médicos Privados", "Salud", 267, {},
     "hereda de Centros de Atención Primaria: misma función asistencial"),
    (857, "Centros Comunitarios de Salud Mental (COSAM)", "Salud", 267, {},
     "hereda de Centros de Atención Primaria"),
    (858, "Centros de Diálisis", "Salud", 267, {"VT": 0.95},
     "VT 0,95, el más alto de Salud: un paciente en diálisis no puede esperar "
     "más de unos días, así que la interrupción es letal y no diferible"),
    (859, "Vacunatorios", "Salud", 267, {"VT": 0.60},
     "VT 0,60: depende de cadena de frío, pero la atención sí es aplazable"),
    (860, "Salas de Toma de Muestras", "Salud", 270, {"VT": 0.60},
     "VT 0,60 frente al 0,90 del laboratorio: la sala toma la muestra, el "
     "laboratorio la procesa; sólo el segundo es insustituible"),
    (861, "Servicios Médico Legales (Forense)", "Salud", 270, {},
     "hereda de Laboratorios Clínicos: misma naturaleza de instalación"),
    (862, "Recintos Penitenciarios", "Seguridad", 353, {"VT": 0.90},
     "VT 0,90 frente al 0,70 del cuartel: la población recluida no puede "
     "desplazarse por sí misma y no hay recinto alternativo al que trasladarla"),
    (863, "Otros Establecimientos de Salud", "Salud", 267, {"FEN": "Media"},
     "CAJÓN DECLARADO: unidades móviles, policlínicos de funcionarios, PRAIS y "
     "direcciones de servicio. Agrupa lo que no sostiene categoría propia, y se "
     "dice que lo es en vez de repartirlo a ojo"),
]


def main():
    m = json.loads(MATRIZ.read_text(encoding="utf-8"))
    por_n = {int(i["n"]): i for i in m["items"]}
    cortes = m.get("cortes", {})
    creados = 0

    for n, elemento, sector, padre, aparta, motivo in NUEVOS:
        if n in por_n:
            print(f"  {n} ya existe, se omite")
            continue
        p = por_n.get(padre)
        if not p:
            print(f"  ⚠️ no existe el ítem padre {padre}, se omite {n}")
            continue
        v = {k: p[k] for k in ("FEN", "FANC", "IB", "VT")}
        v.update(aparta)
        f = micr.fvt(micr.fen_num(v["FEN"]), micr.fanc_num(v["FANC"]), v["VT"])
        pf = micr.pf(v["IB"], f)
        item = {
            "n": n, "elemento": elemento, "sector": sector, **v,
            "FEN_n": round(micr.n01(v["FEN"]), 6),
            "FANC_n": round(micr.n01(v["FANC"]), 6),
            "FVT": round(f, 6), "PF": round(pf, 6), "IRMD": micr.irmd(pf),
            "Pev": round(micr.pev(v["IB"], micr.fanc_num(v["FANC"]), f), 6),
            "Peh": round(micr.peh(micr.fanc_num(v["FANC"]), v["VT"], f), 6),
            "Pen": round(micr.pen(micr.fen_num(v["FEN"]), v["IB"], f), 6),
            "art": p.get("art", "52"),
            "motivo": (f"ítem nuevo 26-ago-2026 · hereda del {padre} "
                       f"«{p['elemento']}» · {motivo}"),
        }
        for k, campo in (("Pev", "Pev_b"), ("Peh", "Peh_b"), ("Pen", "Pen_b")):
            item[campo] = (micr.banda(item[k], cortes[k]) if cortes.get(k)
                           else p.get(campo))
        m["items"].append(item)
        creados += 1
        ap = " · se aparta: " + ", ".join(f"{k}={x}" for k, x in aparta.items()) if aparta else ""
        print(f"\n  ★ {n} · {elemento}  [{sector}]")
        print(f"      hereda del {padre}{ap}")
        print(f"      FEN {v['FEN']} · FANC {v['FANC']} · IB {v['IB']} · "
              f"VT {v['VT']} → PF {pf:.4f} · IRMD {micr.irmd(pf)}")

    m["items"].sort(key=lambda i: int(i["n"]))
    MATRIZ.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    print(f"\n  creados: {creados} · la Matriz queda con {len(m['items'])} ítems")
    return 0


if __name__ == "__main__":
    print("=" * 78)
    print("ÍTEMS NUEVOS · para los catastros que no tenían dónde entrar")
    print("=" * 78)
    sys.exit(main())
