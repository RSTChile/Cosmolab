"""
EL FACTOR QUE FALTABA · cuánto absorbe el sistema antes de romperse
====================================================================

El ICSGS del MCSGS necesita cinco factores. El proyecto ya tiene cuatro medidos
—FCN, FSS, FAS y FPI— y el quinto, **FRC (resiliencia)**, seguía declarado a ojo.

★★ POR QUÉ ESTE ES EL FACTOR DELICADO Y NO UNO MÁS
----------------------------------------------------
La fórmula es:

    ICSGS = min(100, √(FCN × FSS × FAS × FPI) × (1/FRC) × 100)

FRC entra como **1/FRC**: no se suma, DIVIDE. Suponerlo mal no desplaza el
resultado, lo cambia de orden de magnitud. Por eso el proyecto prefirió no
publicar ningún ICSGS antes que publicarlo con este factor inventado.

★ LO QUE SÍ SE PUEDE MEDIR, Y CÓMO
------------------------------------
El registro de SENAPRED trae, para 50.457 emergencias, el daño **separado por
gravedad**:

    Viviendas Con Daño Menor    124.534
    Viviendas Con Daño Mayor     39.350
    Viviendas Destruidas         33.488

Eso permite medir algo que sí es observable: **de todo lo que se dañó, qué
fracción sobrevivió como daño reparable en vez de destrucción**.

    absorcion = (daño menor + daño mayor) / (daño menor + daño mayor + destruidas)

Un territorio donde el mismo tipo de evento deja casas rajadas en vez de casas
en el suelo está absorbiendo el golpe. Uno donde casi todo lo dañado termina
destruido, no.

Y como segunda señal, **cuántos organismos concurren** a responder —presente en
el 92 % de los eventos—, que es capacidad de respuesta efectivamente movilizada.

⚠️⚠️ ESTO NO ES EL FRC CANÓNICO, Y NO SE LE LLAMA ASÍ.
El MCSGS define FRC como capacidad de absorción del sistema; esto mide
**absorción observada del daño en viviendas**, que es una señal de lo mismo pero
no es lo mismo. Se publica con nombre propio —«factor de absorción observada»— y
se declara qué haría falta para aceptarlo como FRC. Ponerle la etiqueta FRC sería
cerrar la fórmula por decreto, que es justo lo que este proyecto no hace.

⚠️ El campo «Total Afectados» NO se usa: suma 342 millones de personas en un país
de 20, así que tiene valores corruptos que nadie ha limpiado.

USO
---
    ../.venv-esa/bin/python medir_absorcion.py
"""

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CRUDO = AQUI / "datos" / "crudo" / "senapred"
SALIDA = AQUI / "web" / "publico" / "datos" / "absorcion.json"

MIN_CASOS = 40          # menos que esto no sostiene una proporción
SEPARADORES = re.compile(r"[,;/]| - ")


def main():
    arch = glob.glob(str(CRUDO / "**" / "Eventos_Emergencia_2015_2024.xlsx"),
                     recursive=True)
    if not arch:
        print("  falta el registro de eventos de SENAPRED")
        return 1
    import openpyxl

    ws = openpyxl.load_workbook(arch[0], read_only=True)["Eventos_de_Emergencia_2015_2024"]
    it = ws.iter_rows(values_only=True)
    cab = list(next(it))
    ic = {str(c): i for i, c in enumerate(cab) if c}

    def num(f, k):
        v = f[ic[k]]
        return v if isinstance(v, (int, float)) and v >= 0 else 0

    por_tipo = defaultdict(lambda: {"menor": 0, "mayor": 0, "destr": 0,
                                    "eventos": 0, "con_dano": 0, "org": []})
    por_region = defaultdict(lambda: {"menor": 0, "mayor": 0, "destr": 0,
                                      "eventos": 0, "con_dano": 0, "org": []})
    organismos = Counter()
    por_region_tipo = defaultdict(lambda: {"menor": 0, "mayor": 0, "destr": 0,
                                           "con_dano": 0})
    n = 0

    for f in it:
        t = f[ic["Tipo Evento"]]
        r = f[ic["Región"]]
        if not t:
            continue
        n += 1
        me, ma, de = (num(f, "Viviendas Con Daño Menor"),
                      num(f, "Viviendas Con Daño Mayor"),
                      num(f, "Viviendas Destruidas"))
        v = f[ic["Organismos de Respuesta"]]
        cuantos = 0
        if isinstance(v, str) and v.strip():
            partes = [p.strip() for p in SEPARADORES.split(v) if p.strip()]
            cuantos = len(partes)
            for p in partes:
                organismos[p.upper()[:26]] += 1
        for destino, clave in ((por_tipo, str(t).strip()[:44]),
                               (por_region, str(r or "?").strip()[:34])):
            d = destino[clave]
            d["eventos"] += 1
            d["menor"] += me
            d["mayor"] += ma
            d["destr"] += de
            if me + ma + de:
                d["con_dano"] += 1
            if cuantos:
                d["org"].append(cuantos)
        rt = por_region_tipo[(str(r or "?").strip()[:34], str(t).strip()[:44])]
        rt["menor"] += me
        rt["mayor"] += ma
        rt["destr"] += de
        if me + ma + de:
            rt["con_dano"] += 1

    print(f"  emergencias leídas: {n:,}")

    def absorcion(d):
        tot = d["menor"] + d["mayor"] + d["destr"]
        return None if tot == 0 else (d["menor"] + d["mayor"]) / tot

    def resumir(fuente, etiqueta):
        filas = []
        for k, d in fuente.items():
            a = absorcion(d)
            if a is None or d["con_dano"] < MIN_CASOS:
                continue
            org = sorted(d["org"])
            filas.append({
                etiqueta: k, "eventos": d["eventos"], "con_dano": d["con_dano"],
                "viviendas": d["menor"] + d["mayor"] + d["destr"],
                "destruidas": d["destr"],
                "absorcion": round(a, 4),
                "organismos_mediana": org[len(org) // 2] if org else 0,
            })
        filas.sort(key=lambda x: x["absorcion"])
        return filas

    tipos = resumir(por_tipo, "tipo")
    regiones = resumir(por_region, "region")

    print(f"\n  ★ ABSORCIÓN POR TIPO DE EVENTO (de lo dañado, qué fracción NO se destruyó)")
    print(f"  {'tipo de evento':<40}{'viviendas':>10}{'absorción':>11}{'organismos':>11}")
    print("  " + "-" * 72)
    for x in tipos[:8]:
        print(f"  {x['tipo'][:39]:<40}{x['viviendas']:>10,}{100*x['absorcion']:>10.1f}%"
              f"{x['organismos_mediana']:>11}")
    if len(tipos) > 8:
        print(f"  {'…y los más absorbidos:':<40}")
        for x in tipos[-3:]:
            print(f"  {x['tipo'][:39]:<40}{x['viviendas']:>10,}{100*x['absorcion']:>10.1f}%"
                  f"{x['organismos_mediana']:>11}")

    print(f"\n  ★ ABSORCIÓN POR REGIÓN")
    print(f"  {'región':<36}{'viviendas':>10}{'absorción':>11}{'organismos':>11}")
    print("  " + "-" * 68)
    for x in regiones[:6]:
        print(f"  {x['region'][:35]:<36}{x['viviendas']:>10,}{100*x['absorcion']:>10.1f}%"
              f"{x['organismos_mediana']:>11}")
    print("  " + "·" * 68)
    for x in regiones[-3:]:
        print(f"  {x['region'][:35]:<36}{x['viviendas']:>10,}{100*x['absorcion']:>10.1f}%"
              f"{x['organismos_mediana']:>11}")

    # ── EL CONTROL: ¿es la región, o es el tipo de evento que le toca? ──────
    # Valparaíso sale con 33 % y Tarapacá con 97 %. Pero Valparaíso arde y
    # Tarapacá se inunda, y el fuego destruye mientras el agua moja. Si el orden
    # entre regiones se mantiene DENTRO de un mismo tipo de evento, la señal es
    # de la región; si se deshace, era composición y la lectura regional sobra.
    print(f"\n  ★★ CONTROL · la misma comparación DENTRO de un mismo tipo")
    control = {}
    for familia, filtro in (("incendios", lambda t: "incendio" in t.lower() or t.lower() == "forestal"),
                            ("agua", lambda t: any(k in t.lower() for k in
                                                   ("inunda", "anegam", "lluvia", "sistema frontal")))):
        acum = defaultdict(lambda: {"menor": 0, "mayor": 0, "destr": 0, "con_dano": 0})
        for (reg, tip), d in por_region_tipo.items():
            if not filtro(tip):
                continue
            a_ = acum[reg]
            for k in ("menor", "mayor", "destr", "con_dano"):
                a_[k] += d[k]
        filas = []
        for reg, d in acum.items():
            tot = d["menor"] + d["mayor"] + d["destr"]
            if tot == 0 or d["con_dano"] < MIN_CASOS:
                continue
            filas.append({"region": reg, "viviendas": tot,
                          "absorcion": round((d["menor"] + d["mayor"]) / tot, 4)})
        filas.sort(key=lambda x: x["absorcion"])
        control[familia] = filas
        print(f"\n     — sólo {familia} — ({len(filas)} regiones con muestra)")
        for x in filas[:3]:
            print(f"       {x['region'][:32]:<34}{x['viviendas']:>9,}{100*x['absorcion']:>9.1f}%")
        if len(filas) > 4:
            print(f"       {'…':<34}")
            for x in filas[-2:]:
                print(f"       {x['region'][:32]:<34}{x['viviendas']:>9,}{100*x['absorcion']:>9.1f}%")

    print(f"\n  organismos que más concurren: "
          f"{[o for o, _ in organismos.most_common(6)]}")

    doc = {
        "que_mide": ("De todas las viviendas dañadas, qué fracción quedó como "
                     "daño reparable en vez de destruida: "
                     "(menor + mayor) / (menor + mayor + destruidas)"),
        "advertencia": (
            "NO es el FRC del MCSGS y no se le llama así. El MCSGS define FRC "
            "como capacidad de absorción del sistema; esto mide absorción "
            "observada del daño en viviendas, que es una señal de lo mismo pero "
            "no es lo mismo. Para aceptarlo como FRC haría falta acordar que la "
            "vivienda representa al sistema, y eso es una decisión, no una "
            "medición."),
        "por_que_importa": (
            "FRC entra en el ICSGS como 1/FRC: divide en vez de sumarse, así que "
            "suponerlo cambia el índice de orden de magnitud. Es el único de los "
            "cinco factores que sigue sin medirse."),
        "no_usado": ("El campo «Total Afectados» suma 342 millones de personas en "
                     "un país de 20: tiene valores corruptos y no se usa."),
        "fuente": "SENAPRED · Eventos de Emergencia 2015-2024 (50.457 registros)",
        "min_casos": MIN_CASOS,
        "confundido_por_el_tipo": (
            "La lectura por región mezcla resiliencia con qué evento le toca a "
            "cada una: el fuego destruye y el agua moja. Por eso se publica "
            "también la comparación dentro de un mismo tipo, que es la que vale."),
        "por_tipo": tipos,
        "por_region": regiones,
        "control_por_familia": control,
    }
    SALIDA.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    print("=" * 78)
    print("FACTOR DE ABSORCIÓN OBSERVADA · el quinto factor, medido con reservas")
    print("=" * 78)
    sys.exit(main())
