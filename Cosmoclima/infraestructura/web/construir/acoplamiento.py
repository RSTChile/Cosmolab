"""
ACOPLAMIENTO MEDIDO · qué falla junto con qué, y con cuánto retardo
====================================================================

El ICSGS del MCSGS pide cinco factores. Este proyecto tenía dos con dato —FCN y
FSS— y tres declarados a ojo. Dos de esos tres se pueden **medir**, y aquí se
miden:

    FAS  acoplamiento   ¿qué tipos de evento ocurren juntos, en el mismo lugar
                        y el mismo día, más de lo que el azar explicaría?
    FPI  propagación    cuando ocurren en secuencia, ¿con cuánto retardo?

★ LA FUENTE, QUE YA ESTABA EN DISCO
-------------------------------------
`Eventos_Emergencia_2015_2024.xlsx` de SENAPRED: **50.457 eventos**, todos con
fecha y comuna, en 171 tipos. Con eso, «acoplamiento» deja de ser un juicio y
pasa a ser una frecuencia observada.

★★ CÓMO SE MIDE, Y POR QUÉ NO BASTA CONTAR COINCIDENCIAS
----------------------------------------------------------
Dos tipos frecuentes coinciden a menudo por pura frecuencia: si los incendios
estructurales son el 10 % de los eventos y los cortes eléctricos el 11 %, se van
a encontrar seguido sin que uno cause al otro.

Por eso se usa el **lift**: cuántas veces más co-ocurren de lo que ocurriría si
fueran independientes.

    lift = P(A y B juntos) / (P(A) × P(B))

Un lift de 1 es indistinguible del azar. Un lift de 5 dice que aparecen juntos
cinco veces más de lo esperable, y **eso es acoplamiento**.

⚠️ ACOPLAMIENTO NO ES CAUSALIDAD. Que dos tipos co-ocurran puede deberse a que
uno provoca al otro, o a que ambos tienen una causa común —el mismo temporal—.
El dato no distingue. Lo que sí permite afirmar es que **fallan juntos**, que es
justo lo que el ICSGS necesita: si A y B están acoplados, dañar A ya no deja a B
intacto.

USO
---
    ../../.venv-esa/bin/python construir/acoplamiento.py
"""

import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
CRUDO = RAIZ / "datos" / "crudo" / "senapred"
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "acoplamiento.json"

MIN_CASOS = 30          # menos que esto no sostiene una frecuencia
VENTANA_DIAS = 3        # para la secuencia: qué sigue a qué


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

    ev, saltadas = [], []
    for f in it:
        t = f[ic["Tipo Evento"]]
        c = f[ic["Comuna"]]
        d = f[ic["Fecha Inicio"]]
        if not (t and c and d):
            continue
        # ⚠️ Una fila de las 50.457 trae la fórmula rota «=$A$45072» en vez de
        #    fecha. Es un error de la planilla de origen; se salta.
        if not hasattr(d, "date"):
            saltadas.append(d)
            continue
        danio = 0
        for k in ("Total Fallecidos", "Total Damnificados", "Viviendas Destruidas"):
            v = f[ic[k]]
            if isinstance(v, (int, float)):
                danio += v
        ev.append((d.date(), str(c), str(t).strip()[:44], danio))
    print(f"  eventos: {len(ev):,}")
    if saltadas:
        print(f"  ⚠️ filas con fecha ilegible en la fuente: {len(saltadas)} "
              f"({saltadas[0]!r})")

    tipos = Counter(t for _, _, t, _ in ev)
    frecuentes = {t for t, n in tipos.items() if n >= MIN_CASOS}
    print(f"  tipos con {MIN_CASOS}+ casos: {len(frecuentes)} de {len(tipos)}")

    # ── co-ocurrencia: mismo día, misma comuna ──────────────────────────────
    dia_comuna = defaultdict(set)
    for d, c, t, _ in ev:
        if t in frecuentes:
            dia_comuna[(d, c)].add(t)

    juntos = Counter()
    solos = Counter()
    for ts in dia_comuna.values():
        for t in ts:
            solos[t] += 1
        l = sorted(ts)
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                juntos[(l[i], l[j])] += 1
    total_dc = len(dia_comuna)
    print(f"  combinaciones día-comuna: {total_dc:,}")

    pares = []
    for (a, b), n in juntos.items():
        if n < 10:
            continue
        pa, pb = solos[a] / total_dc, solos[b] / total_dc
        esperado = pa * pb * total_dc
        if esperado < 1:
            continue
        pares.append({"a": a, "b": b, "juntos": n,
                      "lift": round(n / esperado, 2)})
    pares.sort(key=lambda p: -p["lift"])

    print(f"\n  ★ PARES QUE FALLAN JUNTOS MÁS DE LO ESPERABLE (lift alto)")
    print(f"  {'tipo A':<32}{'tipo B':<32}{'juntos':>7}{'lift':>7}")
    print("  " + "-" * 80)
    for p in pares[:14]:
        print(f"  {p['a'][:31]:<32}{p['b'][:31]:<32}{p['juntos']:>7}{p['lift']:>7.1f}")

    # ── secuencia: qué sigue a qué, y con cuánto retardo ────────────────────
    porcom = defaultdict(list)
    for d, c, t, _ in ev:
        if t in frecuentes:
            porcom[c].append((d, t))
    for c in porcom:
        porcom[c].sort()

    sigue = defaultdict(list)
    for c, lista in porcom.items():
        for i, (d1, t1) in enumerate(lista):
            for d2, t2 in lista[i + 1:]:
                dd = (d2 - d1).days
                if dd <= 0:
                    continue
                if dd > VENTANA_DIAS:
                    break
                if t1 != t2:
                    sigue[(t1, t2)].append(dd)

    seq = [{"a": a, "b": b, "n": len(v),
            "retardo_mediano": sorted(v)[len(v) // 2]}
           for (a, b), v in sigue.items() if len(v) >= 30]
    seq.sort(key=lambda s: -s["n"])

    print(f"\n  ★ QUÉ SIGUE A QUÉ, dentro de {VENTANA_DIAS} días y en la misma comuna")
    print(f"  {'primero':<32}{'después':<32}{'casos':>7}{'días':>6}")
    print("  " + "-" * 79)
    for s in seq[:12]:
        print(f"  {s['a'][:31]:<32}{s['b'][:31]:<32}{s['n']:>7}{s['retardo_mediano']:>6}")

    SALIDA.write_text(json.dumps({
        "fuente": "SENAPRED · Eventos de Emergencia 2015-2024 (50.457 registros)",
        "que_mide": {
            "FAS": ("acoplamiento: pares de tipos de evento que ocurren el mismo "
                    "día y en la misma comuna más de lo que el azar explicaría, "
                    "medido con lift = P(A y B) / (P(A) × P(B))"),
            "FPI": ("propagación: cuando ocurren en secuencia dentro de "
                    f"{VENTANA_DIAS} días, el retardo mediano en días"),
        },
        "advertencia": ("Acoplamiento NO es causalidad: que dos tipos co-ocurran "
                        "puede deberse a que uno provoca al otro o a que ambos "
                        "tienen una causa común. Lo que sí se puede afirmar es "
                        "que fallan juntos — que es lo que el ICSGS necesita."),
        "min_casos": MIN_CASOS,
        "pares": pares[:200],
        "secuencias": seq[:200],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    print(f"  pares con acoplamiento: {len(pares)} · secuencias: {len(seq)}")
    return 0


if __name__ == "__main__":
    print("=" * 82)
    print("ACOPLAMIENTO MEDIDO · qué falla junto con qué")
    print("=" * 82)
    sys.exit(main())
