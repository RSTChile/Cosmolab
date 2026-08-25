"""
CLIMATOLOGÍA · la referencia contra la cual se mide un pronóstico
==================================================================

★ EL PROBLEMA QUE RESUELVE, Y POR QUÉ NO ES OBVIO
---------------------------------------------------
El pronóstico dice «45 mm el jueves». Por sí solo ese número no significa nada:
45 mm en Valdivia son un martes cualquiera y en el desierto de Atacama son un
aluvión. Para convertirlo en peligro hacen falta dos referencias, y las dos se
calculan aquí:

    magnitud nacional   ¿es mucha agua comparada con todo el país y toda la
                        historia?          → percentiles de TODOS los episodios
    excedencia local    ¿supera lo que ESTE lugar aguanta?
                        → la normal anual de la propia celda

`PelPre = √(magnitud × excedencia)` es la media geométrica de ambas: sólo da alto
si las dos lo son. Es la corrección que se hizo tras fallar el ancla de Copiapó,
donde medir sólo rareza saturaba la señal.

★ 4,78 MILLONES DE FILAS SE CONVIERTEN EN 357 NÚMEROS Y UNA CURVA
-------------------------------------------------------------------
`clima_diario_celdas.csv` pesa 124 MB. Pero la aplicación **no necesita la serie**:
necesita saber en qué percentil cae un valor nuevo. Eso son:

    · la normal anual de cada celda                    →  357 números
    · la distribución nacional de magnitudes           →  201 cortes (cada 0,5 %)
    · la distribución nacional de excedencias          →  201 cortes

Unos 120 KB en total. **Mil veces menos, y responde exactamente la misma
pregunta.**

★ Y SE CONGELA
----------------
Igual que los cortes de la Matriz. Estas distribuciones se calculan UNA VEZ sobre
1990-2026 y no se recalculan con cada refresco. Recalcularlas haría que un
pronóstico de hoy se comparara contra una referencia distinta a la de ayer, y
volvería la no estacionariedad por la puerta de atrás.

USO
---
    ../../.venv-esa/bin/python construir/climatologia.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
SERIE = RAIZ / "datos" / "clima_diario_celdas.csv"
DATOS = AQUI.parent / "publico" / "datos"
SALIDA = DATOS / "climatologia.json"

PISO_MM = 0.5          # bajo esto no se cuenta como episodio (igual que pelpre_diario)
CORTES = 201           # percentiles cada 0,5 %


def razon_contra_normal(evento_mm, normal_mm, piso_mm=5.0):
    """Cuánto excede el evento a lo que el lugar recibe en un año típico.

    Copiada de `normalizar.py` para que este archivo no dependa de la ruta del
    proyecto padre. Si allá cambia, aquí tiene que cambiar — y la prueba dorada
    del dominio lo detectaría.
    """
    return evento_mm / max(normal_mm, piso_mm)


def cuantiles(valores, n=CORTES):
    """`n` cortes equiespaciados de la distribución, para buscar percentiles
    por búsqueda binaria en el navegador en vez de cargar la muestra entera."""
    s = sorted(valores)
    if not s:
        return []
    return [round(s[min(len(s) - 1, int(round(i / (n - 1) * (len(s) - 1))))], 4)
            for i in range(n)]


def construir():
    if not SERIE.exists():
        print(f"  ✗ falta {SERIE.name}")
        return None
    print(f"  leyendo {SERIE.name} ({SERIE.stat().st_size/1e6:.0f} MB)…", flush=True)

    # ── una sola pasada: acumulado de 48 h y total por año, por celda ───────
    por_celda = defaultdict(dict)          # celda → {fecha: mm}
    n = 0
    with SERIE.open(encoding="utf-8") as fh:
        for x in csv.DictReader(fh):
            v = x["precip_mm"]
            if v in ("", "None"):
                continue
            por_celda[x["celda"]][x["fecha"]] = float(v)
            n += 1
            if n % 1_000_000 == 0:
                print(f"    {n:,} filas…", flush=True)
    print(f"  filas leídas : {n:,} · celdas: {len(por_celda)}")

    from datetime import date, timedelta
    normales, magnitudes, excedencias = {}, [], []
    for celda, serie in por_celda.items():
        # normal anual: sólo con los años que tienen al menos 330 días
        por_anio, dias = defaultdict(float), defaultdict(int)
        for f, mm in serie.items():
            por_anio[f[:4]] += mm
            dias[f[:4]] += 1
        completos = [v for a, v in por_anio.items() if dias[a] >= 330]
        if not completos:
            continue
        normal = sum(completos) / len(completos)
        normales[celda] = round(normal, 2)

        # acumulado de 48 h. ★ No se rellenan huecos con cero: un día sin dato
        # no es un día sin lluvia.
        for f, mm in serie.items():
            ayer = (date.fromisoformat(f) - timedelta(days=1)).isoformat()
            v = mm + serie[ayer] if ayer in serie else mm
            if v <= PISO_MM:
                continue
            magnitudes.append(v)
            excedencias.append(razon_contra_normal(v, normal))

    print(f"  celdas con normal anual : {len(normales)}")
    print(f"  episodios sobre el piso : {len(magnitudes):,}")

    q_mag = cuantiles(magnitudes)
    q_exc = cuantiles(excedencias)
    DATOS.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps({
        "congelada": "1990-01-01 a 2026-08-21",
        "piso_mm": PISO_MM,
        "normal_anual": normales,
        "cuantiles_magnitud": q_mag,
        "cuantiles_excedencia": q_exc,
        "n_episodios": len(magnitudes),
    }, ensure_ascii=False), encoding="utf-8")

    print(f"\n  distribución nacional de magnitud:")
    for p in (50, 75, 90, 99, 100):
        print(f"     P{p:<4}{q_mag[min(CORTES-1, int(p/100*(CORTES-1)))]:>9.1f} mm")
    print(f"\n  escrito: {SALIDA.name} ({SALIDA.stat().st_size/1e3:.0f} KB)")
    return {"celdas_climatologia": len(normales), "episodios": len(magnitudes)}


if __name__ == "__main__":
    print("=" * 70)
    print("CLIMATOLOGÍA · la referencia congelada")
    print("=" * 70)
    sys.exit(0 if construir() else 1)
