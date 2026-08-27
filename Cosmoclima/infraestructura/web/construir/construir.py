"""
CONSTRUIR · genera todos los artefactos de la aplicación
=========================================================

Un comando que deja `publico/datos/` listo para que la aplicación arranque.

★ EL MANIFIESTO ES LA PIEZA IMPORTANTE
----------------------------------------
Cada corrida escribe `manifiesto.json` con los conteos de todo lo generado, y
**la aplicación se niega a arrancar si lo que carga no coincide**.

No es burocracia. Este proyecto lleva semanas encontrándose el mismo tipo de
error: nada revienta, todo devuelve un número plausible. Un `$batch` que perdió
208 filas devolvió HTTP 200. Una columna calculada que devolvía 1 en todas las
filas se aceptó sin protestar. Un CSV con 0,6667 donde iba 2/3 movió 237 filas de
banda. Ninguno gritó.

El manifiesto convierte «se perdieron filas» en un fallo ruidoso.

USO
---
    ../../.venv-esa/bin/python construir/construir.py
    ../../.venv-esa/bin/python construir/construir.py --sin-pronostico
    ../../.venv-esa/bin/python construir/construir.py --solo matriz,activos
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
DATOS = AQUI.parent / "publico" / "datos"

import activos, celdas, climatologia, comunas, matriz, pronostico, territorios, umbrales  # noqa: E402

# Orden obligado: `activos` necesita `territorios.json` ya escrito.
PASOS = [
    ("territorios", territorios.construir, "jerarquía comuna→provincia→región"),
    ("comunas", comunas.construir, "geometría simplificada"),
    ("matriz", matriz.construir, "los 846 ítems"),
    ("activos", activos.construir, "índice comuna × ítem"),
    ("climatologia", climatologia.construir, "referencia congelada 1990-2026"),
    ("pronostico", pronostico.construir, "★ los próximos 16 días"),
    # ★ `celdas` va después de `pronostico`: necesita saber qué celdas existen.
    ("celdas", celdas.construir, "★ puente comuna ↔ celda climática"),
    # ★ `umbrales` va al final porque lee los CSV que dejan los instrumentos de
    #   la raíz (umbral_por_tramo/elemento), no artefactos de los pasos previos.
    ("umbrales", umbrales.construir, "★ de milímetros a consecuencias"),
]

# Lo que el manifiesto EXIGE: un PISO. Si un conteo baja de aquí, algo se perdió;
# que suba es lo normal según se incorporan fuentes.
# ⚠️ «items» se deja en 846 A PROPÓSITO: son los del Word oficial. Los ítems
#    creados después (846-863) son añadidos de este proyecto y no deben subir el
#    piso, porque entonces el día que alguien los quite el portero no lo notaría.
ESPERADO = {
    "comunas": 345, "provincias": 56, "regiones": 16,
    "comunas_geometria": 345, "items": 846,
    "activos": 96423, "celdas_climatologia": 357,
}


def main():
    solo = None
    for a in sys.argv[1:]:
        if a.startswith("--solo"):
            solo = set((a.split("=", 1)[1] if "=" in a else
                        sys.argv[sys.argv.index(a) + 1]).split(","))
    sin_pron = "--sin-pronostico" in sys.argv

    print("=" * 74)
    print("CONSTRUIR · artefactos de la aplicación MICR")
    print("=" * 74)

    manifiesto, fallos = {}, []
    for nombre, fn, desc in PASOS:
        if solo and nombre not in solo:
            continue
        if nombre == "pronostico" and sin_pron:
            print(f"\n── {nombre} · omitido por --sin-pronostico")
            continue
        print(f"\n── {nombre} · {desc}")
        t0 = time.time()
        try:
            r = fn()
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
            fallos.append(nombre)
            continue
        if r is None:
            fallos.append(nombre)
            continue
        manifiesto.update(r)
        print(f"  ({time.time()-t0:.1f}s)")

    # ── el control ──────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("MANIFIESTO")
    print("=" * 74 + "\n")
    discrepan = []
    for k, v in sorted(manifiesto.items()):
        esp = ESPERADO.get(k)
        if esp is None:
            print(f"  {k:<24} {v:>10,}")
        elif v == esp:
            print(f"  {k:<24} {v:>10,}  ✓")
        else:
            print(f"  {k:<24} {v:>10,}  ✗ se esperaban {esp:,}")
            discrepan.append(k)

    pesos = {p.name: p.stat().st_size for p in sorted(DATOS.glob("*.json"))}
    print(f"\n  artefactos: {len(pesos)} · total {sum(pesos.values())/1e6:.2f} MB")
    for n, b in pesos.items():
        print(f"     {n:<28} {b/1e3:>8.0f} KB")

    (DATOS / "manifiesto.json").write_text(json.dumps({
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conteos": manifiesto,
        "esperado": ESPERADO,
        "discrepancias": discrepan,
        "artefactos": pesos,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    if fallos:
        print(f"\n  ✗ pasos que fallaron: {', '.join(fallos)}")
    if discrepan:
        print(f"  ✗ conteos que no cuadran: {', '.join(discrepan)}")
    if not fallos and not discrepan:
        print("\n  ✓ todo cuadra. La aplicación puede arrancar.")
    return 1 if (fallos or discrepan) else 0


if __name__ == "__main__":
    sys.exit(main())
