#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publicar_cosmogenesis.py — sube el informe de Cosmogénesis a cosmosemiotica.cl
y lo verifica contra lo servido (13-ago-2026).

Reusa el mecanismo ya probado de publicar_cosmoclima.py (autenticación por
llavero de macOS, subida por la API de cPanel, verificación por SHA-256 contra
lo que devuelve el servidor). La clave NUNCA aparece acá ni en ningún archivo.

ALCANCE ESTRICTO: sube la página y sus cuatro ilustraciones. No toca
experimentos.html ni ninguna otra página existente, y no borra nada.
Consecuencia declarada: hasta que alguien enlace esta página desde
experimentos.html, queda accesible sólo por su URL directa.
"""
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/Web/prueba_de_concepto")
import publicar_cosmoclima as base   # Panel, ResolverFijo, clave_desde_llavero

import requests

BASE = Path(__file__).resolve().parent
RAIZ = "/home/geografiasagrada/cosmosemiotica.cl"
SITIO = "https://cosmosemiotica.cl"

ARCHIVOS = [(BASE / "informe-cosmogenesis.html", RAIZ, "informe-cosmogenesis.html")]
for p in sorted((BASE / "imagenes/cosmogenesis").glob("*.jpg")):
    ARCHIVOS.append((p, f"{RAIZ}/imagenes/cosmogenesis", p.name))


def main():
    faltan = [str(p) for p, _, _ in ARCHIVOS if not p.exists()]
    if faltan:
        sys.exit("No existen localmente:\n  " + "\n  ".join(faltan))

    total = sum(p.stat().st_size for p, _, _ in ARCHIVOS)
    print(f"A subir: {len(ARCHIVOS)} archivos · {total/1024/1024:.2f} MB")
    for p, d, n in ARCHIVOS:
        print(f"  {p.stat().st_size/1024:7.0f} KB  {d.replace(RAIZ,'')}/{n}")
    print("\nNo se toca ninguna otra página del sitio.\n")

    panel = base.Panel(os.environ.get("CPANEL_USER", "geografiasagrada"),
                       base.clave_desde_llavero())
    panel.crear_dir(f"{RAIZ}/imagenes", "cosmogenesis")

    print("Subiendo…")
    for ruta, directorio, nombre in ARCHIVOS:
        panel.subir(ruta, nombre, directorio)
        print(f"  · {nombre}")

    print("\nVerificando contra lo SERVIDO (descarga y compara SHA-256)…")
    ses = requests.Session()
    ses.mount("https://", base.ResolverFijo())
    import warnings
    warnings.filterwarnings("ignore")
    fallos = []
    for ruta, directorio, nombre in ARCHIVOS:
        rel = directorio.replace(RAIZ, "").strip("/")
        url = f"{SITIO}/{rel}/{nombre}" if rel else f"{SITIO}/{nombre}"
        try:
            r = ses.get(url, timeout=600)
            if r.status_code != 200:
                fallos.append(f"{nombre}: HTTP {r.status_code}")
            elif hashlib.sha256(r.content).hexdigest() != hashlib.sha256(ruta.read_bytes()).hexdigest():
                fallos.append(f"{nombre}: la huella del servidor NO coincide")
            else:
                print(f"  OK {nombre}")
        except Exception as ex:
            fallos.append(f"{nombre}: {ex}")

    if fallos:
        print(f"\nFALLA — {len(fallos)} archivo(s) no verifican:")
        for f in fallos:
            print("   -", f)
        sys.exit(1)

    print(f"\nOK: los {len(ARCHIVOS)} archivos coinciden byte a byte con lo servido.")
    print(f"\n  {SITIO}/informe-cosmogenesis.html")


if __name__ == "__main__":
    main()
