#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probar_clave_cpanel.py — dice si la clave guardada en el llavero SIRVE.

Por qué existe: el 13-ago-2026 se dio por buena la clave del llavero porque
`security find-generic-password` la devolvía sin error. Pero eso sólo prueba
que hay ALGO guardado y que se puede leer — no que sea la clave correcta. La
subida falló recién al final, después de armar todo. Este script cierra ese
hueco: intenta el login de verdad y no hace nada más.

NO sube archivos. NO toca el sitio. NO imprime la clave.
"""
import os
import sys

sys.path.insert(0, "/Users/alexis/Desktop/RMD/Cosmolab/Cosmoclima/Web/prueba_de_concepto")
import requests

PANEL = "https://cpanel.geografiasagrada.cl"


def clave():
    v = os.environ.get("CPANEL_PASS")
    if v:
        return v, "variable de entorno CPANEL_PASS"
    import subprocess
    try:
        r = subprocess.run(["security", "find-generic-password",
                            "-s", "cosmosemiotica-cpanel", "-w"],
                           capture_output=True, text=True, check=True)
        return r.stdout.strip(), "llavero de macOS"
    except Exception:
        return None, None


def main():
    c, origen = clave()
    if not c:
        sys.exit("No hay clave guardada. Guardala con:\n"
                 "  security add-generic-password -U -a geografiasagrada "
                 "-s cosmosemiotica-cpanel -w")
    print(f"Clave leída de: {origen}  (largo {len(c)} caracteres, no se muestra)")
    try:
        r = requests.post(f"{PANEL}/login/?login_only=1",
                          data={"user": os.environ.get("CPANEL_USER", "geografiasagrada"),
                                "pass": c}, timeout=60)
        d = r.json()
    except Exception as ex:
        sys.exit(f"No se pudo hablar con el panel: {ex}")

    if d.get("status") == 1:
        print("\n  ✓ LA CLAVE SIRVE — el panel aceptó el login.")
        print("    Ya se puede publicar.")
    else:
        print(f"\n  ✗ RECHAZADA: {d.get('message')}")
        print("    Actualizala con:")
        print("      security add-generic-password -U -a geografiasagrada "
              "-s cosmosemiotica-cpanel -w")
        sys.exit(1)


if __name__ == "__main__":
    main()
