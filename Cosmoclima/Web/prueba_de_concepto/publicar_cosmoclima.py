#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publicar_cosmoclima.py — sube SOLO el experimento Cosmoclima a cosmosemiotica.cl
y verifica contra lo servido (12-ago-2026).

ALCANCE ESTRICTO, a pedido de Alexis: "No puedes tocar otras páginas del sitio
por ahora. Sólo subir nuestro experimento y verificar que todo funciona."
  · Sube únicamente los archivos listados en ARCHIVOS (abajo).
  · No toca index.html, experimentos.html ni ninguna otra página existente.
  · No borra nada en el servidor.

REGLAS APLICADAS (de \\192.168.86.205\\ANIMA-Red\\Instruccion_Actualizacion_Versiones):
  1. Verificar CONTRA LO SERVIDO: tras subir cada archivo se vuelve a descargar
     por HTTPS y se compara el SHA-256. Si uno solo no coincide, se ABORTA.
  2. Nada de parches sobre el HTML en vivo: se suben archivos completos.
  3. Credenciales nunca en el repositorio ni en el código.

LA CLAVE NO PASA POR EL CHAT NI SE IMPRIME. Se lee del llavero de macOS:

    security add-generic-password -a geografiasagrada -s cosmosemiotica-cpanel -w

(ese comando pide la clave por teclado y la guarda cifrada; se corre UNA vez).
Alternativa: exportar CPANEL_PASS en el entorno antes de ejecutar este script.
"""

import hashlib
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Falta 'requests'.  pip3 install requests")

PANEL = "https://cpanel.geografiasagrada.cl"
SITIO = "https://cosmosemiotica.cl"
IP_SITIO = "162.249.169.18"
RAIZ = "/home/geografiasagrada/cosmosemiotica.cl"
BASE = Path(__file__).resolve().parent

# (ruta local, directorio remoto, nombre en el servidor)
ARCHIVOS = [
    (BASE / "informe-cosmoclima.html", RAIZ, "informe-cosmoclima.html"),
    (BASE / "sim-cosmoclima.html",     RAIZ, "sim-cosmoclima.html"),
]
# Las imágenes se agregan por patrón, PERO con --solo-html se omiten: ya están
# publicadas y no cambiaron desde entonces, así que volver a subir ~16 MB por
# la API de cPanel es tiempo perdido y una ventana de fallo sin motivo. La
# verificación posterior sí se hace igual sobre lo que se suba.
SOLO_HTML = "--solo-html" in sys.argv
if not SOLO_HTML:
    for p in sorted((BASE / "imagenes/web").glob("*.png")):
        ARCHIVOS.append((p, f"{RAIZ}/imagenes/web", p.name))
    for p in sorted((BASE / "imagenes/web/mini").glob("*.png")):
        ARCHIVOS.append((p, f"{RAIZ}/imagenes/web/mini", p.name))
    for p in sorted((BASE / "imagenes/web/campo").glob("*.jpg")):
        ARCHIVOS.append((p, f"{RAIZ}/imagenes/web/campo", p.name))
    ARCHIVOS.append((BASE / "imagenes/Gyriosomus kingi.png", f"{RAIZ}/imagenes", "Gyriosomus kingi.png"))
    ARCHIVOS.append((BASE / "imagenes/web/desierto-florido.jpg", f"{RAIZ}/imagenes/web", "desierto-florido.jpg"))

VERDE, ROJO, AMAR, FIN = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
OK, MAL, AVISO = f"{VERDE}OK{FIN}", f"{ROJO}FALLA{FIN}", f"{AMAR}aviso{FIN}"


def abortar(msg):
    sys.exit(f"\n{MAL} {msg}\n(no se tocó nada más en el servidor)")


def sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def clave_desde_llavero() -> str:
    """Lee la clave sin imprimirla nunca."""
    v = os.environ.get("CPANEL_PASS")
    if v:
        return v
    try:
        return subprocess.run(
            ["security", "find-generic-password", "-s", "cosmosemiotica-cpanel", "-w"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        abortar("no encontré la clave. Guardala una vez con:\n"
                "  security add-generic-password -a geografiasagrada "
                "-s cosmosemiotica-cpanel -w\n"
                "(te la pide por teclado; queda cifrada en el llavero)")


class ResolverFijo(requests.adapters.HTTPAdapter):
    """Fuerza la IP real del sitio: el DNS local puede estar cacheado."""
    def send(self, request, **kw):
        p = urllib.parse.urlparse(request.url)
        if p.hostname == "cosmosemiotica.cl":
            nueva = p._replace(netloc=IP_SITIO + (f":{p.port}" if p.port else ""))
            request.url = urllib.parse.urlunparse(nueva)
            request.headers["Host"] = "cosmosemiotica.cl"
            kw["verify"] = False
        return super().send(request, **kw)


class Panel:
    def __init__(self, usuario, clave):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "cosmoclima-publish/1.0"})
        r = self.s.post(f"{PANEL}/login/?login_only=1",
                        data={"user": usuario, "pass": clave}, timeout=60)
        try:
            d = r.json()
        except Exception:
            abortar(f"el panel no devolvió JSON al autenticar (HTTP {r.status_code})")
        if d.get("status") != 1:
            abortar(f"login rechazado: {d.get('message')} (el usuario va SIN '.cl')")
        self._cookie(r)
        self.base = f"{PANEL}{d['security_token']}"
        print(f"{OK} autenticado en cPanel")

    def _cookie(self, resp):
        try:
            heads = list(resp.raw.headers.get_all("Set-Cookie") or [])
        except Exception:
            sc = resp.headers.get("Set-Cookie")
            heads = [sc] if sc else []
        for h in heads:
            m = re.search(r"(?:^|,\s*)cpsession=([^;]+)", h)
            if m:
                self.s.cookies.set("cpsession", m.group(1),
                                   domain="cpanel.geografiasagrada.cl", path="/")
                return
        if not self.s.cookies.get("cpsession"):
            abortar("login OK pero no llegó la cookie de sesión")

    def crear_dir(self, padre, nombre):
        self.s.post(f"{self.base}/execute/Fileman/mkdir",
                    data={"path": padre, "name": nombre}, timeout=60)

    def subir(self, ruta, nombre, directorio):
        with open(ruta, "rb") as f:
            r = self.s.post(f"{self.base}/execute/Fileman/upload_files",
                            data={"dir": directorio, "overwrite": "1"},
                            files={"file-1": (nombre, f, "application/octet-stream")},
                            timeout=1800)
        try:
            d = r.json()
        except Exception:
            abortar(f"respuesta no-JSON subiendo {nombre} (HTTP {r.status_code})")
        if d.get("status") != 1 or d["data"]["failed"]:
            abortar(f"falló la subida de {nombre}: {d.get('errors')}")


def url_publica(directorio, nombre):
    rel = directorio.replace(RAIZ, "").strip("/")
    return f"{SITIO}/{rel}/{nombre}" if rel else f"{SITIO}/{nombre}"


def main():
    faltan = [str(p) for p, _, _ in ARCHIVOS if not p.exists()]
    if faltan:
        abortar("no existen localmente:\n  " + "\n  ".join(faltan))

    total = sum(p.stat().st_size for p, _, _ in ARCHIVOS)
    print(f"\nA subir: {len(ARCHIVOS)} archivos · {total/1024/1024:.1f} MB")
    print("Solo el experimento Cosmoclima. No se toca ninguna otra página.\n")

    panel = Panel(os.environ.get("CPANEL_USER", "geografiasagrada"), clave_desde_llavero())

    for sub in ("imagenes", "imagenes/web", "imagenes/web/mini", "imagenes/web/campo"):
        partes = sub.rsplit("/", 1)
        padre = f"{RAIZ}/{partes[0]}" if len(partes) == 2 else RAIZ
        panel.crear_dir(padre, partes[-1])

    print("Subiendo…")
    for ruta, directorio, nombre in ARCHIVOS:
        panel.subir(ruta, nombre, directorio)
        print(f"  · {nombre}")

    print("\nVerificando contra lo SERVIDO (descarga y compara SHA-256)…")
    ses = requests.Session()
    ses.mount("https://", ResolverFijo())
    import warnings
    warnings.filterwarnings("ignore")
    fallos = []
    for ruta, directorio, nombre in ARCHIVOS:
        url = url_publica(directorio, nombre)
        try:
            r = ses.get(url, timeout=600)
            if r.status_code != 200:
                fallos.append(f"{nombre}: HTTP {r.status_code}")
                continue
            if sha(r.content) != sha(ruta.read_bytes()):
                fallos.append(f"{nombre}: la huella del servidor NO coincide")
            else:
                print(f"  {OK} {nombre}")
        except Exception as ex:
            fallos.append(f"{nombre}: {ex}")

    if fallos:
        print(f"\n{MAL} {len(fallos)} archivo(s) no verifican:")
        for f in fallos:
            print("   -", f)
        sys.exit(1)

    print(f"\n{OK} los {len(ARCHIVOS)} archivos coinciden byte a byte con lo servido.")
    print(f"\n  {SITIO}/informe-cosmoclima.html")
    print(f"  {SITIO}/sim-cosmoclima.html")


if __name__ == "__main__":
    main()
