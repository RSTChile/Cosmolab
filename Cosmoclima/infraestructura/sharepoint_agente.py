"""
Agente RMD — SharePoint via Microsoft Graph (Ruta A: sesión delegada).
=====================================================================
Fuente: CONECTAR_OTRO_AGENTE_A_SHAREPOINT.md (23-ago-2026), implementado
verbatim para que DeepSeek Harness opere sobre rstchilecom.sharepoint.com/RMD
con permisos de lectura, escritura, creación de listas y borrado.

Ruta A (código de dispositivo):
  · Alexis pega un código en https://microsoft.com/devicelogin UNA vez.
  · El token de refresco dura ~90 días deslizantes; el agente se renueva solo.
  · El token vive FUERA del proyecto: ~/.cache/rmd_agente/token.json (0o600).

USO (tres modos):
  python3 sharepoint_agente.py login      # paso humano UNA vez (imprime código)
  python3 sharepoint_agente.py prueba     # sección 8 del doc: sitio/escritura/crear lista
  python3 sharepoint_agente.py operar     # (futuro) operaciones sobre el RMD
"""
import json
import os
import sys
import time
import requests
from pathlib import Path

SITIO = ("rstchilecom.sharepoint.com,"
         "d14670c1-197a-4976-ac78-92eb6d9ccff6,"
         "24a0c429-ccba-4544-bd83-e831a7a5a1a8")
GRAPH = "https://graph.microsoft.com/v1.0"
CLIENTE = "14d82eec-204b-4c2f-b7e8-296a70dab67e"   # Microsoft Graph Command Line Tools (público)
AUTORIDAD = "https://login.microsoftonline.com/organizations"
AMBITO = ("https://graph.microsoft.com/Sites.Manage.All "
          "https://graph.microsoft.com/Sites.ReadWrite.All offline_access")

# ★ El token va FUERA del proyecto y fuera de cualquier repositorio.
CACHE = Path.home() / ".cache" / "rmd_agente"
TOKEN = CACHE / "token.json"


def guardar(j):
    CACHE.mkdir(parents=True, exist_ok=True)
    j["expira_en"] = time.time() + j.get("expires_in", 3600) - 120   # margen
    TOKEN.write_text(json.dumps(j))
    os.chmod(TOKEN, 0o600)          # sólo el dueño puede leerlo


def login():
    """Se corre UNA vez. Imprime un código que Alexis pega en el navegador."""
    r = requests.post(f"{AUTORIDAD}/oauth2/v2.0/devicecode",
                      data={"client_id": CLIENTE, "scope": AMBITO},
                      timeout=30).json()
    print(r["message"], flush=True)          # ← el código, VISIBLE (flush obligatorio)
    print("CODIGO_PARA_NAVEGADOR=" + (r.get("user_code") or ""), flush=True)
    espera = r.get("interval", 5)
    for _ in range(180):
        time.sleep(espera)
        t = requests.post(f"{AUTORIDAD}/oauth2/v2.0/token",
                          data={"client_id": CLIENTE,
                                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                                "device_code": r["device_code"]}, timeout=30).json()
        if "access_token" in t:
            guardar(t)
            print("sesión iniciada · ámbitos:", t.get("scope"), flush=True)
            return
        if t.get("error") == "authorization_pending":
            continue
        if t.get("error") == "slow_down":
            espera += 5
            continue
        raise SystemExit(f"falló: {t.get('error_description', t)}")
    raise SystemExit("expiró el código")


def token():
    """Token válido. Se renueva solo con el token de refresco."""
    if not TOKEN.exists():
        raise SystemExit("No hay sesión. Corre login() primero.")
    j = json.loads(TOKEN.read_text())
    if time.time() < j.get("expira_en", 0):
        return j["access_token"]
    if "refresh_token" not in j:
        raise SystemExit("Sesión vencida sin refresco. Corre login() otra vez.")
    n = requests.post(f"{AUTORIDAD}/oauth2/v2.0/token",
                      data={"client_id": CLIENTE, "grant_type": "refresh_token",
                            "refresh_token": j["refresh_token"],
                            "scope": AMBITO}, timeout=30).json()
    if "access_token" not in n:
        raise SystemExit("No se pudo renovar. Corre login() otra vez.")
    guardar(n)
    return n["access_token"]


def llamar(metodo, url, cuerpo=None, reintentos=6):
    """UNA llamada a Graph. Si el servidor pide esperar, se espera lo que pide."""
    for intento in range(reintentos):
        r = requests.request(
            metodo, url if url.startswith("http") else GRAPH + url,
            headers={"Authorization": "Bearer " + token(),
                     "Content-Type": "application/json"},
            data=json.dumps(cuerpo) if cuerpo else None, timeout=120)
        if r.status_code in (429, 503, 504):
            time.sleep(int(r.headers.get("Retry-After", 2 ** intento * 5)))
            continue
        return r
    return r


# ── Sección 8 del doc · prueba de que quedó bien conectado ────────────────────

def prueba():
    # 1 · ¿hay sesión y ve el sitio?  (las dos cosas a la vez)
    r = llamar("GET", f"/sites/{SITIO}/lists?$select=id,displayName&$top=200")
    print("1 · listas del sitio:", r.status_code)
    if r.status_code != 200:
        print("   respuesta:", r.text[:300])
        return
    listas = r.json()["value"]
    print(f"   → {len(listas)} listas visibles")
    # 2 · ¿puede ESCRIBIR? (crea y borra un elemento de prueba)
    lid = next((x["id"] for x in listas if x["displayName"] == "Sub-Matrices"), None)
    if lid is None:
        print("2 · no encontré la lista 'Sub-Matrices'; uso la primera lista")
        lid = listas[0]["id"]
    p = llamar("POST", f"/sites/{SITIO}/lists/{lid}/items",
               {"fields": {"Title": "PRUEBA — borrar"}})
    print("2 · crear elemento:", p.status_code)
    if p.status_code == 201:
        print("   borrar elemento:", llamar(
            "DELETE", f"/sites/{SITIO}/lists/{lid}/items/{p.json()['id']}").status_code)
    else:
        print("   respuesta:", p.text[:300])
    # 3 · ¿puede CREAR LISTAS? (la que exige Sites.Manage.All)
    q = llamar("POST", f"/sites/{SITIO}/lists",
               {"displayName": "PRUEBA AGENTE — borrar",
                "list": {"template": "genericList"}})
    print("3 · crear lista:", q.status_code)
    if q.status_code == 201:
        print("   borrar lista:", llamar("DELETE",
              f"/sites/{SITIO}/lists/{q.json()['id']}").status_code)
    else:
        print("   respuesta:", q.text[:300])
        print("   (403 = falta Sites.Manage.All; volver a login con la cadena completa)")
    print("PRUEBA_COMPLETA")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "prueba"
    if modo == "login":
        login()
    elif modo == "prueba":
        prueba()
    else:
        print("modo desconocido:", modo)
        sys.exit(1)
