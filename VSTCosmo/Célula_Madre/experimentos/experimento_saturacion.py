#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENTO DE SATURACIÓN DE ESTÍMULOS — díada ANIMA A+B.

Mundo = secuencia_saturacion.wav (batería de audios cortos, fuente 'archivo', loopea).
Layout: A -> L=Mundo, R=voz del par · B espejado (L=voz del par, R=Mundo).

Cada PASADA (10 min):
  · Bloque 1 (0–5 min): la secuencia suena (con sus cortes de 3s entre temas). Ambas entradas abiertas.
  · Bloque 2 (5–10 min): ablaciones alternadas vía /mute, en ventanas de 60s:
      normal · CERRAR MUNDO · normal · CERRAR VOZ DEL PAR · normal
DOS pasadas idénticas (cada /start reinicia el cursor del audio → mismo orden). Memoria persiste entre pasadas.
Al terminar CADA pasada: descarga /csv y /sesion de A y B a ~/Downloads.

Uso (detached):  nohup venv/bin/python Célula_Madre/experimentos/experimento_saturacion.py >/dev/null 2>&1 &
"""
import json, os, time, urllib.request, datetime

A, B = ("A", 7788), ("B", 7799)
ORGS = [A, B]
SECUENCIA = "secuencia_saturacion.wav"
DOWNLOADS = os.path.expanduser("~/Downloads")
LOGDIR = os.path.expanduser("~/Library/Logs/vstcosmo-audioserver")
os.makedirs(LOGDIR, exist_ok=True)
LOG = os.path.join(LOGDIR, f"saturacion_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")

BLOQUE1 = int(os.environ.get("SAT_BLOQUE1", "300"))   # 5 min secuencia
VENTANA = int(os.environ.get("SAT_VENTANA", "60"))    # ventanas de ablación de 60s
PASADAS = int(os.environ.get("SAT_PASADAS", "2"))

ARCH = {"tipo": "archivo", "nombre": SECUENCIA}
PAR = {"tipo": "comunicacion", "modo": "R2D2", "nombre": "voz del par"}
# A: L=Mundo, R=par ; B: L=par, R=Mundo
CFG = {
    A: {"left_src": ARCH, "right_src": PAR},
    B: {"left_src": PAR, "right_src": ARCH},
}
# Oído de cada cosa por organismo → para las ablaciones por /mute
MUNDO_OIDO = {A: "left", B: "right"}
PAR_OIDO = {A: "right", B: "left"}


def log(msg):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=15) as r:
        return r.read()


def start_diada():
    for org, port in ORGS:
        cfg = dict(CFG[(org, port)])
        cfg.update(binaural=True, segundos=2, continuo=True, criterio_duracion="min")
        post(port, "/start", {"cfg": cfg, "modo_vida": "experimento"})
    log("díada arrancada · Mundo=secuencia · acople A↔B")


def set_estado(estado):
    """estado ∈ {'normal','cerrar_mundo','cerrar_par'} → /mute por organismo."""
    for org, port in ORGS:
        mute = {"left": False, "right": False}
        if estado == "cerrar_mundo":
            mute[MUNDO_OIDO[(org, port)]] = True
        elif estado == "cerrar_par":
            mute[PAR_OIDO[(org, port)]] = True
        post(port, "/mute", mute)
    log(f"  → estado de entradas: {estado.upper()}")


def salud(tag=""):
    for org, port in ORGS:
        try:
            est = json.loads(get(port, "/estado"))
            niv = json.loads(get(port, "/niveles"))
            m = niv.get("master", {})
            o = niv.get("organismo", {}).get("canales", {})
            vz = {k: round(v.get("rms", 0), 3) for k, v in o.items()}
            log(f"  {org}{tag}: t={est.get('t')}s vivo={est.get('vivo')} voz={est.get('voz_emitida')} "
                f"| oye(master) rms={m.get('rms')} pico={m.get('pico')} | voz_propia_LR={vz}")
        except Exception as e:
            log(f"  {org}{tag}: (sin lectura: {e})")


def espera_con_salud(segundos, cada=60):
    fin = time.time() + segundos
    while time.time() < fin:
        time.sleep(min(cada, max(1, fin - time.time())))
        if time.time() < fin - 1:
            salud()


def descargar(pasada):
    ts = f"{datetime.datetime.now():%Y%m%dT%H%M%S}"
    for org, port in ORGS:
        try:
            csv = get(port, "/csv")
            fcsv = os.path.join(DOWNLOADS, f"saturacion_p{pasada}_Organismo{org}_{ts}.csv")
            with open(fcsv, "wb") as f:
                f.write(csv)
            ses = get(port, "/sesion")
            fbit = os.path.join(DOWNLOADS, f"saturacion_p{pasada}_bitacora_Organismo{org}_{ts}.json")
            with open(fbit, "wb") as f:
                f.write(ses)
            log(f"  ⬇ {org} pasada {pasada}: CSV {len(csv)//1024}KB + bitácora {len(ses)//1024}KB → Descargas")
        except Exception as e:
            log(f"  ⚠ descarga {org} pasada {pasada} falló: {e}")


def pasada(n):
    log("=" * 60)
    log(f"PASADA {n}/{PASADAS} — inicio")
    start_diada()
    time.sleep(6)            # deja que nazcan y se acoplen
    salud(" (baseline)")
    # Bloque 1: secuencia, ambas entradas abiertas
    log(f"BLOQUE 1 (secuencia, {BLOQUE1//60} min) — entradas abiertas")
    set_estado("normal")
    espera_con_salud(BLOQUE1, cada=60)
    # Bloque 2: ablaciones alternadas
    log(f"BLOQUE 2 (ablaciones, ventanas de {VENTANA}s)")
    for estado in ["normal", "cerrar_mundo", "normal", "cerrar_par", "normal"]:
        set_estado(estado)
        espera_con_salud(VENTANA, cada=VENTANA)
        salud(f" [{estado}]")
    set_estado("normal")
    salud(" (fin pasada)")
    descargar(n)
    log(f"PASADA {n} — fin")


def main():
    log("#" * 60)
    log("EXPERIMENTO DE SATURACIÓN DE ESTÍMULOS · díada A+B")
    log(f"secuencia Mundo: {SECUENCIA} · {PASADAS} pasadas de {(BLOQUE1 + 5*VENTANA)//60} min")
    log(f"log: {LOG}")
    for n in range(1, PASADAS + 1):
        pasada(n)
    # al terminar todo, deja los organismos detenidos
    for org, port in ORGS:
        try:
            post(port, "/control", {"action": "stop"})
        except Exception:
            pass
    log("EXPERIMENTO COMPLETO · organismos detenidos · datos en ~/Downloads")
    with open(os.path.join(LOGDIR, "ultimo_saturacion.txt"), "w") as f:
        f.write(LOG)


if __name__ == "__main__":
    main()
