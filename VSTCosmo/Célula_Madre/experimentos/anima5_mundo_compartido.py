#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPERIMENTO ANIMA-5 — Emergencia del valor ecológico de la voz en un MUNDO COMPARTIDO.
Orquestador EXTERNO (no toca la arquitectura; configura vía POST /start qué oye cada organismo por fase).
Los órganos de la genealogía (Alteridad/ValorEcologicoVoz/Expectativa) NO se resetean entre /start → su
aprendizaje ACUMULA entre fases y ciclos. El cursor de wav (continuo) hace que el archivo streamee completo.

CICLO por estímulo (alterna 'primero'): Fase1 'primero' explora (estímulo+voz par) mientras 'segundo' oye
silencio+voz par → conversación → Fase2 a la inversa → conversación → Fase3 ambos el mismo mundo → conversación.
Alternancia: ciclo impar A→B→ambos ; par B→A→ambos.
ENV: GAP_S(25) MIN_CONV_S(35) MAX_MINUTES(60, presupuesto) CHUNK(5) FILES(lista coma; vacío=todos) START_AT(1).
"""
import os, sys, json, time, glob, urllib.request

A_URL = os.environ.get("ANIMA_A_URL", "http://localhost:7788")
B_URL = os.environ.get("ANIMA_B_URL", "http://localhost:7799")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/audio_binaural")
HIST = os.environ.get("VST_HISTORY_HOST", "/Volumes/LaCie/RMD/Cosmolab/VSTCosmo/Docker_Historia")
GAP_S = float(os.environ.get("GAP_S", "25"))
MIN_CONV_S = float(os.environ.get("MIN_CONV_S", "35"))
MAX_MIN = float(os.environ.get("MAX_MINUTES", "60"))     # presupuesto total: se detiene tras superar esto
CHUNK = float(os.environ.get("CHUNK", "5"))              # tamaño de bloque (continuo → el cursor avanza)
FILES = [s.strip() for s in os.environ.get("FILES", "").split(",") if s.strip()]
START_AT = int(os.environ.get("START_AT", "1"))
TIMELINE = os.path.join(HIST, "ANIMA5_timeline.csv")
SILENCIO = {"tipo": "demo", "spec": "demo:silencio"}
T0 = time.time()

def _http(url, obj=None, timeout=10):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def peer_src(url):
    f = _http(url + "/fuentes", timeout=6)
    com = (f.get("comunicacion") or []) if isinstance(f, dict) else []
    return com[0] if com else None

def dur_audio(path):
    import soundfile as sf
    info = sf.info(path); return info.frames / float(info.samplerate)

def listar():
    nombres = FILES if FILES else [f for f in sorted(os.listdir(AUDIO_DIR)) if f.lower().endswith(".wav")]
    out = []
    for f in nombres:
        try: out.append((f, dur_audio(os.path.join(AUDIO_DIR, f))))
        except Exception as e: log(f"  (omito {f}: {e})")
    return out

def cfg_org(org, mundo):
    return {"left_src": mundo, "right_src": PEER["A"]} if org == "A" else {"left_src": PEER["B"], "right_src": mundo}

def start(org, mundo):
    url = A_URL if org == "A" else B_URL
    cfg = cfg_org(org, mundo); cfg.update({"binaural": False, "segundos": CHUNK, "continuo": True, "criterio_duracion": "min"})
    try: _http(url + "/start", {"cfg": cfg, "sim_s": CHUNK, "modo_vida": "experimento"})
    except Exception as e: log(f"    ! /start {org}: {e}")

def stim(f): return {"tipo": "archivo", "nombre": f}
def log(m): print(time.strftime("[%H:%M:%S] ") + m, flush=True)

def tl(cycle, fase, archivo, dur, quien):
    nueva = not os.path.exists(TIMELINE)
    with open(TIMELINE, "a", encoding="utf-8") as fh:
        if nueva: fh.write("ts_real,ciclo,fase,archivo,dur_s,explorador\n")
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{cycle},{fase},{archivo},{dur:.1f},{quien}\n")

def fase_estimulo(explorador, oyente, archivo, D, cyc, et):
    log(f"  {et}: {explorador} explora '{archivo}' ({D:.0f}s) · {oyente} oye silencio+voz")
    tl(cyc, et, archivo, D, explorador)
    start(explorador, stim(archivo)); start(oyente, SILENCIO); time.sleep(D)

def fase_compartida(archivo, D, cyc):
    log(f"  FASE3 compartida: A y B oyen '{archivo}' ({D:.0f}s)")
    tl(cyc, "FASE3_compartida", archivo, D, "ambos")
    start("A", stim(archivo)); start("B", stim(archivo)); time.sleep(D)

def conversacion(cyc, archivo, seg):
    log(f"  conversación libre ({seg:.0f}s)")
    tl(cyc, "conversacion", archivo, seg, "ambos")
    start("A", SILENCIO); start("B", SILENCIO); time.sleep(seg)

def restante(): return MAX_MIN * 60 - (time.time() - T0)

def main():
    global PEER
    PEER = {"A": peer_src(A_URL), "B": peer_src(B_URL)}
    if not PEER["A"] or not PEER["B"]:
        log("ERROR: sin fuente 'comunicacion'. ¿Vivos A y B?"); sys.exit(1)
    audios = listar()
    log(f"ANIMA-5 · {len(audios)} estímulos · presupuesto {MAX_MIN:.0f} min · timeline → {TIMELINE}")
    for i, (archivo, D) in enumerate(audios, start=1):
        if i < START_AT: continue
        gap = max(GAP_S, MIN_CONV_S - D)
        ciclo_est = 3 * D + 3 * gap
        if restante() <= 0:
            log(f"⏱ presupuesto agotado ({MAX_MIN:.0f} min) — detengo antes del ciclo {i}."); break
        if ciclo_est > restante() and D > 60:        # un ciclo largo no cabe → sáltalo (deja preview corto)
            log(f"  (salto '{archivo}': el ciclo (~{ciclo_est/60:.0f} min) no cabe en lo que queda)"); continue
        primero, segundo = ("A", "B") if i % 2 else ("B", "A")
        log(f"═══ CICLO {i}/{len(audios)} · '{archivo}' ({D:.0f}s) · {primero}→{segundo}→ambos · quedan {restante()/60:.0f} min ═══")
        fase_estimulo(primero, segundo, archivo, D, i, "FASE1_individual"); conversacion(i, archivo, gap)
        fase_estimulo(segundo, primero, archivo, D, i, "FASE2_complementaria"); conversacion(i, archivo, gap)
        fase_compartida(archivo, D, i); conversacion(i, archivo, gap)
    log("ANIMA-5 (preview) terminado. Restaurando vida acoplada (silencio + voz del par)…")
    start("A", SILENCIO); start("B", SILENCIO)

if __name__ == "__main__":
    main()
