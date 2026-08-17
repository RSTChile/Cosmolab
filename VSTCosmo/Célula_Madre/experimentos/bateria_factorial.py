#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BATERÍA FACTORIAL NOCTURNA — díada A↔B con AUDIOS GUARDADOS (no música en vivo).
Pregunta central: ¿qué configuración dispara la IMITACIÓN (oao_imitacion_mag)?
Ejes: tipo de audio (voz/música/ruido/silencio) × relación A-B (mismo/diferente/swap)
      × cortes de entrada (ablaciones). Cada condición = 10 min. Captura por sondeo en vivo.
Tras cada condición: resumen ejecutivo (al INFORME .md). El CICLO se repite cada ~1h
(test de aprendizaje) hasta la mañana. Datos completos quedan en Docker_Historia.
"""
import json, os, time, urllib.request, datetime

A = ("A", 7788); B = ("B", 7799)
DL = os.path.expanduser("~/Downloads")
LOGDIR = os.path.expanduser("~/Library/Logs/vstcosmo-audioserver"); os.makedirs(LOGDIR, exist_ok=True)
TS0 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
REPORTE = os.path.join(DL, f"INFORME_BATERIA_FACTORIAL_{TS0}.md")
LOG = os.path.join(LOGDIR, f"bateria_factorial_{TS0}.log")
DUR_COND = int(os.environ.get("DUR_COND", "600"))     # 10 min por condición
GAP_CICLO = int(os.environ.get("GAP_CICLO", "3600"))  # 1 h entre ciclos
HORAS_TOTAL = float(os.environ.get("HORAS_TOTAL", "8.5"))  # duración total de la batería (horas) — robusto a la hora del día
SAMPLE = 4                                            # sondeo cada 4 s

# --- fuentes ---
ARCH = lambda f: {"tipo": "archivo", "nombre": f}
PAR  = {"tipo": "comunicacion", "modo": "R2D2", "nombre": "voz del par"}
SIL  = {"tipo": "demo", "spec": "demo:silencio"}
VOZ = ARCH("Voz_Estudio.wav"); MUS = ARCH("Brandemburgo_pos60deg.wav"); RUI = ARCH("Ruido blanco.wav")

def cfg_org(org, mundo):
    # A: par por oído R -> L=mundo, R=par.  B: par por oído L -> L=par, R=mundo. (dicótico real)
    base = {"binaural": False, "segundos": 2, "continuo": True, "criterio_duracion": "min"}
    if org == "A": base.update(left_src=mundo, right_src=PAR)
    else:          base.update(left_src=PAR, right_src=mundo)
    return base

# (nombre, mundo_A, mundo_B, ablaciones)
CONDICIONES = [
    ("VOZ_ambos_igual",          VOZ, VOZ, False),
    ("MUSICA_ambos_igual",       MUS, MUS, False),
    ("DIFERENTE_Avoz_Bmusica",   VOZ, MUS, False),
    ("SWAP_Amusica_Bvoz",        MUS, VOZ, False),
    ("RUIDO_ambos_igual",        RUI, RUI, False),
    ("SILENCIO_control",         SIL, SIL, False),
    ("VOZ_con_ablaciones",       VOZ, VOZ, True),
]
# oído de cada cosa (para ablaciones): A mundo=L par=R ; B mundo=R par=L
MUNDO_OIDO = {"A": "left", "B": "right"}; PAR_OIDO = {"A": "right", "B": "left"}

def log(m):
    line = f"[{datetime.datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def rep(m):  # al informe markdown
    with open(REPORTE, "a") as f: f.write(m + "\n")

def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r: return r.read()
    except Exception as e: log(f"  post {path}:{port} err {e}"); return None

def get(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=8) as r: return json.loads(r.read())
    except Exception: return {}

def start_diada(mundoA, mundoB):
    post(A[1], "/start", {"cfg": cfg_org("A", mundoA), "modo_vida": "experimento"})
    post(B[1], "/start", {"cfg": cfg_org("B", mundoB), "modo_vida": "experimento"})

def set_estado(estado):  # ablaciones
    for org, port in (A, B):
        mute = {"left": False, "right": False}
        if estado == "cerrar_mundo": mute[MUNDO_OIDO[org]] = True
        elif estado == "cerrar_par": mute[PAR_OIDO[org]] = True
        post(port, "/mute", mute)

def correr_condicion(cic, idx, cond):
    nombre, mA, mB, abl = cond
    log(f"  ── COND {idx+1}/{len(CONDICIONES)}: {nombre} (ablaciones={abl}) ──")
    start_diada(mA, mB); time.sleep(6)
    # acumuladores por organismo
    acc = {o[0]: {"imit_max": 0.0, "imit_sum": 0.0, "exp0": None, "exp1": 0, "cara_sum": 0.0,
                  "echo_max": 0.0, "alt_sum": 0.0, "OI_sum": 0.0, "LF_sum": 0.0, "n": 0,
                  "voz": {}, "voz_propias": 0, "vivo": False} for o in (A, B)}
    # cronograma de ablaciones (si aplica): 5 ventanas
    fin = time.time() + DUR_COND
    ventanas = ["normal", "cerrar_mundo", "normal", "cerrar_par", "normal"] if abl else ["normal"]
    win_dur = DUR_COND / len(ventanas); prox_win = time.time(); wi = -1
    while time.time() < fin:
        if abl and time.time() >= prox_win:
            wi += 1
            if wi < len(ventanas): set_estado(ventanas[wi]); log(f"     estado entradas: {ventanas[wi]}"); prox_win += win_dur
        for org, port in (A, B):
            d = get(port, "/ultima_fila"); f = d.get("fila") or {}
            if not f: continue
            a = acc[org]; a["n"] += 1; a["vivo"] = bool(d.get("corriendo"))
            im = float(f.get("oao_imitacion_mag") or 0); a["imit_max"] = max(a["imit_max"], im); a["imit_sum"] += im
            e = float(f.get("ove_experiencias") or 0)
            if a["exp0"] is None: a["exp0"] = e
            a["exp1"] = e
            a["cara_sum"] += float(f.get("cara_valoracion") or 0)
            a["echo_max"] = max(a["echo_max"], float(f.get("oao_echoica_n") or 0))
            a["alt_sum"] += float(f.get("alt_intencion_comunicativa") or 0)
            a["OI_sum"] += float(f.get("OI") or 0); a["LF_sum"] += float(f.get("LF_op") or 0)
            a["voz_propias"] = int(f.get("voz_propias") or 0)
            vz = f.get("voz_emitida", "-")
            if vz and vz != "-": a["voz"][vz] = a["voz"].get(vz, 0) + 1
        time.sleep(SAMPLE)
    if abl: set_estado("normal")
    # resumen ejecutivo
    def met(o):
        a = acc[o]; n = max(a["n"], 1)
        top = sorted(a["voz"].items(), key=lambda x: -x[1])[:3]
        return {"imit_max": round(a["imit_max"], 4), "imit_med": round(a["imit_sum"]/n, 4),
                "exp_delta": int((a["exp1"] or 0) - (a["exp0"] or 0)), "cara_med": round(a["cara_sum"]/n, 3),
                "echo_max": a["echo_max"], "alt_med": round(a["alt_sum"]/n, 4),
                "OI_med": round(a["OI_sum"]/n, 3), "LF_med": round(a["LF_sum"]/n, 3),
                "voz_propias": a["voz_propias"], "voz_top": ", ".join(k for k, _ in top), "vivo": a["vivo"]}
    mA, mB = met("A"), met("B")
    disparo = "🔥 SÍ" if (mA["imit_max"] > 0.01 or mB["imit_max"] > 0.01) else "—"
    log(f"     resumen: imit_max A={mA['imit_max']} B={mB['imit_max']}  → IMITACIÓN {disparo}")
    rep(f"| C{cic} | {nombre} | **{mA['imit_max']}** / **{mB['imit_max']}** | {disparo} | "
        f"{mA['exp_delta']}/{mB['exp_delta']} | {mA['voz_propias']}/{mB['voz_propias']} | "
        f"{mA['echo_max']:.0f}/{mB['echo_max']:.0f} | {mA['alt_med']}/{mB['alt_med']} | "
        f"{mA['voz_top'][:22]} ‖ {mB['voz_top'][:22]} |")
    return nombre, mA["imit_max"], mB["imit_max"], disparo

def main():
    rep(f"# INFORME BATERÍA FACTORIAL NOCTURNA — díada A↔B (audios guardados)\n")
    rep(f"inicio: {datetime.datetime.now():%Y-%m-%d %H:%M}  ·  cada condición 10 min  ·  ciclo repetido cada ~1h (test de aprendizaje)\n")
    rep("**Pregunta:** ¿qué configuración dispara la imitación (oao_imitacion_mag) con audios guardados (no Rode vivo)?")
    rep("Audios: Voz_Estudio.wav · Brandemburgo.wav (música) · Ruido blanco.wav · silencio. Dicótico (mundo+par por oído).")
    rep("Columnas: imit_max A/B · ¿disparó? · ΔOVE_exp A/B · voz_propias A/B · echoica_max A/B · alt_intención A/B · voz top.\n")
    matriz = {}  # nombre -> [imit por ciclo]
    cic = 0
    FIN_DT = datetime.datetime.now() + datetime.timedelta(hours=HORAS_TOTAL)
    while datetime.datetime.now() < FIN_DT or cic == 0:
        cic += 1
        log(f"════════ CICLO {cic} ════════")
        rep(f"\n## CICLO {cic} — {datetime.datetime.now():%H:%M}\n")
        rep("| Ciclo | Condición | imit_max A/B | ¿disparó? | ΔOVE A/B | voz_propias A/B | echoica A/B | alt_int A/B | voz top A ‖ B |")
        rep("|---|---|---|---|---|---|---|---|---|")
        for idx, cond in enumerate(CONDICIONES):
            try:
                nombre, ia, ib, disp = correr_condicion(cic, idx, cond)
                matriz.setdefault(nombre, []).append(max(ia, ib))
            except Exception as e:
                log(f"  ERROR en condición {cond[0]}: {e}")
        # comparación cruzada acumulada (aprendizaje)
        rep(f"\n### Comparación cruzada — imit_max por condición × ciclo (¿aprenden con la repetición?)")
        rep("| Condición | " + " | ".join(f"C{i+1}" for i in range(cic)) + " |")
        rep("|---|" + "|".join(["---"]*cic) + "|")
        for nombre, _, _, _ in [(c[0],)*4 for c in CONDICIONES]:
            vals = matriz.get(nombre, [])
            rep(f"| {nombre} | " + " | ".join(f"{v}" for v in vals) + " |")
        log(f"CICLO {cic} completo. Esperando {GAP_CICLO//60} min antes del próximo (test de aprendizaje)…")
        if datetime.datetime.now() >= FIN_DT:
            break
        time.sleep(GAP_CICLO)
    # cierre
    for _, port in (A, B): post(port, "/control", {"action": "stop"})
    rep(f"\n---\n**BATERÍA COMPLETA** · {cic} ciclos · {datetime.datetime.now():%Y-%m-%d %H:%M} · organismos detenidos · datos completos en Docker_Historia")
    log(f"BATERÍA COMPLETA: {cic} ciclos. Informe: {REPORTE}")
    with open(os.path.join(LOGDIR, "ultimo_bateria.txt"), "w") as f: f.write(REPORTE + "\n" + LOG)

if __name__ == "__main__":
    main()
