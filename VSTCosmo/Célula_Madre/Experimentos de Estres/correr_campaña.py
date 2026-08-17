#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correr_campaña.py — Orquestador AUTÓNOMO del test de estrés del sistema completo ANIMA.
Corre desatendido hasta END (11:00). Por cada bloque: etiqueta el CSV (/exp_tag), conduce el
tráfico digital si aplica, registra en BITÁCORA (quién habla a quién, cada cambio de entrada),
y mide el acople del oído digital (real vs NULL/SHUFFLED). Tolera fallos: cada bloque en try/except.
"""
import urllib.request, json, time, random, math, os, sys

CARP = os.path.dirname(os.path.abspath(__file__))
A_WEB="http://localhost:7788"; A_BR="http://192.168.86.250:8772"
E_WEB="http://192.168.86.33:7788"
WEB={"A":A_WEB,"B":"http://localhost:7799","C":"http://localhost:7810","D":"http://localhost:7820","E":E_WEB}
CICLO="ESTRES_2026-07-05"
END = time.mktime(time.strptime("2026-07-05 11:00", "%Y-%m-%d %H:%M"))
DUR = 300          # 5 min por bloque
BASAL = 45         # descanso entre bloques
SEND_DT = 2.0      # cadencia de envío digital

def stamp(): return time.strftime("%H:%M:%S")
# nombres FIJOS: si el orquestador se relanza, ACUMULA en los mismos archivos (no fragmenta)
BITA = os.path.join(CARP, "bitacora_campaña_2026-07-05.md")
SNAP = os.path.join(CARP, "snapshots_campaña_2026-07-05.csv")

def bit(msg):
    line = "%s  %s" % (stamp(), msg)
    with open(BITA, "a") as f: f.write(line+"\n")
    print(line, flush=True)

def GET(u, t=5):
    with urllib.request.urlopen(u, timeout=t) as r: return json.loads(r.read().decode())
def fila(web):
    try:
        j=GET(web+"/ultima_fila"); f=j.get("fila")
        return f if isinstance(f,dict) else dict(zip(j.get("cols",[]), f or []))
    except Exception: return {}
def POST(url, data, js=True, t=6):
    try:
        d = json.dumps(data).encode() if js else str(data).encode("ascii","ignore")
        h = {"Content-Type":"application/json"} if js else {}
        urllib.request.urlopen(urllib.request.Request(url, data=d, headers=h, method="POST"), timeout=t).read()
        return True
    except Exception: return False
def a_tx(txt): return POST(A_BR+"/tx", txt, js=False)      # A → E por el puente
def e_tx(txt): return POST(E_WEB+"/nrf/tx", {"text":txt})  # E → A por el ATmega
def tag(orgs, topo, control):
    for o in orgs:
        POST(WEB[o]+"/exp_tag", {"exp_ciclo":CICLO,"exp_topologia":topo,"exp_control":control,
                                 "exp_mundo_audio":"rode","exp_fuente_relacion":"digital+audio"})

def token(f, who):
    voc = 1 if float(f.get("expr_vocalizando") or 0)>0.5 else 0
    vid = str(f.get("voz_id") or "0")[:2]
    ar = int(max(0,min(1,float(f.get("voz_arousal") or 0)))*99)
    va = int((max(-1,min(1,float(f.get("mem_valencia_estado") or 0)))+1)/2*99)
    t = "%s%dw%sa%02dv%02d" % (who,voc,vid,ar,va)
    return t.replace(" ","_")

def shuffle_tok(t):
    body=list(t[1:]); random.shuffle(body); return t[0]+"".join(body)

def corr(x,y):
    n=min(len(x),len(y))
    if n<3: return 0.0
    x,y=x[:n],y[:n]; mx=sum(x)/n; my=sum(y)/n
    num=sum((a-mx)*(b-my) for a,b in zip(x,y))
    dx=math.sqrt(sum((a-mx)**2 for a in x)); dy=math.sqrt(sum((b-my)**2 for b in y))
    return num/(dx*dy) if dx>0 and dy>0 else 0.0

def snap_row(bloque, control, cyc, met):
    new = not os.path.exists(SNAP)
    with open(SNAP,"a") as f:
        if new: f.write("ts,ciclo,bloque,control,cyc,"+",".join(met.keys())+"\n")
        f.write("%s,%s,%s,%s,%d,"%(stamp(),CICLO,bloque,control,cyc)+",".join("%.4f"%v for v in met.values())+"\n")

def cuidar_sdr_e():
    """Si el RSP1 de E se colgó (espectro plano), dispara la recuperación."""
    try:
        f=fila(E_WEB); esp=f.get("sdr_espectro")
        if isinstance(esp,list) and esp and max(esp)<=0.001:
            POST(E_WEB+"/radio/reactivar", {}); bit("  [cuidado] RSP1 de E plano → /radio/reactivar")
    except Exception: pass

def correr_bloque(bloque, dur, driver, control, orgs, cyc):
    tag(orgs, bloque, control)
    bit("── BLOQUE %s (%s) · cyc%d · participantes=%s · dirige=%s ──" % (bloque, control, cyc, "".join(orgs), driver))
    t0=time.time(); Aar=[]; Ear=[]; sends=0; llega=0
    prox_send=t0; prox_muestra=t0
    while time.time()-t0 < dur and time.time()<END:
        ahora=time.time()
        # conducir tráfico digital
        if driver in ("A2E","BIDIR") and ahora>=prox_send:
            fa=fila(A_WEB); tk=token(fa,"A")
            if control=="shuffled": tk=shuffle_tok(tk)
            if control!="null":
                if a_tx(tk): sends+=1
                bit("  A→E  %s" % tk)
        if driver in ("E2A","BIDIR") and ahora>=prox_send:
            fe=fila(E_WEB); tk=token(fe,"E")
            if control=="shuffled": tk=shuffle_tok(tk)
            if control!="null":
                if e_tx(tk): sends+=1
                bit("  E→A  %s" % tk)
        if ahora>=prox_send: prox_send=ahora+SEND_DT
        # muestrear estado cada 20s
        if ahora>=prox_muestra:
            fa=fila(A_WEB); fe=fila(E_WEB)
            Aar.append(float(fa.get("voz_arousal") or 0)); Ear.append(float(fe.get("voz_arousal") or 0))
            bit("    · A[ar=%.2f oido:fiab=%.2f val=%.2f mod=%.2f] E[ar=%.2f oido:fiab=%.2f val=%.2f mod=%.2f] sdrE=%s" % (
                float(fa.get("voz_arousal") or 0), float(fa.get("oido_dig_fiabilidad") or 0),
                float(fa.get("oido_dig_valor_ecologico") or 0), float(fa.get("oido_dig_modulacion") or 1),
                float(fe.get("voz_arousal") or 0), float(fe.get("oido_dig_fiabilidad") or 0),
                float(fe.get("oido_dig_valor_ecologico") or 0), float(fe.get("oido_dig_modulacion") or 1),
                fe.get("sdr_vivo")))
            if "sdr" in bloque.lower() or bloque.startswith("B02") or bloque.startswith("B10"): cuidar_sdr_e()
            prox_muestra=ahora+20
        time.sleep(0.4)
    # métricas de cierre
    fa=fila(A_WEB); fe=fila(E_WEB)
    met={"r_arousal_AE":corr(Aar,Ear),
         "A_oido_fiab":float(fa.get("oido_dig_fiabilidad") or 0),"A_oido_valor":float(fa.get("oido_dig_valor_ecologico") or 0),
         "A_oido_mod":float(fa.get("oido_dig_modulacion") or 1),"A_oido_eventos":float(fa.get("oido_dig_eventos") or 0),
         "E_oido_fiab":float(fe.get("oido_dig_fiabilidad") or 0),"E_oido_valor":float(fe.get("oido_dig_valor_ecologico") or 0),
         "E_oido_mod":float(fe.get("oido_dig_modulacion") or 1),"E_oido_eventos":float(fe.get("oido_dig_eventos") or 0),
         "sends":sends,"A_OI":float(fa.get("OI") or 0),"E_OI":float(fe.get("OI") or 0),
         "A_H":float(fa.get("H_homeostasis") or 0),"E_H":float(fe.get("H_homeostasis") or 0)}
    snap_row(bloque,control,cyc,met)
    bit("── FIN %s · r(arousal A,E)=%.3f · A_oido[fiab=%.2f val=%.2f] E_oido[fiab=%.2f val=%.2f] · envíos=%d ──" % (
        bloque, met["r_arousal_AE"], met["A_oido_fiab"], met["A_oido_valor"], met["E_oido_fiab"], met["E_oido_valor"], sends))
    # limpiar tags
    tag(orgs,"","real")

BLOQUES=[
    ("B00_basal",           "passive","real", ["A","E"]),
    ("B04_digital_A_a_E",   "A2E",    "real", ["A","E"]),
    ("B04c_shuffled_A_a_E", "A2E",    "shuffled",["A","E"]),
    ("B05_digital_E_a_A",   "E2A",    "real", ["A","E"]),
    ("B06_digital_bidir",   "BIDIR",  "real", ["A","E"]),
    ("B06c_null_bidir",     "BIDIR",  "null", ["A","E"]),
    ("B07_audio_canales",   "passive","real", ["A","B","C","D","E"]),
    ("B09_sociedad_ABCD",   "passive","real", ["A","B","C","D"]),
    ("B10_todos_con_todos", "BIDIR",  "real", ["A","B","C","D","E"]),
]

def main():
    bit("=== CAMPAÑA DE ESTRÉS AUTÓNOMA — inicio ===  ciclo=%s · fin previsto=%s" % (CICLO, time.strftime("%H:%M",time.localtime(END))))
    bit("Enlace: A=%s E=%s · bitácora=%s · snapshots=%s" % (A_WEB,E_WEB,os.path.basename(BITA),os.path.basename(SNAP)))
    cyc=0
    while time.time()<END:
        cyc+=1
        bit("========== CICLO %d ==========" % cyc)
        for nombre,driver,control,orgs in BLOQUES:
            if time.time()>=END: break
            try:
                correr_bloque(nombre, DUR, driver, control, orgs, cyc)
            except Exception as e:
                bit("  [ERROR] bloque %s: %s" % (nombre, e))
            # basal entre bloques
            t0=time.time()
            while time.time()-t0<BASAL and time.time()<END: time.sleep(1)
    bit("=== CAMPAÑA TERMINADA === ciclos=%d" % cyc)

if __name__=="__main__":
    main()
