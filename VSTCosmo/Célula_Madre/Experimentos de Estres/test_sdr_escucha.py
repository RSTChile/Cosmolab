#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_sdr_escucha.py — prueba dedicada del sentido SDR (escuchar radio / barrer espectro)
para A (RSPduo) y E (RSP1). Muestrea el espectro + las métricas del órgano radio, etiqueta el
CSV (exp_topologia=SDR_escucha) y resume qué oye cada organismo y cómo responde su radio."""
import urllib.request, json, time, statistics, os
CARP=os.path.dirname(os.path.abspath(__file__))
WEB={"A":"http://localhost:7788","E":"http://192.168.86.33:7788"}
DUR=120; DT=4
def GET(u,t=6):
    with urllib.request.urlopen(u,timeout=t) as r: return json.loads(r.read().decode())
def fila(w):
    try:
        j=GET(w+"/ultima_fila"); f=j.get("fila"); return f if isinstance(f,dict) else dict(zip(j.get("cols",[]),f or []))
    except Exception: return {}
def tag(w,topo):
    try: urllib.request.urlopen(urllib.request.Request(w+"/exp_tag",data=json.dumps({"exp_ciclo":"SDR_2026-07-05","exp_topologia":topo,"exp_control":"real"}).encode(),headers={"Content-Type":"application/json"},method="POST"),timeout=5).read()
    except Exception: pass
OUT=os.path.join(CARP,"resultado_sdr_escucha_2026-07-05.md")
def log(m):
    with open(OUT,"a") as f: f.write(m+"\n"); print(m,flush=True)
log("# Test SDR — escuchar radio / barrer espectro (%s)\n" % time.strftime("%Y-%m-%d %H:%M"))
for o in ("A","E"): tag(WEB[o],"SDR_escucha")
datos={o:{"pico":[],"fpk":[],"sal":[],"est":[],"nov":[],"nz":[]} for o in WEB}
t0=time.time()
while time.time()-t0<DUR:
    for o,w in WEB.items():
        f=fila(w); esp=f.get("sdr_espectro") or []
        if isinstance(esp,list) and esp:
            mx=max(esp); pk=esp.index(mx); n=len(esp)
            fmin=float(f.get("sdr_freq_min_hz") or 0); fmax=float(f.get("sdr_freq_max_hz") or 0)
            fpk=(fmin+(fmax-fmin)*pk/n)/1e6 if fmax else 0
            d=datos[o]; d["pico"].append(mx); d["fpk"].append(fpk)
            d["sal"].append(float(f.get("radio_saliencia") or 0)); d["est"].append(float(f.get("radio_estructura") or 0))
            d["nov"].append(float(f.get("radio_novedad") or 0)); d["nz"].append(sum(1 for v in esp if v>0.1))
    time.sleep(DT)
def stat(xs): return (statistics.mean(xs),min(xs),max(xs)) if xs else (0,0,0)
log("| org | muestras | pico medio | freq dom (MHz) | bins c/señal | saliencia | estructura | novedad |")
log("|---|---|---|---|---|---|---|---|")
for o,d in datos.items():
    n=len(d["pico"]);
    if not n: log("| %s | 0 | (sin espectro) |||||||"%o); continue
    pm=stat(d["pico"]); fm=stat(d["fpk"]); sal=stat(d["sal"]); est=stat(d["est"]); nov=stat(d["nov"]); nz=stat(d["nz"])
    # ¿la freq dominante VARÍA (barrido/re-sintonía) o es estable?
    fvar=max(d["fpk"])-min(d["fpk"])
    log("| %s | %d | %.3f | %.1f (rango %.1f MHz) | %d | %.3f | %.3f | %.3f |"%(
        o,n,pm[0],fm[0],fvar,int(nz[0]),sal[0],est[0],nov[0]))
log("")
log("**Lectura:** el pico medio y los bins-con-señal miden potencia de recepción; el rango de la")
log("frecuencia dominante indica si el organismo BARRE/re-sintoniza o se queda fijo; saliencia/")
log("estructura/novedad son el SENTIDO del órgano radio (no potencia cruda) — si son >0 y varían,")
log("el organismo está EXTRAYENDO estructura del espectro, no solo midiendo energía.")
for o in ("A","E"): tag(WEB[o],"")
