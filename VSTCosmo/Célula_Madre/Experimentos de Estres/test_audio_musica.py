#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_audio_musica.py — AUTO-DISPARO: espera a que haya música/señal en el Rode y entonces
mide cómo responde CADA organismo (energía_L/R, OI, arousal, voz) al mundo sonoro en vivo, con
bloques música-presente vs basal. Etiqueta el CSV (exp_topologia=AUDIO_musica). Corre solo.
Uso:  ~/.venvs/vstcosmo/bin/python test_audio_musica.py   (déjalo; espera la música y arranca)."""
import urllib.request, json, time, statistics, os
CARP=os.path.dirname(os.path.abspath(__file__))
ORG={"A":"http://localhost:7788","B":"http://localhost:7799","C":"http://localhost:7810",
     "D":"http://localhost:7820","E":"http://192.168.86.33:7788"}
DUR=180; DT=4; UMBRAL=0.01
def GET(u,t=6):
    with urllib.request.urlopen(u,timeout=t) as r: return json.loads(r.read().decode())
def fila(w):
    try:
        j=GET(w+"/ultima_fila"); f=j.get("fila"); return f if isinstance(f,dict) else dict(zip(j.get("cols",[]),f or []))
    except Exception: return {}
def niveles(w):
    try:
        d=GET(w+"/niveles"); L=d if isinstance(d,list) else d.get("niveles",[])
        return max([v for v in L if isinstance(v,(int,float))] or [0])
    except Exception: return 0
def tag(topo,control):
    for w in ORG.values():
        try: urllib.request.urlopen(urllib.request.Request(w+"/exp_tag",data=json.dumps({"exp_ciclo":"AUDIO_2026-07-05","exp_topologia":topo,"exp_control":control}).encode(),headers={"Content-Type":"application/json"},method="POST"),timeout=4).read()
        except Exception: pass
OUT=os.path.join(CARP,"resultado_audio_musica_2026-07-05.md")
def log(m):
    with open(OUT,"a") as f: f.write(m+"\n"); print(m,flush=True)

def bloque(nombre, dur):
    tag(nombre, "real" if "musica" in nombre else "basal")
    log("\n## Bloque %s (%s)" % (nombre, time.strftime("%H:%M:%S")))
    agg={o:{"eL":[],"eR":[],"OI":[],"ar":[]} for o in ORG}
    t0=time.time()
    while time.time()-t0<dur:
        for o,w in ORG.items():
            f=fila(w)
            if f:
                agg[o]["eL"].append(float(f.get("energia_L") or 0)); agg[o]["eR"].append(float(f.get("energia_R") or 0))
                agg[o]["OI"].append(float(f.get("OI") or 0)); agg[o]["ar"].append(float(f.get("voz_arousal") or 0))
        time.sleep(DT)
    log("| org | energia_L | energia_R | OI | arousal |")
    log("|---|---|---|---|---|")
    res={}
    for o in ORG:
        m=lambda k: statistics.mean(agg[o][k]) if agg[o][k] else 0
        res[o]=(m("eL"),m("eR"),m("OI"),m("ar"))
        log("| %s | %.3f | %.3f | %.3f | %.3f |" % (o,*res[o]))
    return res

def main():
    log("# Test Audio / Música en vivo (Rode) — %s\n" % time.strftime("%Y-%m-%d %H:%M"))
    log("Espera a que haya señal en el Rode; entonces mide basal(silencio-relativo)→música→basal.\n")
    # 1) basal previo (lo que hay ahora)
    log("Esperando MÚSICA en el Rode (nivel > %.3f)…" % UMBRAL)
    t0=time.time()
    while niveles(ORG["A"])<UMBRAL:
        if time.time()-t0>3600: log("(1 h sin música — abandono)"); tag("","basal"); return
        time.sleep(5)
    log("¡MÚSICA DETECTADA! (nivel %.3f) — arrancando protocolo." % niveles(ORG["A"]))
    base=bloque("AUDIO_basal_pre", 90)
    mus =bloque("AUDIO_musica", DUR)
    postb=bloque("AUDIO_basal_post", 90)
    log("\n## Δ música vs basal (media de ambos basales)")
    log("| org | Δenergia_L | Δenergia_R | ΔOI | Δarousal | ¿responde? |")
    log("|---|---|---|---|---|---|")
    for o in ORG:
        b=[(base[o][i]+postb[o][i])/2 for i in range(4)]
        d=[mus[o][i]-b[i] for i in range(4)]
        resp = "SÍ" if abs(d[0])+abs(d[1])>0.02 else "leve"
        log("| %s | %+.3f | %+.3f | %+.3f | %+.3f | %s |" % (o,d[0],d[1],d[2],d[3],resp))
    tag("","basal")
    log("\n**Lectura:** Δ>0 en energía_L/R con la música ⇒ el organismo INCORPORA el mundo sonoro;")
    log("cambios en OI/arousal ⇒ el afecto responde a la música. Δ≈0 ⇒ el canal no llega a ese oído.")

if __name__=="__main__": main()
