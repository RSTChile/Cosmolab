#!/usr/bin/env python3
# Test: A y E se comunican SOLO por la radio digital (nRF24). Cada uno emite un token de su
# estado expresivo; el harness lo relaya únicamente por el canal digital. Observamos qué pasa.
import urllib.request, json, time

A_WEB="http://localhost:7788"; A_BR="http://192.168.86.250:8772"
E_WEB="http://192.168.86.33:7788"
N=24; PERIODO=1.4

def GET(u,t=5):
    with urllib.request.urlopen(u,timeout=t) as r: return json.loads(r.read().decode())
def fila(web):
    j=GET(web+"/ultima_fila"); c=j.get("cols",[]); f=j.get("fila") or []
    return f if isinstance(f,dict) else dict(zip(c,f))
def a_tx(t):
    try: urllib.request.urlopen(urllib.request.Request(A_BR+"/tx",data=t.encode("ascii","ignore"),method="POST"),timeout=5).read()
    except: pass
def e_tx(t):
    try: urllib.request.urlopen(urllib.request.Request(E_WEB+"/nrf/tx",data=json.dumps({"text":t}).encode(),headers={"Content-Type":"application/json"},method="POST"),timeout=6).read()
    except: pass
def rx(get_url):
    try: return str(GET(get_url).get("nrf_last_rx","")).strip()
    except: return ""

def token(d, who):
    """Codifica el estado expresivo en <=20 chars, sin espacios de borde."""
    vocal = 1 if float(d.get("expr_vocalizando") or 0) > 0.5 else 0
    vid = str(d.get("voz_id") or "0")[:2]
    ar = int(max(0,min(1,float(d.get("voz_arousal") or 0)))*99)
    va = int((max(-1,min(1,float(d.get("mem_valencia_estado") or 0)))+1)/2*99)
    return f"{who}{vocal}w{vid}a{ar:02d}v{va:02d}"   # p.ej. A1w15a26v60

def wait_rx(get_url, exp, to=2.5):
    t0=time.time()
    while time.time()-t0<to:
        if rx(get_url)==exp: return True
        time.sleep(0.1)
    return False

def main():
    print("=== A y E SOLO por radio digital (nRF24) — qué emite y qué recibe cada uno ===")
    print("token = <quien><vocaliza>w<voz_id>a<arousal>v<valencia>  (0-99)\n")
    print(f"  {'ci':>2} | {'A emite':<14} {'→E ok':<6} | {'E emite':<14} {'→A ok':<6} | Aar Ava  Ear Eva")
    okAE=okEA=0; serieA=[]; serieE=[]; conv=[]
    for i in range(N):
        try: dA=fila(A_WEB); dE=fila(E_WEB)
        except Exception as e:
            print(f"  {i:>2} | (fila no disponible: {e})"); time.sleep(PERIODO); continue
        tA=token(dA,"A"); tE=token(dE,"E")
        a_tx(tA); gAE=wait_rx(E_WEB+"/nrf", tA)          # A -> E
        e_tx(tE); gEA=wait_rx(A_BR+"/nrf", tE)           # E -> A
        okAE+=gAE; okEA+=gEA
        Aar=float(dA.get("voz_arousal") or 0); Ava=float(dA.get("mem_valencia_estado") or 0)
        Ear=float(dE.get("voz_arousal") or 0); Eva=float(dE.get("mem_valencia_estado") or 0)
        serieA.append(Aar); serieE.append(Ear); conv.append((tA,gAE,tE,gEA))
        print(f"  {i:>2} | {tA:<14} {'✓' if gAE else '✗':<6} | {tE:<14} {'✓' if gEA else '✗':<6} | {Aar:.2f} {Ava:+.2f} {Ear:.2f} {Eva:+.2f}")
        time.sleep(PERIODO)
    # análisis
    def corr(x,y):
        n=len(x); mx=sum(x)/n; my=sum(y)/n
        num=sum((a-mx)*(b-my) for a,b in zip(x,y))
        dx=sum((a-mx)**2 for a in x)**.5; dy=sum((b-my)**2 for b in y)**.5
        return num/(dx*dy) if dx>0 and dy>0 else 0.0
    n=len(conv)
    print("\n── QUÉ SUCEDIÓ ──")
    print(f"  Entrega digital: A→E {okAE}/{n} · E→A {okEA}/{n}")
    vocA=sum(1 for t,_,_,_ in conv if t[1]=='1'); vocE=sum(1 for _,_,t,_ in conv if t[1]=='1')
    print(f"  Vocalizaciones (vocal=1): A={vocA}/{n} · E={vocE}/{n}")
    print(f"  Correlación arousal A↔E durante el intercambio: r={corr(serieA,serieE):+.3f}")
    print(f"    (r≈0 ⇒ no se acoplan: el canal digital es 'boca sin oído' — nrf_rx no entra aún en la cognición)")
    tokensA=set(t for t,_,_,_ in conv); tokensE=set(t for _,_,t,_ in conv)
    print(f"  Repertorio digital distinto: A={len(tokensA)} tokens · E={len(tokensE)} tokens")

main()
