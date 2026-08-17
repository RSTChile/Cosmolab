import urllib.request, json, time
A="http://192.168.86.250:8772"; E="http://192.168.86.33:7788"
RAW=("Enlace digital nRF24 bajo estres: A y E se lanzan una parrafada larga por la banda "
     "ISM de 2.4 GHz para medir perdidas bajo rafaga apretada. Con ACK automatico y 15 "
     "reintentos por paquete el canal Shannon cierra el lazo sin perder un byte 0123456789")
MSG=RAW.replace(" ","_")   # a prueba de strip del parser serial
PL=22
cs=[f"{i:02d}|{MSG[i*PL:(i+1)*PL]}" for i in range((len(MSG)+PL-1)//PL)]
def GET(u):
    with urllib.request.urlopen(u,timeout=5) as r: return json.loads(r.read().decode())
def a_tx(t):
    try: urllib.request.urlopen(urllib.request.Request(A+"/tx",data=t.encode("ascii","ignore"),method="POST"),timeout=5).read()
    except: pass
def e_tx(t):
    try: urllib.request.urlopen(urllib.request.Request(E+"/nrf/tx",data=json.dumps({"text":t}).encode(),headers={"Content-Type":"application/json"},method="POST"),timeout=6).read()
    except: pass
def wait(u,exp,to=2.5):
    t0=time.time()
    while time.time()-t0<to:
        try:
            if str(GET(u).get("nrf_last_rx","")).strip()==exp: return True,time.time()-t0
        except: pass
        time.sleep(0.1)
    return False,to
print(f"MENSAJE {len(MSG)} chars → {len(cs)} paquetes · RAFAGA APRETADA (0.03s) · sin espacios de borde\n")
okE=okA=rt=0; li=[]; lv=[]; gE=[None]*len(cs); gA=[None]*len(cs)
for i,c in enumerate(cs):
    a_tx(c); oi,ti=wait(E+"/nrf",c)
    e_tx(c); ov,tv=wait(A+"/nrf",c)
    if oi: okE+=1; li.append(ti); gE[i]=c.split("|",1)[1]
    if ov: okA+=1; lv.append(tv); gA[i]=c.split("|",1)[1]
    if oi and ov: rt+=1
    print(f"  {c!r:<28} A→E:{'✓%.2fs'%ti if oi else '✗':<8} E→A:{'✓%.2fs'%tv if ov else '✗'}")
    time.sleep(0.03)
n=len(cs)
rE="".join(p for p in gE if p); rA="".join(p for p in gA if p)
print(f"\n── RESULTADO (ráfaga apretada, espacios seguros) ──")
print(f"  IDA A→E:   {okE}/{n} ({100*okE//n}%) · lat {sum(li)/len(li):.2f}s" if li else f"  IDA {okE}/{n}")
print(f"  VUELTA E→A:{okA}/{n} ({100*okA//n}%) · lat {sum(lv)/len(lv):.2f}s" if lv else f"  VUELTA {okA}/{n}")
print(f"  ROUND-TRIP completo: {rt}/{n} ({100*rt//n}%)")
print(f"  Integridad E (ida)  : {rE.replace('_',' ')==RAW}")
print(f"  Integridad A (vuelta): {rA.replace('_',' ')==RAW}")
