import os,sys,json,time,importlib.util,glob
import numpy as np, soundfile as sf
AQUI=os.path.dirname(os.path.abspath(__file__)); RAIZ=os.path.dirname(AQUI)
sp=importlib.util.spec_from_file_location("rec",os.path.join(AQUI,"reclasificar_vocabulario_sintiente.py"))
rec=importlib.util.module_from_spec(sp); sp.loader.exec_module(rec)
SR=int(rec.A.SR); CAND=os.path.join(RAIZ,"voces_r2d2_candidatas")
def resample(x,f): return np.interp(np.arange(0,len(x),f),np.arange(len(x)),x)
sem={p.split("__")[1]:rec._load(os.path.join(RAIZ,"voces_r2d2",p)) for p in os.listdir(os.path.join(RAIZ,"voces_r2d2")) if p.startswith("CERRADO__Cierre__")}
base=list(sem.values())[0] if sem else None
print("criando desde Cierre...",flush=True); res={}; t0=time.time()
variantes={}
for f in (0.7,0.9,1.1,1.25,1.5,1.7): variantes[f"cria_Cierre_x{f}"]=resample(np.asarray(base,float),f)
variantes["cria_Cierre_doble"]=np.concatenate([np.asarray(base,float)]*2)
variantes["cria_Cierre_lento"]=resample(np.asarray(base,float),0.6)
for k,y in variantes.items():
    sf.write(os.path.join(CAND,k+".wav"),(y/(np.max(np.abs(y))+1e-9))*0.9,SR,subtype="PCM_16")
    r=rec._vivir_con(rec._viable(np.asarray(y,float),0.012),3,pasos=16)
    res[k]={"regimen":r["regimen"],"W":r["W"]}
    print(f"  {k:20s} {r['regimen']:14s} W={r['W']:.3f} ({time.time()-t0:.0f}s)",flush=True)
json.dump(res,open(os.path.join(AQUI,"resultado_cria_cierre.json"),"w"),indent=1)
viables=[k for k,r in res.items() if r["regimen"] in ("CERRADO","JARDIN_FERTIL")]
print(f"\n=== CRIA FIN === viables: {viables}",flush=True)
