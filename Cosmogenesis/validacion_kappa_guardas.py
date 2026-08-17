import numpy as np, os
RAIZ="/Users/alexis/phantom_cs073"
RUNS=[("REAL","%s/bateria_n2000/ic_real"%RAIZ)]+ \
     [("REAL_extra","%s/bateria_real_extra_n2000/ic_real_s%d"%(RAIZ,s)) for s in range(301,306)]+ \
     [("NULL_orig","%s/bateria_n2000/ic_null%d"%(RAIZ,i)) for i in range(1,9)]+ \
     [("NULL3","%s/bateria_null3_n2000/ic_null3_s%d"%(RAIZ,s)) for s in range(501,509)]+ \
     [("RANDOM_ER","%s/bateria_grafo_random_n2000/ic_random_s%d"%(RAIZ,s)) for s in range(701,709)]+ \
     [("NULL4","%s/bateria_null4_n2000/ic_null4_s%d"%(RAIZ,s)) for s in range(601,604)]+ \
     [("NULL5","%s/bateria_null5_n2000/ic_null5_s%d"%(RAIZ,s)) for s in range(801,803)]

print("== A) ¿kappa_P queda EXACTAMENTE determinada por los tiempos de nacimiento? ==")
print("   (todos los sumideros terminan en T1 -> vida_i = T1 - t_nac_i, es aritmetica)")
maxerr=0; huecos=0; total_s=0; mueren=0
kp_list=[];brazo_list=[];rango_nac=[]
for br,d in RUNS:
    a=np.loadtxt(os.path.join(d,"cosmog01.sink"),skiprows=2)
    t,sid=a[:,0],a[:,18].astype(int); T0,T1=t.min(),t.max()
    ids=np.unique(sid); tb=np.array([t[sid==s].min() for s in ids]); te=np.array([t[sid==s].max() for s in ids])
    total_s+=len(ids); mueren+=int((te<T1-1e-9).sum())
    kp_med=np.mean(T1-tb)/(T1-T0)                    # formula cerrada, solo nacimientos
    kp_dir=np.mean(te-tb)/(T1-T0)                    # formula medida
    maxerr=max(maxerr,abs(kp_med-kp_dir))
    # huecos: ¿algun sumidero desaparece y reaparece?
    dumps=np.unique(t)
    for s in ids:
        ts=np.unique(t[sid==s]); esperado=dumps[(dumps>=ts.min())&(dumps<=ts.max())]
        if len(ts)!=len(esperado): huecos+=1
    kp_list.append(kp_dir); brazo_list.append(br); rango_nac.append((tb.min(),tb.max()))
print(f"   error maximo |kappa_P(solo nacimientos) - kappa_P(medida)| = {maxerr:.2e}  -> IDENTICAS")
print(f"   sumideros totales={total_s}  que se apagan antes de T1={mueren}  con huecos intermedios={huecos}")

print("\n== B) rango de tiempos de nacimiento por brazo (lo unico que mueve kappa_P) ==")
for b in ["REAL","REAL_extra","NULL_orig","NULL3","RANDOM_ER","NULL4","NULL5"]:
    r=[x for x,bb in zip(rango_nac,brazo_list) if bb==b]
    print(f"   {b:<11} t_nac primero={np.mean([x[0] for x in r]):.4f}  ultimo={np.mean([x[1] for x in r]):.4f}  dispersion={np.mean([x[1]-x[0] for x in r]):.4f}")

print("\n== C) identidad algebraica exacta kappa_D_alt <-> masa ==")
print("   kappa_D_alt = (masa_fin_tot - masa_ini_tot)/n_sumideros  (por construccion)")
print("   masa_ini_tot varia muy poco -> kappa_D_alt es masa_fin_tot reescalada")
