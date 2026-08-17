import sys, json
SRC="/Users/alexis/Desktop/RMD/Cosmolab/Cosmogenesis"
sys.path.insert(0,SRC)
from cs074_energia_holistica import correr_holistico_energia
idx=int(sys.argv[1]); s=int(sys.argv[2]); modo=sys.argv[3]
smoke=json.load(open(SRC+"/resultados_cs074D_barrido_fino/cs074D_result_smoke.json"))
cfg=[r for r in smoke["filas"] if r["idx"]==idx][0]["cfg"]
kw=dict(nq=cfg["nq"],naq=cfg["naq"],ne=cfg["ne"],npos=cfg["npos"],pasos_basal=150,
        amp_rugosidad=cfg["amp_rugosidad"],tasa_expansion=cfg["tasa_expansion"],
        E_reserva=cfg["E_reserva"],n_pasos_estructura=60,seed_layout=12345+s,guardar_curva=False)
if modo=="null": kw["seed_dens_null"]=90000+s
r=correr_holistico_energia(**kw)
json.dump(dict(idx=idx,s=s,modo=modo,ok=bool(r.get("ok")),
               v=r.get("frac_masa_ligada"),nota=r.get("nota")),
          open(f"{SRC}/verif_null/{idx}_{s}_{modo}.json","w"))
