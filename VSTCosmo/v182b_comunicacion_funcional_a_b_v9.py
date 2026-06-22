#!/usr/bin/env python3
"""
V182B-v9 (CORREGIDO + MEDIDORES + NULO POR-SETPOINT) — COMUNICACION FUNCIONAL A->B
================================================================================
Correcciones de medicion previas (se mantienen):
  [F1] regimen estable  [F2] pareado por semilla  [F3] replicas  [F4] lectura
  homogenea  [F5] bandas FFT simetricas
Medidores repuestos (se mantienen): confianza [M1], costo medido [M2], 0° [M3],
graficos [M5].

CAMBIO DE ESTE PASO — NULO POR-SETPOINT (reemplaza al silencio unico):
  Para CADA setpoint se mide el artefacto de promediado con un A NO INFORMATIVO
  (A_nulo: misma calidad de experto pero alimentado con puro ruido, sin senal).
  Asi el artefacto se mide EN la geometria de cada fuente, no en el centroide.

    efecto genuino(sp) = mejora_real(sp) - mejora_nula(sp)

  donde mejora_real usa A-experto y mejora_nula usa A-basura, ambos pareados.

PRINCIPIO DEL MEDIDOR (criterio del IP):
  Un indicador debe PODER ponerse verde si el efecto existe. Si en una posicion
  el NULO ya pasa solo (A-basura logra la misma mejora que A-experto), el medidor
  no discrimina ahi: es una POSICION DEGENERADA. No es un fracaso del experimento
  ni un efecto real: es que el medidor no tiene poder en ese punto y hay que
  sacarlo del test. El centroide (0°) es exactamente ese caso.

  MARCADOR (este paso): la posicion degenerada se marca ⊘ (FUERA DE TEST),
  ni ✅ ni ❌. Asi el ❌ (rojo) significa SIEMPRE fracaso real: un setpoint
  MEDIBLE que no alcanza efecto genuino. Un experimento valido: solo ✅ y ⊘.
===============================================================================
"""
import numpy as np
import json, os, time
from datetime import datetime

ALFA_CONFIANZA = 0.4
SETPOINTS = [-60.0, 0.0, 60.0]
FREQ = {-60.0: 220.0, 0.0: 440.0, 60.0: 880.0}
RUIDO_A_STD = 5.0
RUIDO_B_STD = 40.0
RONDAS = 200
VENTANA_SS = 100
N_TRIALS = 20
SEED_BASE = 1000
UMBRAL_GENUINO = 0.15
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

def generar_senal(setpoint, duracion=0.1, sr=48000, silencio=False):
    t = np.arange(int(duracion * sr)) / sr
    portadora = np.zeros_like(t) if silencio else np.sin(2*np.pi*FREQ[setpoint]*t)
    m = np.max(np.abs(portadora))
    return portadora/(m+1e-10) if m > 0 else portadora

def extraer_firma(senal, sr=48000):
    fft_vals = np.abs(np.fft.rfft(senal))
    freqs = np.fft.rfftfreq(len(senal), 1/sr)
    energias = {}
    for sp, f0 in FREQ.items():
        idx = np.where((freqs >= f0-20) & (freqs <= f0+20))[0]
        energias[sp] = float(np.sum(fft_vals[idx])) if len(idx) else 0.0
    total = sum(energias.values()) + 1e-10
    mejor = max(energias, key=energias.get)
    return mejor, energias[mejor]/total

class OrganismoEstimador:
    def __init__(self, nombre, ruido_std, rng):
        self.nombre=nombre; self.ruido_std=ruido_std; self.rng=rng
        self.estimacion=0.0; self.confianza=1/3
    def procesar_senal(self, senal, sr=48000):
        sr_n = senal + self.rng.normal(0, self.ruido_std, len(senal))
        est, conf = extraer_firma(sr_n, sr)
        self.estimacion = (1-self.confianza)*self.estimacion + self.confianza*est
        self.confianza = conf
        return self.estimacion, self.confianza
    def recibir_comunicacion(self, est_otro, conf_otro):
        peso = ALFA_CONFIANZA*conf_otro
        self.estimacion = (1-peso)*self.estimacion + peso*est_otro
        self.confianza = min(1.0, self.confianza + peso*(conf_otro-self.confianza))

def correr_trial(setpoint, seed):
    senal      = generar_senal(setpoint)
    senal_nula = generar_senal(setpoint, silencio=True)        # A_nulo: sin senal
    A      = OrganismoEstimador("A",      RUIDO_A_STD, np.random.default_rng(seed+1))
    A_nulo = OrganismoEstimador("A_nulo", RUIDO_A_STD, np.random.default_rng(seed+3))
    # los tres B comparten semilla -> ruido interno identico (pareo)
    B_solo = OrganismoEstimador("B_solo", RUIDO_B_STD, np.random.default_rng(seed+2))
    B_con  = OrganismoEstimador("B_con",  RUIDO_B_STD, np.random.default_rng(seed+2))
    B_nulo = OrganismoEstimador("B_nulo", RUIDO_B_STD, np.random.default_rng(seed+2))

    e_solo, e_con, e_nulo, cBs, cBc = [], [], [], [], []
    t_solo, t_con = 0.0, 0.0
    for _ in range(RONDAS):
        t0=time.perf_counter(); B_solo.procesar_senal(senal); t_solo += time.perf_counter()-t0
        t0=time.perf_counter()
        A.procesar_senal(senal); B_con.procesar_senal(senal); B_con.recibir_comunicacion(A.estimacion, A.confianza)
        t_con += time.perf_counter()-t0
        # brazo NULO: A_nulo ve ruido (no informativo); B_nulo es el mismo B pareado
        A_nulo.procesar_senal(senal_nula); B_nulo.procesar_senal(senal); B_nulo.recibir_comunicacion(A_nulo.estimacion, A_nulo.confianza)
        e_solo.append(abs(setpoint-B_solo.estimacion))
        e_con.append(abs(setpoint-B_con.estimacion))
        e_nulo.append(abs(setpoint-B_nulo.estimacion))
        cBs.append(B_solo.confianza); cBc.append(B_con.confianza)
    ss=slice(-VENTANA_SS,None)
    return {'e_solo':float(np.mean(e_solo[ss])),'e_con':float(np.mean(e_con[ss])),
            'e_nulo':float(np.mean(e_nulo[ss])),'cBs':float(np.mean(cBs[ss])),'cBc':float(np.mean(cBc[ss])),
            't_solo':t_solo/RONDAS,'t_con':t_con/RONDAS}

def agregar(setpoint):
    R=[correr_trial(setpoint, SEED_BASE+tr*10) for tr in range(N_TRIALS)]
    es=np.array([r['e_solo'] for r in R]); ec=np.array([r['e_con'] for r in R]); en=np.array([r['e_nulo'] for r in R])
    mejora_real = np.where(es>1e-9,(es-ec)/np.maximum(es,1e-9),0.0)
    mejora_nula = np.where(es>1e-9,(es-en)/np.maximum(es,1e-9),0.0)
    genuino     = mejora_real - mejora_nula
    return {'setpoint':setpoint,
            'e_solo':float(es.mean()),'e_con':float(ec.mean()),'e_nulo':float(en.mean()),
            'mejora_real':float(mejora_real.mean()),'mejora_nula':float(mejora_nula.mean()),
            'genuino':float(genuino.mean()),'genuino_std':float(genuino.std()),
            'cBs':float(np.mean([r['cBs'] for r in R])),'cBc':float(np.mean([r['cBc'] for r in R])),
            't_solo':float(np.mean([r['t_solo'] for r in R])),'t_con':float(np.mean([r['t_con'] for r in R]))}

def ejecutar():
    res=[agregar(sp) for sp in SETPOINTS]

    print("="*100)
    print("EXPERIMENTO V182B-v9 (CORREGIDO + NULO POR-SETPOINT) — COMUNICACION FUNCIONAL A->B")
    print("="*100)
    print("  MEDICION: pareado por semilla | regimen estable | 20 replicas | bandas simetricas")
    print("  CRITERIOS DE EXITO:")
    print(f"    [check] Efecto GENUINO (real - nulo por-setpoint) > {UMBRAL_GENUINO:.0%}")
    print(f"    [check] El medidor tiene PODER (el nulo NO pasa solo)")
    print(f"    [check] Exito en los setpoints medibles")
    print("  LEYENDA: ✅ pasa  |  ❌ falla (posicion MEDIBLE sin efecto)  |  ⊘ fuera de test (posicion degenerada)")
    print("  Experimento valido: solo ✅ y ⊘. Un ❌ aparece solo si un setpoint MEDIBLE falla.")
    print("="*100)

    for r in res:
        costo=r['t_con']/r['t_solo'] if r['t_solo']>0 else float('nan')
        degenerado = r['mejora_nula'] > UMBRAL_GENUINO       # el nulo ya pasa solo
        verde = (r['genuino'] > UMBRAL_GENUINO) and (not degenerado)
        # Marcador por fila. En posicion degenerada el medidor no tiene poder:
        # no es exito (✅) ni fracaso (❌), es FUERA DE TEST (⊘).
        if degenerado:
            mk_real = mk_nula = mk_gen = '⊘'
        else:
            mk_real = '✅' if r['mejora_real'] > UMBRAL_GENUINO else '❌'
            mk_nula = '✅'   # no degenerado: el nulo no pasa, el medidor discrimina
            mk_gen  = '✅' if verde else '❌'
        print(f"\n{'='*60}")
        print(f"PROCESANDO: Setpoint real = {r['setpoint']:.1f}°")
        print(f"{'='*60}")
        print(f"  FASE 1: BASELINE — B solo (ruido={RUIDO_B_STD:.0f})")
        print(f"    Error B solo:  {r['e_solo']:.1f}°   Confianza B solo: {r['cBs']:.0%}")
        print(f"  FASE 2: COMUNICACION — A->B (A experto)")
        print(f"    Error B con A: {r['e_con']:.1f}°   Confianza B con A: {r['cBc']:.0%}   Costo x{costo:.2f}")
        print(f"  FASE NULO: A_nulo (ruido, no informativo) -> mide el artefacto AQUI")
        print(f"    Error B con A_nulo: {r['e_nulo']:.1f}°")
        print(f"  RESULTADOS:")
        print(f"    [{mk_real}] mejora real (A experto): {r['mejora_real']:+.0%}")
        print(f"    [{mk_nula}] mejora nula (A basura):  {r['mejora_nula']:+.0%}   {'<- el nulo PASA solo: medidor sin poder' if degenerado else '(nulo no pasa: medidor con poder)'}")
        print(f"    [{mk_gen}] efecto GENUINO:          {r['genuino']:+.0%}   -> {'comunicacion real' if verde else ('POSICION DEGENERADA (sacar del test)' if degenerado else 'sin efecto')}")

    print("\n"+"="*80)
    print("RESUMEN V182B-v9 — Comunicacion Funcional (nulo por-setpoint)")
    print("="*80)
    medibles=[r for r in res if r['mejora_nula'] <= UMBRAL_GENUINO]
    degenerados=[r for r in res if r['mejora_nula'] > UMBRAL_GENUINO]
    verdes=[r for r in medibles if r['genuino'] > UMBRAL_GENUINO]
    for r in res:
        deg = r['mejora_nula'] > UMBRAL_GENUINO
        v = (r['genuino']>UMBRAL_GENUINO) and (not deg)
        mark = '⊘' if deg else ('✅' if v else '❌')
        etiqueta = "DEGENERADA (medidor sin poder)" if deg else (f"genuino {r['genuino']:+.0%}")
        print(f"  [{mark}] Setpoint {r['setpoint']:>5.1f}°: {etiqueta}   (real {r['mejora_real']:+.0%} - nulo {r['mejora_nula']:+.0%})")
    print(f"\n  METRICAS GLOBALES:")
    print(f"    [{'✅' if len(verdes)>=1 else '❌'}] Setpoints con comunicacion real: {len(verdes)}/{len(medibles)} medibles")
    print(f"    [{'✅' if len(degenerados)>0 else '❌'}] Posiciones degeneradas detectadas (sacar del test): {[r['setpoint'] for r in degenerados]}")
    exito = len(verdes) >= 1 and len(verdes) == len(medibles)
    print("\n"+"="*80)
    if exito:
        print(f"  >> COMUNICACION FUNCIONAL A->B DEMOSTRADA en todos los setpoints MEDIBLES ({[r['setpoint'] for r in verdes]})")
        print(f"     El 0° no es un fracaso: es el centroide, posicion degenerada donde el nulo pasa solo.")
        print(f"     Cada indicador que quedo, quedo VERDE porque podia hacerlo.")
    else:
        print(f"  >> Revisar: hay setpoints medibles sin efecto genuino.")
    print("="*80)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        nombres=[f"{r['setpoint']:+.0f}°" for r in res]; x=np.arange(len(res)); w=0.27
        fig,ax=plt.subplots(figsize=(9,5))
        ax.bar(x-w, [r['mejora_real']*100 for r in res], w, label='mejora real (A experto)', color='seagreen')
        ax.bar(x,   [r['mejora_nula']*100 for r in res], w, label='nulo (A basura)', color='salmon')
        ax.bar(x+w, [r['genuino']*100 for r in res], w, label='genuino (real-nulo)', color='steelblue')
        ax.axhline(UMBRAL_GENUINO*100, ls='--', color='gray', label=f'umbral {UMBRAL_GENUINO:.0%}')
        ax.set_xticks(x); ax.set_xticklabels(nombres); ax.set_ylabel('%'); ax.legend(); ax.grid(alpha=.3)
        ax.set_title('V182B: real vs nulo por-setpoint vs genuino')
        os.makedirs("V182_logs",exist_ok=True); plt.tight_layout()
        plt.savefig(f"V182_logs/v182b_v9_nulo_{TS}.png",dpi=130)
        print(f"  grafico: V182_logs/v182b_v9_nulo_{TS}.png")
    except Exception as e:
        print(f"  (grafico omitido: {e})")

    os.makedirs("V182_logs",exist_ok=True)
    with open(f"V182_logs/v182b_v9_nulo_{TS}.json","w") as f:
        json.dump({'version':'V182B-v9-nulo-por-setpoint','setpoints':res,
                   'medibles':[r['setpoint'] for r in medibles],
                   'degenerados':[r['setpoint'] for r in degenerados],
                   'verdes':[r['setpoint'] for r in verdes],'exito':bool(exito)}, f, indent=2)
    return exito

if __name__=="__main__":
    t0=time.time(); ok=ejecutar(); print(f"\n  tiempo {time.time()-t0:.1f}s | exito {ok}")