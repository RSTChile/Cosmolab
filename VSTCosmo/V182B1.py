#!/usr/bin/env python3
"""
V182B.1 — COMUNICACION BIDIRECCIONAL (A <-> B)
================================================================================
Completa lo que V182B dejo a medias: V182B demostro comunicacion funcional
UNIDIRECCIONAL (A experto -> B alumno). Aqui se prueba el ida-y-vuelta: que B
tambien mueva a A, y que cada direccion cargue contenido GENUINO.

DOS MECANISMOS EN EL MISMO SCRIPT (se comparan, no se decide de antemano):
  (a) EXPERTISE POR BANDA  — sigma depende del setpoint:
        A experto en -60° (sigma=5) y novato en +60° (sigma=40); B al reves.
        "Bidireccional" aqui = mutualidad por contexto: cada uno es experto del
        otro en su banda. Se espera A->B genuino en -60° y B->A genuino en +60°.
  (b) CANALES INDEPENDIENTES — ambos con sigma=25, ruido INDEPENDIENTE:
        ninguno experto; cada uno ve la fuente con su propio ruido. "Bidireccional"
        aqui = fusion sensorial: juntar dos vistas baja varianza en ambos sentidos.

MEDICION (heredada y validada de V182B):
  pareado por semilla | regimen estable (ventana 100) | 20 replicas |
  bandas FFT simetricas | lectura homogenea.

DOBLE NULO POR DIRECCION:
  Cada direccion tiene su propio control nulo, y el nulo PARTICIPA DEL LOOP
  (sensa silencio, pero recibe al socio y le rebota lo que oyo). Asi el nulo
  reproduce la incestualidad de datos (el rebote) sin aportar informacion real:

      genuino(dir) = mejora_real(dir) - mejora_nula(dir)

  Si una direccion gana solo por rebote, real ≈ nulo y el genuino colapsa a ≈0.

CRITERIO DE BIDIRECCIONALIDAD:
  Bidireccional DEMOSTRADA = A->B y B->A cargan genuino (>UMBRAL) cada una en
  al menos un setpoint MEDIBLE.

MARCADORES:
  ✅ pasa (direccion medible con efecto genuino).
  ❌ falla (direccion MEDIBLE sin efecto genuino) -> NEGATIVO REAL, no artefacto.
  ⊘ fuera de test (medidor sin poder). DOS causas distintas, ambas degeneradas:
      - CENTROIDE: el nulo pasa solo; la geometria no discrimina (caso 0°).
      - RECEPTOR SATURADO: el receptor ya es experto (error solo ≈0). No hay
        headroom: el medidor no PUEDE ponerse verde porque no se mejora lo
        perfecto; solo puede quedarse igual o ir a negativo si el otro lo arrastra.
        Por el criterio del IP (un indicador debe PODER ponerse verde si el efecto
        existe), esa celda esta fuera de test, igual que el centroide.
================================================================================
"""
import numpy as np
import json, os, time
from datetime import datetime

ALFA_CONFIANZA = 0.4
SETPOINTS = [-60.0, 0.0, 60.0]
FREQ = {-60.0: 220.0, 0.0: 440.0, 60.0: 880.0}
RONDAS = 200
VENTANA_SS = 100
N_TRIALS = 20
SEED_BASE = 1000
UMBRAL_GENUINO = 0.15
UMBRAL_SATURACION = 5.0   # error-solo del receptor < esto -> saturado (sin headroom)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# Mecanismo (a): expertise por banda (sigma por setpoint)
RUIDO_A_BANDA = {-60.0: 5.0, 0.0: 40.0, 60.0: 40.0}   # A experto en -60°
RUIDO_B_BANDA = {-60.0: 40.0, 0.0: 40.0, 60.0: 5.0}   # B experto en +60°
# Mecanismo (b): canales independientes (mismo sigma, ruido independiente)
RUIDO_FUSION = 25.0

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
    """Cuerpo EXACTO de V182B. Lo unico que cambia entre mecanismos es el sigma."""
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

def correr_trial(setpoint, seed, sigma_A, sigma_B):
    senal      = generar_senal(setpoint)
    senal_nula = generar_senal(setpoint, silencio=True)   # sensor del nulo: silencio

    # Pareo por semilla: familia-A (sensa senal) comparte ruido entre sus copias;
    # idem familia-B; los nulos llevan su propia semilla.
    sA, sB, sAn, sBn = seed+1, seed+2, seed+3, seed+4
    A_solo = OrganismoEstimador("A_solo", sigma_A, np.random.default_rng(sA))
    B_solo = OrganismoEstimador("B_solo", sigma_B, np.random.default_rng(sB))
    A_bi   = OrganismoEstimador("A_bi",   sigma_A, np.random.default_rng(sA))
    B_bi   = OrganismoEstimador("B_bi",   sigma_B, np.random.default_rng(sB))
    A_nb   = OrganismoEstimador("A_nb",   sigma_A, np.random.default_rng(sA))   # A recibe B_nulo
    B_na   = OrganismoEstimador("B_na",   sigma_B, np.random.default_rng(sB))   # B recibe A_nulo
    A_nulo = OrganismoEstimador("A_nulo", sigma_A, np.random.default_rng(sAn))  # sensa silencio
    B_nulo = OrganismoEstimador("B_nulo", sigma_B, np.random.default_rng(sBn))  # sensa silencio

    eAs, eBs, eAbi, eBbi, eAnb, eBna = [], [], [], [], [], []
    t_solo, t_bi = 0.0, 0.0
    for _ in range(RONDAS):
        # --- BASELINE (sin comunicacion) ---
        t0=time.perf_counter()
        A_solo.procesar_senal(senal); B_solo.procesar_senal(senal)
        t_solo += time.perf_counter()-t0

        # --- LOOP BIDIRECCIONAL REAL (A_bi <-> B_bi, simultaneo) ---
        t0=time.perf_counter()
        A_bi.procesar_senal(senal); B_bi.procesar_senal(senal)
        eA, cA = A_bi.estimacion, A_bi.confianza      # se leen pre-update (simetrico)
        eB, cB = B_bi.estimacion, B_bi.confianza
        A_bi.recibir_comunicacion(eB, cB)             # B -> A
        B_bi.recibir_comunicacion(eA, cA)             # A -> B
        t_bi += time.perf_counter()-t0

        # --- NULO direccion A->B: B recibe A_nulo; A_nulo rebota a B (incesto) ---
        A_nulo.procesar_senal(senal_nula); B_na.procesar_senal(senal)
        eAn, cAn   = A_nulo.estimacion, A_nulo.confianza
        eBn2, cBn2 = B_na.estimacion, B_na.confianza
        A_nulo.recibir_comunicacion(eBn2, cBn2)       # rebote: B -> A_nulo
        B_na.recibir_comunicacion(eAn, cAn)           # A_nulo -> B

        # --- NULO direccion B->A: A recibe B_nulo; B_nulo rebota a A (incesto) ---
        B_nulo.procesar_senal(senal_nula); A_nb.procesar_senal(senal)
        eBn, cBn   = B_nulo.estimacion, B_nulo.confianza
        eAn2, cAn2 = A_nb.estimacion, A_nb.confianza
        B_nulo.recibir_comunicacion(eAn2, cAn2)       # rebote: A -> B_nulo
        A_nb.recibir_comunicacion(eBn, cBn)           # B_nulo -> A

        # --- registrar errores ---
        eAs.append(abs(setpoint-A_solo.estimacion)); eBs.append(abs(setpoint-B_solo.estimacion))
        eAbi.append(abs(setpoint-A_bi.estimacion));  eBbi.append(abs(setpoint-B_bi.estimacion))
        eAnb.append(abs(setpoint-A_nb.estimacion));  eBna.append(abs(setpoint-B_na.estimacion))

    ss=slice(-VENTANA_SS,None)
    return {'eAs':float(np.mean(eAs[ss])), 'eBs':float(np.mean(eBs[ss])),
            'eAbi':float(np.mean(eAbi[ss])),'eBbi':float(np.mean(eBbi[ss])),
            'eAnb':float(np.mean(eAnb[ss])),'eBna':float(np.mean(eBna[ss])),
            't_solo':t_solo/RONDAS,'t_bi':t_bi/RONDAS}

def agregar(setpoint, sigma_A, sigma_B):
    R=[correr_trial(setpoint, SEED_BASE+tr*10, sigma_A, sigma_B) for tr in range(N_TRIALS)]
    def arr(k): return np.array([r[k] for r in R])
    eAs, eBs   = arr('eAs'),  arr('eBs')
    eAbi, eBbi = arr('eAbi'), arr('eBbi')
    eAnb, eBna = arr('eAnb'), arr('eBna')
    # Direccion A->B (receptor B): real = solo vs loop ; nulo = solo vs (B recibe A_nulo)
    real_AB = np.where(eBs>1e-9,(eBs-eBbi)/np.maximum(eBs,1e-9),0.0)
    nula_AB = np.where(eBs>1e-9,(eBs-eBna)/np.maximum(eBs,1e-9),0.0)
    gen_AB  = real_AB - nula_AB
    # Direccion B->A (receptor A): real = solo vs loop ; nulo = solo vs (A recibe B_nulo)
    real_BA = np.where(eAs>1e-9,(eAs-eAbi)/np.maximum(eAs,1e-9),0.0)
    nula_BA = np.where(eAs>1e-9,(eAs-eAnb)/np.maximum(eAs,1e-9),0.0)
    gen_BA  = real_BA - nula_BA
    return {'setpoint':setpoint,'sigma_A':float(sigma_A),'sigma_B':float(sigma_B),
            'eAs':float(eAs.mean()),'eBs':float(eBs.mean()),
            'eAbi':float(eAbi.mean()),'eBbi':float(eBbi.mean()),
            'eAnb':float(eAnb.mean()),'eBna':float(eBna.mean()),
            'real_AB':float(real_AB.mean()),'nula_AB':float(nula_AB.mean()),'gen_AB':float(gen_AB.mean()),
            'real_BA':float(real_BA.mean()),'nula_BA':float(nula_BA.mean()),'gen_BA':float(gen_BA.mean()),
            't_solo':float(np.mean([r['t_solo'] for r in R])),'t_bi':float(np.mean([r['t_bi'] for r in R]))}

def estado_direccion(saturado, centroide, genuino):
    """Devuelve (marcador, medible, verde). Glifos literales, sin ocultar nada."""
    degenerado = saturado or centroide
    verde = (genuino > UMBRAL_GENUINO) and (not degenerado)
    marcador = '⊘' if degenerado else ('✅' if verde else '❌')
    return marcador, (not degenerado), verde

def ejecutar_mecanismo(clave, nombre, sigma_de):
    print("\n"+"#"*100)
    print(f"#  MECANISMO ({clave}): {nombre}")
    print("#"*100)
    res=[agregar(sp, *sigma_de(sp)) for sp in SETPOINTS]
    for r in res:
        sat_AB = r['eBs'] < UMBRAL_SATURACION   # receptor B ya experto
        cen_AB = r['nula_AB'] > UMBRAL_GENUINO   # centroide
        sat_BA = r['eAs'] < UMBRAL_SATURACION   # receptor A ya experto
        cen_BA = r['nula_BA'] > UMBRAL_GENUINO
        mk_AB, _, v_AB = estado_direccion(sat_AB, cen_AB, r['gen_AB'])
        mk_BA, _, v_BA = estado_direccion(sat_BA, cen_BA, r['gen_BA'])
        costo = r['t_bi']/r['t_solo'] if r['t_solo']>0 else float('nan')
        print(f"\n{'='*64}")
        print(f"PROCESANDO: Setpoint = {r['setpoint']:+.1f}°   (sigma_A={r['sigma_A']:.0f}, sigma_B={r['sigma_B']:.0f})")
        print(f"{'='*64}")
        print(f"  FASE BASELINE (sin comunicacion)")
        print(f"    Error A solo: {r['eAs']:.1f}°    Error B solo: {r['eBs']:.1f}°")
        print(f"  FASE BIDIRECCIONAL (A <-> B, ida y vuelta cada ronda)")
        print(f"    Error A en loop: {r['eAbi']:.1f}°    Error B en loop: {r['eBbi']:.1f}°    Costo x{costo:.2f}")
        print(f"  FASE NULO (uno por direccion, con rebote -> mide el artefacto)")
        print(f"    Error B con A_nulo: {r['eBna']:.1f}°    Error A con B_nulo: {r['eAnb']:.1f}°")
        print(f"  RESULTADOS:")
        if sat_AB:
            print(f"    [{mk_AB}] A->B: receptor B ya experto (solo {r['eBs']:.1f}°) -> FUERA DE TEST (saturado)")
        elif cen_AB:
            print(f"    [{mk_AB}] A->B genuino: {r['gen_AB']:+.0%}   (real {r['real_AB']:+.0%} - nulo {r['nula_AB']:+.0%})   -> FUERA DE TEST (centroide)")
        else:
            print(f"    [{mk_AB}] A->B genuino: {r['gen_AB']:+.0%}   (real {r['real_AB']:+.0%} - nulo {r['nula_AB']:+.0%})   -> {'comunicacion real' if v_AB else 'sin efecto genuino'}")
        if sat_BA:
            print(f"    [{mk_BA}] B->A: receptor A ya experto (solo {r['eAs']:.1f}°) -> FUERA DE TEST (saturado)")
        elif cen_BA:
            print(f"    [{mk_BA}] B->A genuino: {r['gen_BA']:+.0%}   (real {r['real_BA']:+.0%} - nulo {r['nula_BA']:+.0%})   -> FUERA DE TEST (centroide)")
        else:
            print(f"    [{mk_BA}] B->A genuino: {r['gen_BA']:+.0%}   (real {r['real_BA']:+.0%} - nulo {r['nula_BA']:+.0%})   -> {'comunicacion real' if v_BA else 'sin efecto genuino'}")
    return res

def resumen_mecanismo(clave, nombre, res):
    print(f"\n{'='*80}")
    print(f"RESUMEN MECANISMO ({clave}): {nombre}")
    print(f"{'='*80}")
    ab_medibles, ba_medibles, ab_verdes, ba_verdes = [], [], [], []
    for r in res:
        sat_AB = r['eBs'] < UMBRAL_SATURACION; cen_AB = r['nula_AB'] > UMBRAL_GENUINO
        sat_BA = r['eAs'] < UMBRAL_SATURACION; cen_BA = r['nula_BA'] > UMBRAL_GENUINO
        mk_AB, med_AB, v_AB = estado_direccion(sat_AB, cen_AB, r['gen_AB'])
        mk_BA, med_BA, v_BA = estado_direccion(sat_BA, cen_BA, r['gen_BA'])
        if med_AB: ab_medibles.append(r['setpoint'])
        if med_BA: ba_medibles.append(r['setpoint'])
        if v_AB: ab_verdes.append(r['setpoint'])
        if v_BA: ba_verdes.append(r['setpoint'])
        razon_AB = '' if med_AB else (' (saturado)' if sat_AB else ' (centroide)')
        razon_BA = '' if med_BA else (' (saturado)' if sat_BA else ' (centroide)')
        v_AB_txt = f"{r['gen_AB']:+.0%}" if med_AB else razon_AB.strip()
        v_BA_txt = f"{r['gen_BA']:+.0%}" if med_BA else razon_BA.strip()
        print(f"  Setpoint {r['setpoint']:+5.1f}°:   A->B [{mk_AB}] {v_AB_txt:<12}   B->A [{mk_BA}] {v_BA_txt}")
    ab_ok = len(ab_verdes) >= 1
    ba_ok = len(ba_verdes) >= 1
    bidir = ab_ok and ba_ok
    print(f"\n  [{'✅' if ab_ok else '❌'}] A->B carga genuino en: {ab_verdes if ab_verdes else 'ninguno'}   (medibles: {ab_medibles if ab_medibles else 'ninguno'})")
    print(f"  [{'✅' if ba_ok else '❌'}] B->A carga genuino en: {ba_verdes if ba_verdes else 'ninguno'}   (medibles: {ba_medibles if ba_medibles else 'ninguno'})")
    print(f"  [{'✅' if bidir else '❌'}] BIDIRECCIONAL (ambas direcciones con genuino) -> {'DEMOSTRADA' if bidir else 'NO demostrada'}")
    return {'clave':clave,'nombre':nombre,'bidireccional':bool(bidir),
            'ab_verdes':ab_verdes,'ba_verdes':ba_verdes,
            'ab_medibles':ab_medibles,'ba_medibles':ba_medibles}

def ejecutar():
    print("="*100)
    print("EXPERIMENTO V182B.1 — COMUNICACION BIDIRECCIONAL (A <-> B)")
    print("="*100)
    print("  Dos mecanismos en el MISMO script, comparados sin decidir de antemano cual gana:")
    print("    (a) Expertise por banda    — A experto en -60°, B experto en +60° (mutualidad por contexto)")
    print("    (b) Canales independientes — ambos sigma=25, ruido independiente (fusion sensorial)")
    print("  MEDICION: pareado por semilla | regimen estable (ventana 100) | 20 replicas | doble nulo por direccion")
    print("  LEYENDA: ✅ pasa  |  ❌ falla (direccion MEDIBLE sin efecto, NEGATIVO REAL)")
    print("           ⊘ fuera de test (degenerada): CENTROIDE (nulo pasa solo) o RECEPTOR SATURADO (ya experto, sin headroom)")
    print(f"  CRITERIO: bidireccional = A->B y B->A cargan genuino (>{UMBRAL_GENUINO:.0%}) cada una en algun setpoint medible.")
    print("            El nulo con REBOTE descuenta la incestualidad: si gana por rebote, real≈nulo y genuino->0.")
    print("="*100)

    def sigma_a(sp): return (RUIDO_A_BANDA[sp], RUIDO_B_BANDA[sp])
    def sigma_b(sp): return (RUIDO_FUSION, RUIDO_FUSION)

    res_a = ejecutar_mecanismo('a', 'Expertise por banda', sigma_a)
    sum_a = resumen_mecanismo('a', 'Expertise por banda', res_a)
    res_b = ejecutar_mecanismo('b', 'Canales independientes (fusion)', sigma_b)
    sum_b = resumen_mecanismo('b', 'Canales independientes (fusion)', res_b)

    print("\n"+"#"*100)
    print("#  COMPARACION (a) vs (b)  — el dato decide")
    print("#"*100)
    print(f"  (a) Expertise por banda:      bidireccional {'✅ DEMOSTRADA' if sum_a['bidireccional'] else '❌ NO'}")
    print(f"        A->B genuino en {sum_a['ab_verdes'] or 'ninguno'} | B->A genuino en {sum_a['ba_verdes'] or 'ninguno'}")
    print(f"  (b) Canales independientes:   bidireccional {'✅ DEMOSTRADA' if sum_b['bidireccional'] else '❌ NO'}")
    print(f"        A->B genuino en {sum_b['ab_verdes'] or 'ninguno'} | B->A genuino en {sum_b['ba_verdes'] or 'ninguno'}")
    print("  Nota: cada mecanismo es una definicion distinta de 'bidireccional'.")
    print("        (a) mutualidad por contexto: cada uno experto en su banda; donde es experto el RECEPTOR, la")
    print("            direccion sale por saturacion (no se mejora lo perfecto), no por fracaso.")
    print("        (b) fusion simetrica: si el cuerpo discreto no fusiona, el genuino lo dice sin piedad.")
    print("#"*100)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,(clave,nombre,res) in zip(axes,[('a','Expertise por banda',res_a),
                                               ('b','Canales independientes',res_b)]):
            nombres=[f"{r['setpoint']:+.0f}°" for r in res]; x=np.arange(len(res)); w=0.35
            ax.bar(x-w/2,[r['gen_AB']*100 for r in res],w,label='A→B genuino',color='steelblue')
            ax.bar(x+w/2,[r['gen_BA']*100 for r in res],w,label='B→A genuino',color='darkorange')
            ax.axhline(UMBRAL_GENUINO*100, ls='--', color='gray', label=f'umbral {UMBRAL_GENUINO:.0%}')
            ax.axhline(0, color='black', lw=0.8)
            ax.set_ylim(-30, 60)
            ax.set_xticks(x); ax.set_xticklabels(nombres); ax.set_title(f"({clave}) {nombre}")
            ax.grid(alpha=.3); ax.set_ylabel('% genuino')
        axes[0].legend()
        fig.suptitle('V182B.1: genuino por direccion — (a) expertise por banda  vs  (b) fusion')
        os.makedirs("V182_logs",exist_ok=True); plt.tight_layout()
        plt.savefig(f"V182_logs/v182b1_bidi_{TS}.png",dpi=130)
        print(f"\n  grafico: V182_logs/v182b1_bidi_{TS}.png")
    except Exception as e:
        print(f"  (grafico omitido: {e})")

    os.makedirs("V182_logs",exist_ok=True)
    with open(f"V182_logs/v182b1_bidi_{TS}.json","w") as f:
        json.dump({'version':'V182B.1-bidireccional',
                   'mecanismo_a':{'resumen':sum_a,'setpoints':res_a},
                   'mecanismo_b':{'resumen':sum_b,'setpoints':res_b}}, f, indent=2)
    print(f"  datos: V182_logs/v182b1_bidi_{TS}.json")
    return sum_a, sum_b

if __name__=="__main__":
    t0=time.time(); ejecutar(); print(f"\n  tiempo {time.time()-t0:.1f}s")