#!/usr/bin/env python3
"""
V182A.1 — ROLES EMERGENTES POR MEMORIA RELACIONAL (cuerpo de campo)
================================================================================
Monta sobre el cuerpo de campo de V182A-v9 (Hemisferio / ValenciaLocal /
OrganismoCompleto IDENTICOS, verbatim). Lo unico nuevo es:

  1) MemoriaRelacional FUERTE, por contexto (banda). La que existia en V182A-v9
     era un buffer de 1 paso que NO gobernaba conducta (el peso de incorporacion
     era fijo, 0.3). Aqui la memoria relacional ES la que decide cuanto le hace
     caso un organismo al otro EN CADA BANDA, y eso adapta con la historia.

  2) Competencia EMERGENTE por exposicion diferencial (no parametrizada / no
     Shannon): A se expone mas a -60° y B mas a +60°; su valencia (competencia)
     CRECE por experiencia, no se fija con un knob. El 0° es banda compartida
     (exposicion simetrica).

  3) ROLES: en el acoplamiento, cada organismo ajusta su CONFIANZA en el otro por
     banda segun la competencia relativa observada (|val_otro| - |val_self|):
     si el otro esta mas consolidado que yo en la banda, mi confianza sube y le
     DEFIERO; si yo se mas, no. El peso de incorporacion = esa confianza.
     -> El menos competente en la banda defiere; el mas competente ancla.
        Eso es un ROL, y emerge por contexto (la comunicacion es contextual).

ABLACION (a ver que sucede):
  ON : confianza adaptativa por banda (memoria relacional gobierna).
  OFF: peso fijo 0.3 (arnes original) -> promediado simetrico, sin rol.

NOTA DE ESCALA: por defecto corre en config REDUCIDA (rapida, para ver el
mecanismo). Para la corrida de fidelidad completa en el iMac, subir DUR_* y los
conteos donde se indica (-> IMAC).
================================================================================
"""
import numpy as np
import json, os, time
from collections import deque
from datetime import datetime

# ---- constantes del cuerpo (verbatim de V182A-v9) ----
DT = 0.01
DIM_HEMISFERIO = 32
SESGO_L, SESGO_R = 0.05, -0.05
TAU_CB, CB_MAX = 10.0, 500.0
HABITO_SETPOINT = -60.0
TRAUMA_SETPOINT = 60.0
PESO_ESTIMULO = 0.3
TASA_APRENDIZAJE = 0.01
ESCALA_REWARD = 20.0
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---- config del experimento de roles ----
BANDAS = [-60.0, 0.0, 60.0]
DUR_EXP = 0.3      # -> IMAC: 1.0   (duracion real por paso de exposicion)
DUR_CPL = 0.3      # -> IMAC: 1.0   (duracion real por paso de acoplamiento)
N_EXP_FUERTE = 60  # -> IMAC: 150   (exposicion a la banda propia)
N_EXP_DEBIL  = 6   # exposicion a la banda ajena
N_EXP_COMPARTIDA = 25  # exposicion a la banda compartida (0°), igual para ambos
RONDAS_CPL = 45    # -> IMAC: 300   (rondas de acoplamiento, ciclando bandas)
K_COMP = 0.40      # sensibilidad de la confianza a la competencia relativa
LR_CONF = 0.05     # velocidad de adaptacion de la confianza
CONF_MIN, CONF_MAX = 0.02, 0.95
UMBRAL_ROL = 0.20  # |conf(A->B) - conf(B->A)| > esto en una banda -> hay rol


# ============================================================
# HEMISFERIO (verbatim V182A-v9)
# ============================================================
class Hemisferio:
    def __init__(self, nombre, tau, generar_entrada_func, seed=None, sesgo=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.nombre = nombre
        self.tau = tau
        self.generar_entrada = generar_entrada_func
        self.sesgo = sesgo
        self.Phi = np.random.normal(sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.entrada = None
        self.sr = 48000
        self.en_inanicion = False
        self.factor_inanicion = 1.0
        self.estimulos_externos = deque()

    def anadir_estimulo(self, valor):
        self.estimulos_externos.append(valor)

    def _calcular_omega(self):
        return np.mean(self.Phi[:DIM_HEMISFERIO])

    def entrada_t(self, t, duracion_total):
        if self.estimulos_externos:
            return self.estimulos_externos.popleft()
        if self.entrada is None:
            self.entrada = self.generar_entrada(duracion_total, self.sr)
        idx = int(t * self.sr)
        if idx >= len(self.entrada):
            return 0.0
        return self.entrada[idx] * self.factor_inanicion

    def actualizar(self, t, dt, duracion_total, otro_hemisferio=None):
        entrada = self.entrada_t(t, duracion_total)
        laplaciano = np.zeros_like(self.Phi)
        for i in range(1, DIM_HEMISFERIO - 1):
            laplaciano[i] = self.Phi[i-1] - 2*self.Phi[i] + self.Phi[i+1]
        reaccion = self.Phi * (1 - self.Phi * self.Phi)
        forzamiento = np.zeros_like(self.Phi)
        forzamiento[0] = entrada
        forzamiento[-1] = -entrada
        acoplamiento = np.zeros_like(self.Phi)
        if otro_hemisferio is not None:
            divergencia = abs(self._calcular_omega() - otro_hemisferio._calcular_omega())
            if divergencia > 0.5:
                acoplamiento = 0.01 * (otro_hemisferio.Phi - self.Phi)
        dPhi_vel = laplaciano + reaccion + forzamiento + acoplamiento
        self.Phi_vel += dPhi_vel * dt
        self.Phi += self.Phi_vel * dt
        self.Phi = np.clip(self.Phi, -1.0, 1.0)
        return {'omega': self._calcular_omega()}

    def reset(self):
        self.Phi = np.random.normal(self.sesgo, 0.1, DIM_HEMISFERIO)
        self.Phi_vel = np.zeros(DIM_HEMISFERIO)
        self.entrada = None
        self.estimulos_externos.clear()


# ============================================================
# VALENCIA LOCAL (verbatim V182A-v9)
# ============================================================
class ValenciaLocal:
    def __init__(self):
        self.valencia = {}
        self.lr = TASA_APRENDIZAJE
        self.historial = {}

    def actualizar_con_estimulo(self, setpoint, estimulo, dt, peso=PESO_ESTIMULO, recompensa=0.0):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        if key not in self.valencia:
            self.valencia[key] = 0.0
            self.historial[key] = []
        self.valencia[key] += peso * (estimulo - self.valencia[key]) * self.lr * dt
        if recompensa > 0:
            self.valencia[key] += recompensa * self.lr * dt * ESCALA_REWARD
        self.valencia[key] = np.clip(self.valencia[key], -100, 100)
        self.historial[key].append(self.valencia[key])
        return self.valencia[key]

    def get(self, setpoint):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        return self.valencia.get(key, 0.0)

    def set(self, setpoint, valor):
        key = round(setpoint/5)*5 if setpoint != 0 else 0
        self.valencia[key] = valor
        if key not in self.historial:
            self.historial[key] = []
        self.historial[key].append(valor)

    def reset(self):
        self.valencia = {}
        self.historial = {}


# ============================================================
# MEMORIA RELACIONAL FUERTE (por contexto) — LO NUEVO
# ============================================================
class MemoriaRelacional:
    """
    Modelo del OTRO, por banda, acumulado en la historia conjunta:
      - modelo[banda]:    EMA de la valencia emitida por el otro en la banda.
      - confianza[banda]: cuanto le hago caso al otro en la banda (= peso de
                          incorporacion). Adapta hacia un objetivo dado por la
                          competencia RELATIVA observada (|val_otro| - |val_self|):
                          el otro mas consolidado que yo -> confianza sube (defiero).
    Emerge de la historia; no se fija. Si nunca se actualiza, devuelve PESO_ESTIMULO
    (identico al arnes original -> sirve de rama OFF).
    """
    def __init__(self, conf_ini=PESO_ESTIMULO):
        self.conf_ini = conf_ini
        self.confianza = {}
        self.modelo = {}
        self.hist_conf = {}

    def _key(self, banda):
        return round(banda/5)*5 if banda != 0 else 0

    def confianza_en(self, banda):
        return self.confianza.get(self._key(banda), self.conf_ini)

    def actualizar(self, banda, val_otro, val_self):
        k = self._key(banda)
        if k not in self.confianza:
            self.confianza[k] = self.conf_ini
            self.modelo[k] = val_otro
            self.hist_conf[k] = []
        self.modelo[k] = 0.8*self.modelo[k] + 0.2*val_otro
        comp_rel = abs(val_otro) - abs(val_self)                 # el otro vs yo, aqui
        s = 1.0/(1.0 + np.exp(-K_COMP*comp_rel))                 # sigmoide -> [0,1]
        objetivo = CONF_MIN + (CONF_MAX - CONF_MIN)*s
        self.confianza[k] += LR_CONF*(objetivo - self.confianza[k])
        self.confianza[k] = float(np.clip(self.confianza[k], CONF_MIN, CONF_MAX))
        self.hist_conf[k].append(self.confianza[k])
        return self.confianza[k]

    def reset(self):
        self.confianza = {}
        self.modelo = {}
        self.hist_conf = {}


# ============================================================
# ORGANISMO COMPLETO (verbatim V182A-v9; cuerpo intacto)
# ============================================================
class OrganismoCompleto:
    def __init__(self, seed, nombre):
        self.nombre = nombre
        self.seed = seed

        def generar_ruido_rosa(duracion, sr):
            n = int(duracion * sr)
            ruido = np.random.normal(0, 1, n)
            fft = np.fft.rfft(ruido)
            freqs = np.fft.rfftfreq(n, 1/sr)
            filtro = 1.0 / np.sqrt(freqs + 0.01)
            fft_filtrado = fft * filtro
            ruido_rosa = np.fft.irfft(fft_filtrado, n=n)
            return ruido_rosa / (np.max(np.abs(ruido_rosa)) + 1e-10)

        def generar_clicks_poisson(duracion, tasa=0.5, sr=48000):
            n = int(duracion * sr)
            clicks = np.zeros(n)
            n_clicks = int(duracion * tasa)
            for _ in range(n_clicks):
                pos = int(np.random.exponential(1.0/tasa) * sr)
                if pos < n:
                    clicks[pos] = 1.0
            return clicks

        self.L = Hemisferio("L", 30, generar_ruido_rosa, seed, SESGO_L)
        self.R = Hemisferio("R", 300, generar_clicks_poisson, seed+100, SESGO_R)
        self.BL = Hemisferio("BL", 30, generar_ruido_rosa, seed+200, SESGO_L)
        self.BR = Hemisferio("BR", 300, generar_clicks_poisson, seed+300, SESGO_R)
        self.hemisferios = [self.L, self.R, self.BL, self.BR]
        self.Cb = 0.0
        self.D = 0.0
        self.valencia = ValenciaLocal()
        self.memoria_relacional = MemoriaRelacional()
        self.historial_valencia = []
        self.historial_Cb = []
        self.historial_D = []

    def get_valencia(self, setpoint):
        return self.valencia.get(setpoint)

    def procesar_senal(self, setpoint, dt, duracion_total):
        t = 0
        while t < duracion_total:
            for h in self.hemisferios:
                h.actualizar(t, dt, duracion_total, None)
            t += dt
        omega_L = self.L._calcular_omega(); omega_R = self.R._calcular_omega()
        gradiente = omega_L - omega_R
        self.Cb = min(CB_MAX, self.Cb + abs(gradiente) * duracion_total)
        self.Cb *= (1 - duracion_total / TAU_CB)
        val_h = self.valencia.get(HABITO_SETPOINT); val_t = self.valencia.get(TRAUMA_SETPOINT)
        self.D = min(1.0, abs(val_h - val_t) / 100.0)

    def recibir_estimulo(self, estimulo, setpoint, dt, duracion_total, peso=PESO_ESTIMULO, recompensa=0.0):
        for h in self.hemisferios:
            h.anadir_estimulo(estimulo)
        t = 0
        while t < duracion_total:
            for h in self.hemisferios:
                h.actualizar(t, dt, duracion_total, None)
            t += dt
        self.valencia.actualizar_con_estimulo(setpoint, estimulo, duracion_total, peso, recompensa)
        omega_L = self.L._calcular_omega(); omega_R = self.R._calcular_omega()
        gradiente = omega_L - omega_R
        self.Cb = min(CB_MAX, self.Cb + abs(gradiente) * duracion_total)
        self.Cb *= (1 - duracion_total / TAU_CB)
        val_h = self.valencia.get(HABITO_SETPOINT); val_t = self.valencia.get(TRAUMA_SETPOINT)
        self.D = min(1.0, abs(val_h - val_t) / 100.0)

    def obtener_resultado(self, setpoint):
        return self.valencia.get(setpoint)

    def reset(self):
        for h in self.hemisferios:
            h.reset()
        self.valencia.reset()
        self.Cb = 0.0; self.D = 0.0
        self.memoria_relacional.reset()


# ============================================================
# FASE 1 — EXPOSICION DIFERENCIAL (competencia EMERGENTE)
# ============================================================
def exponer(org, banda, n, dur):
    """Exposicion repetida a la banda con refuerzo: la valencia (competencia)
    CRECE por experiencia. No se asigna ningun valor a mano."""
    for _ in range(n):
        org.procesar_senal(banda, DT, dur)
        org.valencia.actualizar_con_estimulo(banda, org.valencia.get(banda),
                                             dur, peso=PESO_ESTIMULO, recompensa=1.0)

def fase_exposicion(A, B):
    # A experto emergente en -60°, B en +60°, ambos exposicion simetrica en 0°
    exponer(A, -60.0, N_EXP_FUERTE, DUR_EXP); exponer(A, 60.0, N_EXP_DEBIL, DUR_EXP)
    exponer(B,  60.0, N_EXP_FUERTE, DUR_EXP); exponer(B, -60.0, N_EXP_DEBIL, DUR_EXP)
    exponer(A, 0.0, N_EXP_COMPARTIDA, DUR_EXP); exponer(B, 0.0, N_EXP_COMPARTIDA, DUR_EXP)


# ============================================================
# FASE 2 — ACOPLAMIENTO CON ROLES (memoria relacional gobierna)
# ============================================================
def acoplar(A, B, usar_memoria_relacional):
    """Cicla bandas. En cada ronda, A y B emiten su valencia en la banda; cada uno
    ajusta su confianza en el otro (si memoria ON) y la usa como peso de incorporacion."""
    traza = {b: {'cAB': [], 'cBA': [], 'difA': [], 'vA': [], 'vB': []} for b in BANDAS}
    for ronda in range(RONDAS_CPL):
        for b in BANDAS:
            vA = A.obtener_resultado(b); vB = B.obtener_resultado(b)
            if usar_memoria_relacional:
                cAB = A.memoria_relacional.actualizar(b, vB, vA)   # confianza de A en B (banda b)
                cBA = B.memoria_relacional.actualizar(b, vA, vB)   # confianza de B en A
            else:
                cAB = cBA = PESO_ESTIMULO                          # peso fijo (arnes original)
            A.recibir_estimulo(vB, b, DT, DUR_CPL, peso=cAB)       # A incorpora a B ponderado por su confianza
            B.recibir_estimulo(vA, b, DT, DUR_CPL, peso=cBA)
            traza[b]['cAB'].append(cAB); traza[b]['cBA'].append(cBA)
            traza[b]['vA'].append(A.obtener_resultado(b)); traza[b]['vB'].append(B.obtener_resultado(b))
            traza[b]['difA'].append(abs(A.obtener_resultado(b) - B.obtener_resultado(b)))
    return traza


# ============================================================
# METRICAS DE ROLES
# ============================================================
def analizar(A, B, traza):
    filas = []
    for b in BANDAS:
        compA = abs(A.obtener_resultado(b)); compB = abs(B.obtener_resultado(b))
        cAB = traza[b]['cAB'][-1]; cBA = traza[b]['cBA'][-1]   # confianza final en cada sentido
        # quien DEFIERE = quien tiene MAS confianza en el otro (le hace mas caso)
        # quien ANCLA   = el lider de la banda
        if abs(cAB - cBA) <= UMBRAL_ROL:
            rol = "simetrico (sin rol)"; lider = "—"
        elif cAB > cBA:
            rol = "A defiere -> B ancla"; lider = "B"
        else:
            rol = "B defiere -> A ancla"; lider = "A"
        experto = "A" if compA > compB else ("B" if compB > compA else "—")
        coincide = (lider == experto) if lider in ("A", "B") and experto in ("A", "B") else (lider == "—" and experto == "—")
        filas.append({'banda': b, 'compA': compA, 'compB': compB,
                      'cAB': cAB, 'cBA': cBA, 'rol': rol, 'lider': lider,
                      'experto': experto, 'coincide': bool(coincide),
                      'dif_final': traza[b]['difA'][-1]})
    return filas


def imprimir(titulo, filas):
    print(f"\n{'='*88}")
    print(titulo)
    print('='*88)
    print(f"  {'banda':>6} | {'|valA|':>6} {'|valB|':>6} | conf A->B  conf B->A | {'rol emergente':<22} | experto coincide")
    print(f"  {'-'*6}-+-{'-'*13}-+-{'-'*19}-+-{'-'*22}-+-{'-'*16}")
    for f in filas:
        mk = '✅' if f['coincide'] else '❌'
        print(f"  {f['banda']:>+6.0f} | {f['compA']:>6.2f} {f['compB']:>6.2f} |   {f['cAB']:>5.2f}      {f['cBA']:>5.2f}   | {f['rol']:<22} |   {f['experto']:^3}    [{mk}]")


def ejecutar(usar_memoria_relacional, etiqueta):
    A = OrganismoCompleto(seed=44, nombre="A")
    B = OrganismoCompleto(seed=77, nombre="B")
    fase_exposicion(A, B)
    # competencia emergente tras exposicion (antes de acoplar)
    comp_ini = {b: (abs(A.obtener_resultado(b)), abs(B.obtener_resultado(b))) for b in BANDAS}
    traza = acoplar(A, B, usar_memoria_relacional)
    filas = analizar(A, B, traza)
    print(f"\n  [competencia EMERGENTE tras exposicion diferencial — |valencia| por banda]")
    for b in BANDAS:
        print(f"     {b:>+5.0f}°:  A={comp_ini[b][0]:.2f}   B={comp_ini[b][1]:.2f}")
    imprimir(f"RESULTADO — {etiqueta}", filas)
    roles = [f for f in filas if f['lider'] in ('A', 'B')]
    coinciden = [f for f in roles if f['coincide']]
    print(f"\n  Roles emergidos: {len(roles)}/{len(BANDAS)} bandas con rol asimetrico")
    print(f"  Roles que coinciden con la competencia emergente: {len(coinciden)}/{len(roles) if roles else 0}")
    return filas, traza


def main():
    print("="*88)
    print("V182A.1 — ROLES EMERGENTES POR MEMORIA RELACIONAL (cuerpo de campo)")
    print("="*88)
    print("  Exposicion diferencial -> competencia EMERGENTE (no parametrizada).")
    print("  Acoplamiento por bandas -> la confianza por contexto gobierna quien defiere.")
    print("  Ablacion: ON (memoria relacional) vs OFF (peso fijo 0.3).")
    print(f"  Config reducida (DUR={DUR_EXP}s, {RONDAS_CPL} rondas). -> IMAC: subir DUR_* y conteos.")
    print("="*88)

    t0 = time.time()
    filas_on, traza_on = ejecutar(True,  "MEMORIA RELACIONAL ON  (confianza adaptativa por banda)")
    filas_off, _       = ejecutar(False, "MEMORIA RELACIONAL OFF (peso fijo 0.3, arnes original)")

    print(f"\n{'#'*88}")
    print("#  LECTURA")
    print('#'*88)
    roles_on  = sum(1 for f in filas_on  if f['lider'] in ('A','B'))
    coin_on   = sum(1 for f in filas_on  if f['lider'] in ('A','B') and f['coincide'])
    roles_off = sum(1 for f in filas_off if f['lider'] in ('A','B'))
    print(f"  ON : {roles_on} bandas con rol; {coin_on} coinciden con la competencia emergente.")
    print(f"  OFF: {roles_off} bandas con rol (se espera 0: peso fijo no diferencia).")
    print("  Si ON produce roles que siguen la competencia y OFF no, la memoria relacional")
    print("  es la que hace emerger la division del trabajo. A ver que dice el dato.")

    os.makedirs("V182_logs", exist_ok=True)
    with open(f"V182_logs/v182a1_roles_{TS}.json", "w") as f:
        json.dump({'on': filas_on, 'off': filas_off}, f, indent=2)
    print(f"\n  datos: V182_logs/v182a1_roles_{TS}.json   |   tiempo {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()