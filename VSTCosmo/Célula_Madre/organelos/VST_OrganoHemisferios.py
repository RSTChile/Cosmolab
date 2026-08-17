#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoHemisferios — CAPA HEMISFÉRICA FUNCIONAL (emula v121, integrada como organelo).

POR QUÉ EXISTE
--------------
La célula ya tiene dos "hemisferios" Φ (L, R) PERO funcionan como OÍDOS espaciales: ambos con
τ=30 sin filtro, fed por el audio izq/der, y se colapsan a un único ω_A=(L+R)/2. Por eso R₂
(predicción rápida) y la lateralidad (lenta) salen del mismo substrato simétrico y se ANULAN.
v121 demostró que coexisten si se procesan en PARALELO en dos hemisferios asimétricos integrados
por un cuerpo calloso selectivo. Este organelo añade esa asimetría FUNCIONAL, sin tocar los oídos
espaciales ni el motor.

PRINCIPIO (corrección de Alexis): la audición humana es CONTRALATERAL — cada oído viaja al
hemisferio OPUESTO, y los hemisferios se REÚNEN por el cuerpo calloso.

ARQUITECTURA
------------
    oído DERECHO   ─cruza→  H_RÁPIDO  (τ corto, "highpass" temporal: transitorios)  → R₂ / temporal
    oído IZQUIERDO ─cruza→  H_LENTO   (τ largo,  "lowpass" temporal: sostenido)      → lateralidad / espectral
                               ↕  CUERPO CALLOSO selectivo (solo si |ω_ráp−ω_len| > umbral)
                         integración funcional (reunión)

- Entrada: energia_R / energia_L (RMS por oído que el soma YA expone en la fila) — el motor NO se toca.
- Diferenciación HP/LP en el TIEMPO (un-polo): rápido usa el TRANSITORIO (drive − media), lento usa la
  media SOSTENIDA. Equivale al highpass/lowpass espectral de v121 sin necesitar la onda cruda.
- τ en pasos de ciclo, proporción 1:10 (rápido:lento) — lo que da la coexistencia.
- Cuerpo calloso idéntico en forma a v121/célula: acoplamiento = ganancia·(Φ_otro − Φ_self) SOLO al divergir.

NO decide; solo OBSERVA y secreta columnas hemi_*. Convive en paralelo con omega_L/omega_R (espaciales),
sin pisarlos, hasta validar. Persistible (snapshot/restore de los dos campos Φ).
"""
import numpy as np

# columnas que aporta a la fila (se concatenan en el soma WebLive)
COLS_HEMI = ["hemi_rapido_omega", "hemi_lento_omega", "hemi_R2", "hemi_lateralidad_func",
             "hemi_divergencia", "hemi_calloso_activo", "hemi_integracion"]

DT = 0.1                       # paso del ciclo (consistente con el genoma)
N_NODOS = 32                   # tamaño del campo Φ por hemisferio (como la célula)

# --- constantes del calloso (de v121; umbral más bajo que el de los oídos 0.5, para que SÍ se reúnan) ---
UMBRAL_CALLOSO = 0.2           # v121: divergencia mínima para activar el calloso
GANANCIA_CALLOSO = 0.05        # v121: acople débil (selectivo)
# --- constantes de tiempo en PASOS de ciclo (proporción 1:10) ---
TAU_RAPIDO = 3.0               # rápido → transitorios, R₂
TAU_LENTO = 30.0              # lento  → sostenido, lateralidad
PHIVEL_CLIP = 5.0


class _HemisferioFuncional:
    """Campo Φ (reacción-difusión) con su τ y su filtro temporal (HP rápido / LP lento)."""
    def __init__(self, modo, tau_pasos, seed=0):
        rng = np.random.RandomState(seed)
        self.modo = modo                       # 'rapido' | 'lento'
        self.tau = float(tau_pasos)
        self.Phi = rng.normal(0.0, 0.1, N_NODOS)
        self.Phi_vel = np.zeros(N_NODOS)
        self.env = 0.0                         # estado del filtro un-polo (media móvil de la energía)
        self.hist_omega = []                   # registro para R₂ (error a τ-lag)

    def omega(self):
        return float(np.mean(self.Phi))

    def _drive(self, energia):
        """Filtro temporal un-polo sobre la energía del oído contralateral.
        LENTO (lowpass): usa la media sostenida.  RÁPIDO (highpass): usa el transitorio (novedad)."""
        alpha = 1.0 / max(1.0, self.tau)       # más lento → alpha menor → más suavizado
        self.env += alpha * (energia - self.env)
        if self.modo == 'lento':
            return self.env                    # sostenido
        return energia - self.env              # transitorio (cambio respecto a la media)

    def actualizar(self, energia, otro):
        drive = self._drive(float(energia or 0.0))
        # laplaciano (difusión)
        lap = np.zeros_like(self.Phi)
        lap[1:-1] = self.Phi[:-2] - 2 * self.Phi[1:-1] + self.Phi[2:]
        # reacción natural Φ(1−Φ²)
        reaccion = self.Phi * (1.0 - self.Phi * self.Phi)
        # forzamiento dipolar (media cero) por el drive del oído contralateral
        forz = np.zeros_like(self.Phi)
        forz[0], forz[-1] = drive, -drive
        # CUERPO CALLOSO selectivo (reunión solo al divergir)
        acopl = np.zeros_like(self.Phi)
        calloso = False
        if otro is not None:
            diverg = abs(self.omega() - otro.omega())
            if diverg > UMBRAL_CALLOSO:
                acopl = GANANCIA_CALLOSO * (otro.Phi - self.Phi)
                calloso = True
        # evolución (escala temporal por τ: el lento integra más despacio)
        k = 1.0 / max(1.0, self.tau)
        self.Phi_vel += (lap + reaccion + forz + acopl) * (DT * k)
        self.Phi_vel = np.clip(self.Phi_vel, -PHIVEL_CLIP, PHIVEL_CLIP)
        self.Phi = np.clip(self.Phi + self.Phi_vel * DT, -1.0, 1.0)
        om = self.omega()
        self.hist_omega.append(om)
        if len(self.hist_omega) > 1000:
            self.hist_omega = self.hist_omega[-1000:]
        return om, calloso

    def error_R2(self):
        """R₂ = error de predicción a τ-lag (¿el rápido predijo su propio estado tras τ?).
        Bajo error = R₂ (autopredicción estable). Devuelve el error (0=predijo perfecto)."""
        n = int(self.tau)
        if len(self.hist_omega) <= n + 1:
            return 0.0
        return abs(self.hist_omega[-1] - self.hist_omega[-1 - n])

    def snapshot(self):
        # hist_omega también se guarda: sin él, R₂ (autopredicción a τ-lag) leería 0 los primeros
        # τ pasos tras un reinicio. Se recorta a 2·τ (lo único que error_R2 necesita) para no inflar el blob.
        return {"modo": self.modo, "tau": self.tau, "Phi": self.Phi.tolist(),
                "Phi_vel": self.Phi_vel.tolist(), "env": self.env,
                "hist_omega": self.hist_omega[-(int(self.tau) * 2 + 2):]}

    def restore(self, d):
        try:
            self.Phi = np.array(d["Phi"], dtype=float)
            self.Phi_vel = np.array(d["Phi_vel"], dtype=float)
            self.env = float(d.get("env", 0.0))
            self.hist_omega = list(d.get("hist_omega", []))
        except Exception:
            pass


class OrganoHemisferios:
    """Organelo de fila: lee energia_L/energia_R, cruza contralateral, integra vía calloso, secreta hemi_*."""
    def __init__(self, organismo_id=None, sr=48000):
        self.organismo_id = organismo_id
        self.rapido = _HemisferioFuncional('rapido', TAU_RAPIDO, seed=7)
        self.lento = _HemisferioFuncional('lento', TAU_LENTO, seed=11)

    def observar(self, fila, dt=DT):
        eL = float(fila.get("energia_L", 0.0) or 0.0)
        eR = float(fila.get("energia_R", 0.0) or 0.0)
        # CRUCE CONTRALATERAL: oído DERECHO → hemisferio RÁPIDO ; oído IZQUIERDO → hemisferio LENTO
        oF, cF = self.rapido.actualizar(eR, self.lento)
        oS, cS = self.lento.actualizar(eL, self.rapido)
        diverg = abs(oF - oS)
        calloso = cF or cS
        R2 = self.rapido.error_R2()
        # lateralidad FUNCIONAL = separación rápido(temporal) vs lento(espectral), NO los oídos
        lateralidad_func = oF - oS
        # integración: reunión por calloso (si se reúnen, promedio; si no, dominan en paralelo → guardamos media)
        integracion = (oF + oS) / 2.0
        out = {
            "hemi_rapido_omega": round(oF, 5),
            "hemi_lento_omega": round(oS, 5),
            "hemi_R2": round(R2, 5),
            "hemi_lateralidad_func": round(lateralidad_func, 5),
            "hemi_divergencia": round(diverg, 5),
            "hemi_calloso_activo": 1.0 if calloso else 0.0,
            "hemi_integracion": round(integracion, 5),
        }
        fila.update(out)
        return out

    def snapshot(self):
        return {"rapido": self.rapido.snapshot(), "lento": self.lento.snapshot()}

    def restore(self, d):
        if not isinstance(d, dict):
            return
        if "rapido" in d:
            self.rapido.restore(d["rapido"])
        if "lento" in d:
            self.lento.restore(d["lento"])
