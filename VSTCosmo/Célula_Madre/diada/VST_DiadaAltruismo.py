#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_DiadaAltruismo — APLICA el locus de altruismo a la COMUNICACIÓN entre dos organismos
================================================================================
QUIÉN SOY
---------
El puente que hace OPERAR el locus de altruismo del genoma ([[VST_Genoma]] · O-N22)
en la díada A↔B. El locus ya vive en el genoma (la célula madre lo CONTIENE en
potencia — pluripotencia); aquí se EXPRESA para el escalón inter-organismo, que es
el primer peldaño de la escala (sin él no operan pluricelular ni sociedad).

Lo importan A (VST_CelulaMadre_WebLive_A.py) y B (..._B.py) — idéntico en ambos.

LA DEDUCCIÓN DE BOORMAN, HECHA CÓDIGO (comunicar = acto altruista: costo τ=e_R,
beneficio σ=ΔA del otro):
  1. UMBRAL/BIESTABILIDAD (Cap.2): la comunicación genuina o se enciende (ambos sobre
     β_crit) o colapsa al silencio. No hay medio tibio estable.
  2. MUTUALIDAD: un altruista unilateral es explotado → τ resetea si el otro no recíproca.
  3. SEÑAL COSTOSA (hándicap): la voz cuesta y se invierte EN PROPORCIÓN a la disposición
     → `voice_rms = base · disposicion_cooperar` (con un piso de exploración para permitir
     la ignición — la díada "prueba" cooperar; eso es juego/LF).
  4. HISTORIA: τ acumula mutualismo sostenido; β_crit baja con LF↑ y e_R↓ (la relación
     se construye y cooperar se vuelve más fácil cuanto más ha funcionado).
  5. DÍADA COMO UNIDAD (selección de grupo): criterio de que A⊕B es unidad de orden
     superior = subaditividad e_R(A⊕B)<e_R(A)+e_R(B) (≈ costo_desacople>0).
  6. GATE DE SUJETO (Ψ_alma O-N3.4b): A coopera sólo si lee a B como sujeto (B.Cb>0).
     Sin esto no hay cooperación voluntaria (anti-Shannon).

La GOBERNANZA (decidir) vive en el organelo del genoma; este módulo lo CONDUCE con el
estado propio (fila) + el del par (vía /comunicacion/estado) y devuelve la disposición
y la ganancia de voz. Constantes/mapeos CALIBRABLES con datos de la díada.
================================================================================
"""
from __future__ import annotations
import json
import urllib.request
import urllib.parse
from typing import Optional

from VST_Genoma import organelo_altruismo, beta_crit, Milieu   # noqa: F401  (beta_crit reexport útil)


# ------------------------------------------------------------------------------
# Lectura del estado del PAR (su fila fisiológica) vía HTTP, robusta
# ------------------------------------------------------------------------------
def leer_estado_par(url_estado: str, timeout: float = 1.5) -> Optional[dict]:
    """GET al /comunicacion/estado del par y devuelve su dict (incluye 'fila'). None si falla.
    El consumidor decide qué hacer si es None (típicamente: mantener el último conocido)."""
    try:
        req = urllib.request.Request(url_estado, headers={"User-Agent": "VST-DiadaAltruismo/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def url_estado_desde_voz(url_voz: str) -> str:
    """Deriva la URL de /comunicacion/estado a partir de la URL de la voz (/comunicacion/bloque.wav)."""
    p = urllib.parse.urlparse(url_voz)
    base = f"{p.scheme}://{p.netloc}"
    return f"{base}/comunicacion/estado"


def _norm_cb(cb: float, escala: float = 5.0) -> float:
    """C_b es un conteo (estímulos procesados); se normaliza a [0,1] para Ψ_alma/S_shared."""
    try:
        return max(0.0, min(1.0, float(cb) / max(1e-9, escala)))
    except Exception:
        return 0.0


def _s_shared(fila_propia: dict, fila_par: dict) -> float:
    """SENTIDO COMPARTIDO (r de Hamilton): requiere que AMBOS tengan capacidad representacional
    (C_b>0 en los dos) y que sus estados estén alineados (|OI propio − OI par| bajo).
    S_shared = min(Cb_propio, Cb_par) · (1 − |ΔOI|). Calibrable."""
    cb_p = _norm_cb(fila_propia.get("C_b", fila_propia.get("Cb", 0.0)))
    cb_o = _norm_cb(fila_par.get("C_b", fila_par.get("Cb", 0.0)))
    oi_p = float(fila_propia.get("OI", 0.0)); oi_o = float(fila_par.get("OI", 0.0))
    alineacion = max(0.0, 1.0 - abs(oi_p - oi_o))
    return max(0.0, min(1.0, min(cb_p, cb_o) * alineacion))


# ------------------------------------------------------------------------------
# GOBERNANZA DE LA DÍADA — conduce el organelo de altruismo del genoma (lado de UN organismo)
# ------------------------------------------------------------------------------
class GobernanzaAltruismo:
    """Estado de cooperación de la díada del lado de ESTE organismo. Cada paso:
       1) mapea la fila propia + la del par a las entradas del organelo de altruismo,
       2) corre el organelo (β_crit, Hamilton, Ψ_alma, τ, coopera — lógica del genoma),
       3) devuelve disposición, coopera, β_crit, Ψ_alma… y la GANANCIA DE VOZ (señal costosa).
    No toca el ciclo metabólico del organismo: es una capa de gobernanza dedicada y determinista."""

    def __init__(self, base_voice_rms: float = 0.40, piso_exploracion: float = 0.08,
                 plast: Optional[dict] = None) -> None:
        self.org = organelo_altruismo(plast=plast)   # el locus DESARROLLADO, del genoma
        self.mil = Milieu()
        self.base_voice_rms = float(base_voice_rms)   # RMS de voz cuando coopera al máximo (señal costosa)
        self.piso_exploracion = float(piso_exploracion)  # voz mínima para permitir la IGNICIÓN (probar)
        self._A_solo: Optional[float] = None          # baseline de A (medido antes de cooperar) → costo_desacople
        self.ultimo: dict = {}

    def paso(self, fila_propia: dict, estado_par: Optional[dict],
             dt: float = 0.1, tempo: float = 1.0) -> dict:
        fp = fila_propia or {}
        par = (estado_par or {}).get("fila", {}) if estado_par else {}
        m = self.mil

        # --- baseline de A "en solitario" (se fija mientras aún no se coopera) ---
        A = float(fp.get("A_sys_env", 0.0))
        if self._A_solo is None or self.org.disposicion < 0.05:
            self._A_solo = A if self._A_solo is None else min(self._A_solo, A)

        # --- MAPEO fila → entradas canónicas del organelo (la deducción de Boorman) ---
        m.secretar("delta_struct", float(fp.get("R2", 0.0)))              # readiness social (meta-rep R₂)
        m.secretar("LF", float(fp.get("LF_op", fp.get("LF_struct", 0.0))))  # libertad funcional
        m.secretar("e_R", abs(float(fp.get("e_R", 0.0))))                # error operativo (= costo τ)
        m.secretar("A_sys_env", A)                                       # acoplamiento (beneficio σ)
        m.secretar("A_sys_env_solo", float(self._A_solo))               # contrafáctico para costo_desacople
        m.secretar("ME", _s_shared(fp, par))                            # S_shared (r de Hamilton)
        # estado del OTRO (lo que su altruismo y mi Ψ_alma necesitan)
        m.secretar("otro.Cb", float(par.get("C_b", par.get("Cb", 0.0))))
        m.secretar("otro.valencia", float(par.get("disposicion_cooperar", 0.0)))  # ¿el otro quiere? (reciprocidad)
        m.secretar("costo_cooperar", abs(float(fp.get("e_R", 0.0))))    # comunicar cuesta (hándicap)
        m.secretar("estado_reproductivo", 1.0)                          # disponibilidad (1=libre; calibrable)

        # --- correr el organelo (gobernanza del genoma) ---
        self.org.percibir(m); self.org.metabolizar(dt, tempo); self.org.secretar(m)
        disp = float(self.org.disposicion)

        # --- SEÑAL COSTOSA: la voz se invierte en proporción a la disposición (+ piso de ignición) ---
        voice_rms = self.base_voice_rms * max(disp, self.piso_exploracion)

        self.ultimo = {
            "disposicion_cooperar": round(disp, 4),
            "coopera": bool(self.org.coopera),
            "beta_crit": round(self.org.beta_crit, 4),
            "supera_umbral": bool(self.org.supera_umbral),
            "hamilton_ok": bool(self.org.hamilton_ok),
            "psi_alma": round(self.org.psi_alma, 4),
            "tau_simbiosis": round(self.org.tau, 2),
            "costo_desacople": round(self.org.costo_desacople, 4),
            "S_shared": round(m.leer("ME"), 4),
            "voice_rms": round(voice_rms, 4),
            "atractor": ("comunicando" if self.org.coopera
                         else "emergiendo" if disp > 0.15 else "mudo"),
        }
        return self.ultimo


# ==============================================================================
# SMOKE / TESTS — simula la díada SIN HTTP (dos gobernanzas que se reciprocan)
# ==============================================================================
def _smoke() -> None:
    # Dos organismos; cada uno ve la disposición del otro como 'otro.valencia' (reciprocidad).
    gobA = GobernanzaAltruismo(plast=dict(tau_min=0.5))
    gobB = GobernanzaAltruismo(plast=dict(tau_min=0.5))

    def fila(progreso, cb=5.0, oi=0.5):
        # A_sys_env BAJO al inicio (solo) y sube al acoplarse → costo_desacople>0 (díada como unidad)
        return {"R2": 0.9, "LF_op": 0.5, "e_R": 0.05, "C_b": cb, "OI": oi,
                "A_sys_env": 0.30 + 0.50 * progreso}

    rA = {"fila": fila(0.0)}; rB = {"fila": fila(0.0)}   # estados iniciales del par
    for k in range(60):
        prog = min(1.0, k / 20.0)
        rA_next = gobA.paso(fila(prog), rB)
        rB_next = gobB.paso(fila(prog), rA)
        # cada uno expone su disposición en su 'fila' para que el otro la lea (reciprocidad)
        rA = {"fila": {**fila(prog), "disposicion_cooperar": rA_next["disposicion_cooperar"]}}
        rB = {"fila": {**fila(prog), "disposicion_cooperar": rB_next["disposicion_cooperar"]}}

    assert gobA.ultimo["disposicion_cooperar"] > 0.5, gobA.ultimo
    assert gobA.ultimo["coopera"] is True and gobB.ultimo["coopera"] is True, "díada recíproca → coopera"
    assert gobA.ultimo["voice_rms"] > gobA.base_voice_rms * 0.5, "voz costosa sube con la disposición"
    assert gobA.ultimo["atractor"] == "comunicando"

    # DESALMAMIENTO: el par NO es sujeto (C_b=0) → no emerge cooperación, voz al piso
    gobC = GobernanzaAltruismo(plast=dict(tau_min=0.5))
    parNS = {"fila": fila(1.0, cb=0.0)}
    for _ in range(60):
        r = gobC.paso(fila(1.0), parNS)
    assert gobC.ultimo["psi_alma"] == 0.0 and gobC.ultimo["disposicion_cooperar"] < 0.05, gobC.ultimo
    assert gobC.ultimo["coopera"] is False
    assert abs(gobC.ultimo["voice_rms"] - gobC.base_voice_rms * gobC.piso_exploracion) < 1e-6, "voz al piso de ignición"

    print("OK VST_DiadaAltruismo:  díada recíproca → coopera y voz costosa sube ✓ · "
          "desalmamiento bloquea y voz al piso ✓")
    print(f"   A → disposicion={gobA.ultimo['disposicion_cooperar']} coopera={gobA.ultimo['coopera']} "
          f"β_crit={gobA.ultimo['beta_crit']} voz={gobA.ultimo['voice_rms']} "
          f"S_shared={gobA.ultimo['S_shared']} costo_desacople={gobA.ultimo['costo_desacople']} "
          f"atractor={gobA.ultimo['atractor']}")
    print(f"   sin-sujeto → disposicion={gobC.ultimo['disposicion_cooperar']} "
          f"Ψ_alma={gobC.ultimo['psi_alma']} voz={gobC.ultimo['voice_rms']} atractor={gobC.ultimo['atractor']}")


if __name__ == "__main__":
    _smoke()
