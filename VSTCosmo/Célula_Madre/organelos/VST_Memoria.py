#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Memoria — OrganeloMemoria: la historia interna del organismo (un órgano, 6 capas)
================================================================================
"Sin memoria no hay futuro: no se recuerda el pasado, sólo hay devenir" (Alexis, 24-jun-2026).

RE-TRANSCRIBE capacidades PROBADAS y perdidas del linaje (no inventa): cada capa hereda de
un experimento validado. Encapsulado: percibir→metabolizar→secretar sobre el milieu/fila.

CAPAS (procedencia):
  1. BUFFER CORTO      — deque de estados recientes (RegistroRepresentaciones v180).
  2. PERSISTENCIA      — mantiene el último estímulo en ausencia; confianza exp(−t/τ),
                         τ CRECE con la vida vivida (MemoriaAusencia v180/V155).
  3. VALOR/AFECTIVA    — valencia[estado] con DOBLE escala: corto (τ≈8s) y largo (τ≈240s)
                         = hábito vs IDENTIDAD (ValenciaLocal v180 + v035 + valor_estructural v051).
  4. EPISÓDICA         — eventos {t,clave,tipo,intensidad,valencia}; recall por similitud CON
                         COSTO; + el recall explícito que faltaba (MemoriaEpisódica v180, 0/50).
  5. ESTRUCTURAL       — LEE (no reconstruye) la memoria implícita del soma (W hebbiana +
                         Phi_int_historia, v074/v80h) → familiaridad/novedad consultables.
  6. RELACIONAL        — confianza hacia el OTRO (Hebb social, v182); plena sólo en díada A+B.

CONEXIÓN Cb→act_perm: necesidad = act_perm·(1+Cb_norm), con Cb = presion_desacople (integrador
con fuga YA vivo). Convierte la DISPOSICIÓN instantánea (act_perm) en NECESIDAD con historia,
y añade SACIEDAD/REFRACTARIEDAD tras el re-acople (lo que a act_perm le faltaba).

ANTI-SHANNON: nada de setpoints ni "recordar para que pase X". Saliencia, consolidación y olvido
EMERGEN de sorpresa·valencia·energía. BIOENERGÉTICA: buffers acotados; no se graba sin energía;
escalas τ fisiológicas DECLARADAS. NO cierra el lazo conductual (no toca soma ni orientación).
================================================================================
"""
from __future__ import annotations
from collections import deque
import math


def _c01(v, d=0.0):
    try:
        x = float(v)
    except Exception:
        return d
    return 0.0 if x != x else max(0.0, min(1.0, x))


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if x == x else d
    except Exception:
        return d


COLS_MEM = ["mem_familiaridad", "mem_novedad", "mem_carga_estructural", "mem_valencia_estado",
            "mem_persistencia", "mem_recall", "mem_recall_tipo", "mem_recall_costo",
            "mem_episodios_n", "Cb_integrado", "necesidad", "necesidad_efectiva",
            "mem_saciedad", "mem_relacional_confianza"]

# Códigos de tipo de episodio (numéricos para CSV/gráficos)
TIPO = {"ninguno": 0, "amenaza": 1, "logro": 2, "novedad": 3, "neutro": 4}


class OrganeloMemoria:
    """La historia interna del organismo. Driven a nivel fila por actualizar(); internamente
    respeta percibir→metabolizar→secretar. Constantes FISIOLÓGICAS declaradas (no para forzar)."""

    def __init__(self,
                 buffer_n: int = 200,            # ventana de corto plazo (v180 deque)
                 tau_val_corto: float = 8.0,     # decay del valor de corto plazo (hábito)
                 tau_val_largo: float = 240.0,   # decay del valor de largo plazo (identidad)
                 eta_corto: float = 0.10,         # aprendizaje rápido del valor
                 eta_largo: float = 0.02,         # consolidación lenta corto→largo
                 umbral_saliencia: float = 0.40, # por encima → se graba un episodio
                 max_episodios: int = 64,        # capacidad episódica (olvido selectivo si se llena)
                 tau_persist_base: float = 3.0,  # permanencia base (s); crece con la vida
                 k_persist_vida: float = 0.5,    # cuánto alarga τ la historia vivida
                 escala_dA: float = 0.05,        # escala de |ΔA| que cuenta como sorpresa
                 escala_fam: float = 0.5,        # escala de divergencia Φ↔historia (familiaridad)
                 tau_saciedad: float = 6.0,      # cuánto dura la refractariedad tras saciarse
                 olvido_relacional: float = 0.99) -> None:
        self.buffer_n = buffer_n
        self.tau_val_corto = tau_val_corto; self.tau_val_largo = tau_val_largo
        self.eta_corto = eta_corto; self.eta_largo = eta_largo
        self.umbral_saliencia = umbral_saliencia; self.max_episodios = max_episodios
        self.tau_persist_base = tau_persist_base; self.k_persist_vida = k_persist_vida
        self.escala_dA = escala_dA; self.escala_fam = escala_fam
        self.tau_saciedad = tau_saciedad; self.olvido_relacional = olvido_relacional
        self.reset()

    def reset(self) -> None:
        self.buffer = deque(maxlen=self.buffer_n)   # capa 1
        self.valencia = {}                           # capa 3: clave -> {"corto","largo"}
        self.episodios = []                          # capa 4: lista de dicts
        self._A_prev = None                          # para ΔA (sorpresa)
        self._clave_presente = None; self._t_ausencia = 0.0   # capa 2
        self.confianza_persist = 0.0
        self.saciedad = 0.0                          # refractariedad de la necesidad
        self.necesidad_prev = 0.0
        self.confianza_otro = 0.0                    # capa 6
        self.t = 0.0; self.n_visitas = {}

    # ----- PERSISTENCIA: la historia interna sobrevive al apagón (incremento 1) -----
    def snapshot(self) -> dict:
        """Estado VIVO a guardar (la historia, no las constantes del genoma): valencias aprendidas,
        episodios, vínculo con el otro (confianza_otro), vida vivida (t). Todo JSON-serializable."""
        return {"valencia": self.valencia, "episodios": self.episodios,
                "confianza_otro": self.confianza_otro, "confianza_persist": self.confianza_persist,
                "saciedad": self.saciedad, "necesidad_prev": self.necesidad_prev,
                "t": self.t, "n_visitas": self.n_visitas, "buffer": list(self.buffer),
                "_clave_presente": self._clave_presente, "_t_ausencia": self._t_ausencia,
                "_A_prev": self._A_prev}

    def restore(self, d: dict) -> None:
        """Reconstituye la historia interna desde un snapshot (al despertar tras un reinicio)."""
        if not d:
            return
        self.valencia = d.get("valencia", {}) or {}
        self.episodios = d.get("episodios", []) or []
        self.confianza_otro = float(d.get("confianza_otro", 0.0) or 0.0)
        self.confianza_persist = float(d.get("confianza_persist", 0.0) or 0.0)
        self.saciedad = float(d.get("saciedad", 0.0) or 0.0)
        self.necesidad_prev = float(d.get("necesidad_prev", 0.0) or 0.0)
        self.t = float(d.get("t", 0.0) or 0.0)
        self.n_visitas = d.get("n_visitas", {}) or {}
        self.buffer = deque([tuple(x) for x in (d.get("buffer", []) or [])], maxlen=self.buffer_n)
        self._clave_presente = d.get("_clave_presente")
        self._t_ausencia = float(d.get("_t_ausencia", 0.0) or 0.0)
        self._A_prev = d.get("_A_prev")

    # --------------------------------------------------------------- utilidades
    @staticmethod
    def _clave(lat: float, A: float):
        """Firma discreta de la situación: dónde está la fuente × qué tan acoplado estoy."""
        lb = -1 if lat < -0.15 else (1 if lat > 0.15 else 0)   # izq / centro / der
        ab = int(min(4, max(0, round(A * 4))))                  # 0..4 niveles de acople
        return (lb, ab)

    def _familiaridad_soma(self, soma):
        """CAPA 5 (sólo lectura): convierte la memoria IMPLÍCITA del campo (W + Phi_int_historia)
        en señales consultables. familiaridad = ¿el campo actual se parece a su propia historia?"""
        if soma is None:
            return None
        import numpy as np
        divs, cargas = [], []
        for h in ("L", "R", "BL", "BR"):
            hemi = getattr(soma, h, None)
            if hemi is None or not hasattr(hemi, "Phi_int_historia"):
                continue
            hist = hemi.Phi_int_historia
            if float(np.sum(np.abs(hist))) < 1e-9:        # historia aún vacía
                divs.append(0.0)
            else:
                divs.append(float(np.sqrt(np.mean((hemi.Phi - hist) ** 2))))   # RMS por nodo
            if hasattr(hemi, "W"):
                cargas.append(float(np.mean(np.abs(hemi.W))))
        if not divs:
            return None
        div = sum(divs) / len(divs)
        familiaridad = math.exp(-div / max(1e-6, self.escala_fam))
        carga = (sum(cargas) / len(cargas)) if cargas else 0.0
        return familiaridad, carga

    # --------------------------------------------------------------- ciclo
    def actualizar(self, fila: dict, dt: float = 0.1, milieu=None, soma=None) -> dict:
        self.t += dt
        # ---- PERCIBIR (de la fila + milieu + soma) ----
        A = _c01(fila.get("A_sys_env"))
        lat = _num(fila.get("lateralidad"))
        e_R = _num(fila.get("e_R"))
        icr_r = _c01(fila.get("ICR_ratio")); irde_r = _c01(fila.get("IRDE_ratio"))
        act_perm = _c01(fila.get("act_perm"))
        H_real = _c01(fila.get("H_homeostasis_real"))
        S_shared = max(_c01(fila.get("altruismo_S_shared")), _c01(fila.get("ME")))
        disp = _c01(fila.get("disposicion_cooperar"))
        Cb = _num(milieu.leer("presion_desacople", 0.0)) if milieu is not None else _num(fila.get("presion_desacople"))
        fat = _num(milieu.leer("fatiga_activa", 0.0)) if milieu is not None else _num(fila.get("act_fatiga"))
        vida = _num(milieu.leer("historia", 0.0)) if milieu is not None else self.t
        fat_norm = fat / (1.0 + fat)
        if self._A_prev is None:
            self._A_prev = A
        dA = A - self._A_prev; self._A_prev = A

        # ---- CAPA 5: memoria estructural implícita del soma → familiaridad/novedad ----
        fam_carga = self._familiaridad_soma(soma)
        if fam_carga is not None:
            familiaridad, carga_estructural = fam_carga
        else:
            familiaridad, carga_estructural = 0.0, 0.0
        novedad = 1.0 - familiaridad if fam_carga is not None else _c01(abs(dA) / self.escala_dA)

        # ---- CAPA 1: buffer corto ----
        self.buffer.append((round(self.t, 2), round(A, 3), round(lat, 3), round(H_real, 3), round(act_perm, 3)))

        clave = self._clave(lat, A)
        self.n_visitas[clave] = self.n_visitas.get(clave, 0) + 1

        # ---- CAPA 3: valencia (doble escala). bondad = viabilidad canónica (H_real) ----
        bondad = H_real
        d_corto = math.exp(-dt / self.tau_val_corto); d_largo = math.exp(-dt / self.tau_val_largo)
        for k, v in self.valencia.items():                 # olvido de lo no visitado (dos τ)
            v["corto"] *= d_corto; v["largo"] *= d_largo
        vc = self.valencia.setdefault(clave, {"corto": 0.0, "largo": 0.0})
        vc["corto"] += self.eta_corto * (bondad - vc["corto"])     # aprende rápido
        vc["largo"] += self.eta_largo * (vc["corto"] - vc["largo"])  # consolida lento
        valencia_estado = max(-1.0, min(1.0, 0.5 * vc["corto"] + 0.5 * vc["largo"]))

        # ---- CAPA 4: episódica. Saliencia EMERGENTE (sorpresa·intensidad·energía) ----
        sorpresa = max(_c01(novedad), _c01(abs(dA) / self.escala_dA), _c01(irde_r))
        intensidad = _c01(0.5 * (e_R / (1.0 + e_R)) + 0.5 * act_perm)
        energia = 1.0 - fat_norm
        saliencia = sorpresa * intensidad * energia
        if dA < -self.escala_dA and irde_r > 0.5:        tipo = "amenaza"
        elif dA > self.escala_dA and H_real > 0.5:        tipo = "logro"
        elif novedad > 0.6:                               tipo = "novedad"
        else:                                             tipo = "neutro"
        if saliencia > self.umbral_saliencia:
            self.episodios.append({"t": self.t, "clave": clave, "tipo": tipo,
                                   "intensidad": round(saliencia, 3), "valencia": round(valencia_estado, 3)})
            if len(self.episodios) > self.max_episodios:   # OLVIDO SELECTIVO: poda el menos saliente/más viejo
                self.episodios.sort(key=lambda e: e["intensidad"] * math.exp(-(self.t - e["t"]) / 600.0))
                self.episodios.pop(0)

        # ---- CAPA 4 (lectura): RECALL explícito por similitud de clave, con COSTO ----
        coincidencias = [e for e in self.episodios if e["clave"] == clave]
        if coincidencias:
            ep = max(coincidencias, key=lambda e: e["intensidad"])
            mem_recall = 1.0; mem_recall_tipo = TIPO[ep["tipo"]]
            mem_recall_costo = round(0.05 + 0.03 * len(coincidencias), 3)   # recordar cuesta (latencia)
        else:
            mem_recall = 0.0; mem_recall_tipo = TIPO["ninguno"]; mem_recall_costo = 0.0

        # ---- CAPA 2: persistencia (permanencia del objeto). τ crece con la vida vivida ----
        hay_estimulo = (e_R > 0.5) or (abs(lat) > 0.15)
        tau_persist = self.tau_persist_base + self.k_persist_vida * math.log1p(max(0.0, vida))
        if hay_estimulo:
            self._clave_presente = clave; self._t_ausencia = 0.0; self.confianza_persist = 1.0
        else:
            self._t_ausencia += dt
            self.confianza_persist = math.exp(-self._t_ausencia / max(1e-6, tau_persist))

        # ---- CAPA 6: relacional (Hebb social). Plena sólo en díada (S_shared>0) ----
        reciproco = S_shared * disp
        self.confianza_otro = self.olvido_relacional * self.confianza_otro + (1.0 - self.olvido_relacional) * reciproco
        self.confianza_otro = max(0.0, min(1.0, self.confianza_otro))

        # ---- NECESIDAD: Cb→act_perm (disposición × presión acumulada) + SACIEDAD/REFRACTARIEDAD ----
        Cb_norm = Cb / (1.0 + Cb)
        necesidad = max(0.0, min(1.0, act_perm * (1.0 + Cb_norm)))
        # SATISFACCIÓN: la necesidad se sacia por re-acople (A sube con H alta) O por COMER bien
        # (met_nutricion, del OrganeloMetabolismo) — así el lazo necesidad→comer→saciedad CIERRA.
        comer = _c01(fila.get("met_nutricion"))
        satisfaccion = max(_c01(dA / self.escala_dA) * H_real, comer) * self.necesidad_prev
        d_sac = math.exp(-dt / self.tau_saciedad)
        self.saciedad = max(0.0, min(1.0, d_sac * self.saciedad + (1.0 - d_sac) * satisfaccion))
        necesidad_efectiva = necesidad * (1.0 - self.saciedad)
        self.necesidad_prev = necesidad

        # ---- SECRETAR (al milieu, si está; y devolver columnas para la fila) ----
        out = {
            "mem_familiaridad": round(familiaridad, 4), "mem_novedad": round(_c01(novedad), 4),
            "mem_carga_estructural": round(carga_estructural, 5), "mem_valencia_estado": round(valencia_estado, 4),
            "mem_persistencia": round(self.confianza_persist, 4), "mem_recall": mem_recall,
            "mem_recall_tipo": mem_recall_tipo, "mem_recall_costo": mem_recall_costo,
            "mem_episodios_n": float(len(self.episodios)), "Cb_integrado": round(Cb, 4),
            "necesidad": round(necesidad, 4), "necesidad_efectiva": round(necesidad_efectiva, 4),
            "mem_saciedad": round(self.saciedad, 4), "mem_relacional_confianza": round(self.confianza_otro, 4),
        }
        if milieu is not None:
            milieu.secretar("necesidad", out["necesidad"])
            milieu.secretar("mem_novedad", out["mem_novedad"])
            milieu.secretar("mem_valencia_estado", out["mem_valencia_estado"])
        return out


if __name__ == "__main__":
    mem = OrganeloMemoria()
    f = {"A_sys_env": 0.6, "lateralidad": -0.3, "e_R": 4.0, "ICR_ratio": 0.6, "IRDE_ratio": 0.4,
         "act_perm": 0.5, "H_homeostasis_real": 0.4, "presion_desacople": 2.0, "act_fatiga": 1.0}
    for _ in range(50):
        r = mem.actualizar(f, dt=0.1)
    print("smoke:", r)
    print("episodios:", len(mem.episodios), "valencia claves:", list(mem.valencia.keys()))
