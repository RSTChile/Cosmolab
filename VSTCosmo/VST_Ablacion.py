#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Ablacion — ANATOMÍA FUNCIONAL CUANTITATIVA (ablación) + TEST ENERGÍA/ESTRUCTURA
================================================================================

Responde a las DOS preguntas convergentes del equipo (Qwen/GPT/DeepSeek) sobre la
célula madre funcional:

  (A) ABLACIÓN SISTEMÁTICA — mismo audio, apagando UN bloque por vez → tabla de
      "anatomía funcional": qué métrica muere al quitar cada organelo. Atribución
      causal medida, no afirmada (metodología del ADDENDUM).

  (B) TEST ENERGÍA-vs-ESTRUCTURA — la pregunta fina (la lección V103): ¿la
      diferenciación entre estímulos es genuina (estructura) o trivial (solo refleja
      que un audio mete más energía/e_R)? Se corren estímulos de estructura distinta
      NORMALIZADOS a la misma energía (RMS). Si SIGUEN diferenciándose → estructura;
      si COLAPSAN → era energía.

Reutiliza el motor validado (no reescribe nada): importa la célula madre funcional.
Uso:  venv/bin/python3 VST_Ablacion.py            # corre A y B por defecto
================================================================================
"""

from __future__ import annotations
import os, sys, csv as csvmod
from datetime import datetime
import numpy as np
from VST_CelulaMadre_Web import cmf   # cmf = Célula_Madre_Funcional_001 (ya cargada)

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_binaural")


# ==============================================================================
# Runner: corre la célula sobre un audio con interruptores, devuelve métricas finales
# ==============================================================================
def run_metrics(audio: np.ndarray, toggles: dict, sim_s: float) -> dict:
    cel = cmf.celula_madre_funcional(audio)
    apagados = []
    for name, org in cel.organelos.items():
        if name != "soma" and not toggles.get(name, True):
            org.expresar = False; apagados.append(name)
    soma = cel.organelos["soma"]
    dur = len(audio) / cmf.SR
    sim = min(dur, sim_s)
    pasos = max(1, int(sim / cmf.DT))
    xe_onset = lf_onset = None; cm_peak = 0.0; omega = []
    for _ in range(pasos):
        cel.vivir_un_paso(cmf.DT)
        m = cel.milieu
        if xe_onset is None and m.leer("XE", 0.0) > 0.0: xe_onset = round(cel.t, 2)
        if lf_onset is None and m.leer("lf_nivel", 0) >= 1: lf_onset = round(cel.t, 2)
        cm_peak = max(cm_peak, m.leer("C_m", 0.0)); omega.append(m.leer("Omega", 0.0))
    s = cel.salud(); m = cel.milieu
    return {
        "OI": round(s["OI"], 4), "nivel": s["nivel_OI"], "R2": round(m.leer("R2", 0.0), 4),
        "LF_op": round(m.leer("LF_op", 0.0), 4), "XE": round(min(1.0, m.leer("XE", 0.0)), 4),
        "C_m_fin": round(m.leer("C_m", 0.0), 4), "C_m_pico": round(cm_peak, 4),
        "H": round(m.leer("H_homeostasis", 0.0), 4), "Lambda_Cos": round(s["Lambda_Cos"], 5),
        "Omega_med": round(float(np.mean(omega)), 4), "e_R": round(m.leer("e_R", 0.0), 3),
        "A_sys_env": round(m.leer("A_sys_env", 0.0), 3),
        "inv": sum(1 for v in s["invariantes"].values() if v),
        "xe_onset": xe_onset, "lf_onset": lf_onset, "campo_finito": bool(soma.finito),
        "apagados": ",".join(apagados) or "—",
    }


def cargar(spec: str, rms_objetivo: float | None = None):
    """Carga audio (ruta .wav, ruta relativa a audio_binaural/, o 'demo:...'). Si
    rms_objetivo, NORMALIZA la energía (RMS) a ese valor (para el test estructura)."""
    if not spec.startswith("demo:") and not os.path.isabs(spec):
        cand = os.path.join(AUDIO_DIR, spec)
        if os.path.exists(cand): spec = cand
    nombre, audio = cmf.cargar_audio(spec)
    if rms_objetivo is not None:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 1e-9:
            audio = np.clip(audio * (rms_objetivo / rms), -1.0, 1.0)
    return nombre, audio


# ==============================================================================
# (A) ABLACIÓN SISTEMÁTICA
# ==============================================================================
ABLACIONES = [
    ("TODO ON (baseline)",        {}),
    ("sin Consciencia (B5)",      {"consciencia_basica": False, "meta_representacion": False, "self": False}),
    ("sin R₂ (meta-repr.)",       {"meta_representacion": False}),
    ("sin LF (libertad)",         {"LF": False}),
    ("sin Exaptación (B8)",       {"exaptacion": False}),
    ("sin C_m (metacognición)",   {"consciencia_metacognitiva": False}),
    ("sin Homeostasis",           {"homeostasis::x_interna": False}),
]

def ablacion(spec: str, sim_s: float = 20.0) -> None:
    nombre, audio = cargar(spec)
    print("=" * 100)
    print(f"(A) ANATOMÍA FUNCIONAL — ablación sobre '{nombre}'  (sim {sim_s:.0f}s, {len(audio)/cmf.SR:.1f}s audio)")
    print("=" * 100)
    print(f"  {'configuración':<26}{'OI':>7}{'R₂':>7}{'LF_op':>7}{'XE':>7}{'C_m':>7}{'H':>7}{'Λ_Cos':>8}{'κ':>4}  nivel")
    base = None; filas = []
    for label, tog in ABLACIONES:
        r = run_metrics(audio, tog, sim_s); filas.append((label, r))
        if base is None: base = r
        dOI = "" if r is base else f"  (ΔOI {r['OI']-base['OI']:+.3f})"
        print(f"  {label:<26}{r['OI']:>7.3f}{r['R2']:>7.3f}{r['LF_op']:>7.3f}{r['XE']:>7.3f}"
              f"{r['C_m_fin']:>7.3f}{r['H']:>7.3f}{r['Lambda_Cos']:>8.4f}{r['inv']:>3}/6  {r['nivel']}{dOI}")
    print("\n  LECTURA (qué muere al apagar cada pieza, vs baseline):")
    b = base
    for label, r in filas[1:]:
        efectos = []
        if r["R2"] < b["R2"] - 0.05:   efectos.append(f"R₂↓({b['R2']:.2f}→{r['R2']:.2f})")
        if r["LF_op"] < b["LF_op"]-0.02: efectos.append(f"LF_op↓({b['LF_op']:.2f}→{r['LF_op']:.2f})")
        if r["XE"] < b["XE"] - 0.02:   efectos.append(f"XE↓({b['XE']:.2f}→{r['XE']:.2f})")
        if r["C_m_fin"] < b["C_m_fin"]-0.02: efectos.append(f"C_m↓({b['C_m_fin']:.2f}→{r['C_m_fin']:.2f})")
        if r["H"] < b["H"] - 0.02:     efectos.append(f"H↓({b['H']:.2f}→{r['H']:.2f})")
        print(f"    · {label:<26} OI {b['OI']:.3f}→{r['OI']:.3f}   {' '.join(efectos) or '(sin efecto neto)'}")
    _guardar("ablacion", nombre, [{"config": l, **r} for l, r in filas])


# ==============================================================================
# (B) TEST ENERGÍA-vs-ESTRUCTURA
# ==============================================================================
def comparar(specs: list[str], sim_s: float = 20.0, rms: float = 0.12) -> None:
    print("\n" + "=" * 100)
    print(f"(B) ENERGÍA-vs-ESTRUCTURA — estímulos NORMALIZADOS a la misma energía (RMS={rms})  sim {sim_s:.0f}s")
    print("    Si siguen diferenciándose con MISMA energía → la diferenciación es ESTRUCTURAL (no trivial).")
    print("=" * 100)
    print(f"  {'estímulo':<26}{'OI':>7}{'R₂':>7}{'LF_op':>7}{'XE':>7}{'C_m_pico':>9}{'Λ_Cos':>8}{'e_R':>7}{'Ω':>7}  nivel")
    filas = []
    for spec in specs:
        try:
            nombre, audio = cargar(spec, rms_objetivo=rms)
        except Exception as e:
            print(f"  ⚠ {spec}: {e}"); continue
        r = run_metrics(audio, {}, sim_s); filas.append((nombre, r))
        print(f"  {nombre:<26}{r['OI']:>7.3f}{r['R2']:>7.3f}{r['LF_op']:>7.3f}{r['XE']:>7.3f}"
              f"{r['C_m_pico']:>9.3f}{r['Lambda_Cos']:>8.4f}{r['e_R']:>7.2f}{r['Omega_med']:>7.3f}  {r['nivel']}")
    if len(filas) > 1:
        oi = [r["OI"] for _, r in filas]; cm = [r["C_m_pico"] for _, r in filas]
        eR = [r["e_R"] for _, r in filas]
        print(f"\n  rango OI={max(oi)-min(oi):.3f}  rango C_m_pico={max(cm)-min(cm):.3f}  "
              f"rango e_R={max(eR)-min(eR):.2f} (a igual energía, e_R parejo ⇒ diferencias = estructura)")
    _guardar("estructura", "estimulos", [{"estimulo": n, **r} for n, r in filas])


def _guardar(tag: str, nombre: str, filas: list[dict]) -> None:
    os.makedirs("CelulaMadre_logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = f"CelulaMadre_logs/{tag}_{nombre}_{ts}.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    print(f"  📁 {ruta}")


if __name__ == "__main__":
    SIM = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    # (A) anatomía funcional sobre Blue Monday (estímulo complejo)
    ablacion("Blue_Monday_binaural_expandido.wav", sim_s=SIM)
    # (B) estructura vs energía: voz / ruido / música, MISMA energía
    comparar(["Voz_Estudio_pos60deg.wav", "Ruido blanco_neg60deg.wav", "musica_pos60deg.wav",
              "Brandemburgo.wav"], sim_s=SIM)
