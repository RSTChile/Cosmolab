#!/usr/bin/env python3
"""
Genera estímulos pre-registrados Fase 1 (empalme CG002↔ANIMA).
Salida: empalme_estimulos/*.wav + empalme_estimulos_fase1.json (SHA256).

Basado en generadores de investigar_diferenciacion_audio.py (AV=0.012 viable).
No requiere motor ANIMA — solo numpy + wave.
"""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np

SR = 48000
SEG = 2.0
AV = 0.012  # intensidad viable (pre-registrada)
OUT_DIR = Path(__file__).parent
WAV_DIR = OUT_DIR / "empalme_estimulos"
MANIFEST = OUT_DIR / "empalme_estimulos_fase1.json"


def _tono(f: float, amp: float, seg: float = SEG) -> np.ndarray:
    t = np.arange(int(SR * seg)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float64)


def _ruido(amp: float, seed: int = 3, seg: float = SEG) -> np.ndarray:
    return (amp * np.random.RandomState(seed).standard_normal(int(SR * seg))).astype(np.float64)


def _rms_match(x: np.ndarray, target_rms: float) -> np.ndarray:
    r = float(np.sqrt(np.mean(x * x)) + 1e-12)
    return x * (target_rms / r)


def _write_wav_stereo(path: Path, left: np.ndarray, right: np.ndarray | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if right is None:
        right = left
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo)) + 1e-12)
    if peak > 0.99:
        stereo = stereo * (0.99 / peak)
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Pre-registro Fase 1 — clases para veredicto κ_Δ
ESTIMULOS = [
    # --- DIFIEREN (estructura temporal distinta, RMS comparable donde indicado) ---
    {
        "audio_id": "E01_tono_220_sostenido",
        "clase": "difieren",
        "descripcion": "Tono 220 Hz sostenido — estructura periódica simple",
        "generator": lambda: (_tono(220, AV), None),
    },
    {
        "audio_id": "E02_ruido_banda_amplia",
        "clase": "difieren",
        "descripcion": "Ruido blanco — estructura aleatoria; RMS matched a E01 post-hoc en manifest",
        "generator": lambda: (_ruido(AV, seed=3), None),
    },
    {
        "audio_id": "E03_pulsos_220_01s",
        "clase": "difieren",
        "descripcion": "Tono 220 Hz con envolvente pulso 0.1 s — modulación temporal",
        "generator": lambda: (_pulsos(220, AV, 0.1), None),
    },
    # --- NO DIFIEREN (mínima Δ_struct esperada) ---
    {
        "audio_id": "N01_silencio",
        "clase": "no_difieren",
        "descripcion": "Silencio digital — presión cero",
        "generator": lambda: (np.zeros(int(SR * SEG)), None),
    },
    {
        "audio_id": "N02_tono_ultra_estable",
        "clase": "no_difieren",
        "descripcion": "Tono 60 Hz muy bajo, amplitud mínima viable — casi estacionario",
        "generator": lambda: (_tono(60, AV * 0.5), None),
    },
    # --- NULL SHANNON (mismo RMS, forma distinta) ---
    {
        "audio_id": "S01_rms_match_tono_vs_ruido",
        "clase": "null_shannon",
        "descripcion": "Ruido con RMS igualado al tono E01 — test RMS-only",
        "generator": lambda: _pair_rms_null(),
    },
    {
        "audio_id": "S02_rms_match_440_vs_220",
        "clase": "null_shannon",
        "descripcion": "440 Hz RMS-matched a 220 Hz — misma energía, distinta frecuencia",
        "generator": lambda: _pair_freq_null(),
    },
    # --- Ruptura Fase 4 (pre-registro) ---
    {
        "audio_id": "R01_saturacion_colapso",
        "clase": "ruptura_kappa_O",
        "descripcion": "Tono alta amplitud (0.08) — reflejo / e_R alto esperado",
        "generator": lambda: (_tono(220, 0.08), None),
    },
    {
        "audio_id": "R02_homogeneizacion_dc",
        "clase": "ruptura_kappa_delta",
        "descripcion": "Offset DC constante (no waveform) — mínima diferenciación temporal",
        "generator": lambda: (np.full(int(SR * SEG), AV * 0.3), None),
    },
]


def _pulsos(f: float, amp: float, periodo_s: float, seg: float = SEG) -> np.ndarray:
    t = np.arange(int(SR * seg)) / SR
    env = ((t % periodo_s) < periodo_s / 2).astype(float)
    return amp * np.sin(2 * np.pi * f * t) * env


def _pair_rms_null():
    ref = _tono(220, AV)
    rms = float(np.sqrt(np.mean(ref * ref)))
    noise = _rms_match(_ruido(1.0, seed=7), rms)
    return noise, None


def _pair_freq_null():
    ref = _tono(220, AV)
    rms = float(np.sqrt(np.mean(ref * ref)))
    high = _rms_match(_tono(440, 1.0), rms)
    return high, None


def main():
    entries = []
    for spec in ESTIMULOS:
        left, right = spec["generator"]()
        wav_path = WAV_DIR / f"{spec['audio_id']}.wav"
        sha = _write_wav_stereo(wav_path, left, right)
        rms = float(np.sqrt(np.mean(left * left)))
        entries.append({
            "audio_id": spec["audio_id"],
            "clase": spec["clase"],
            "descripcion": spec["descripcion"],
            "path": str(wav_path.relative_to(OUT_DIR)),
            "sha256": sha,
            "sr": SR,
            "seg": SEG,
            "amp_nominal": AV if "ruptura" not in spec["clase"] else "variable",
            "rms_L": round(rms, 6),
        })
    manifest = {
        "protocolo": "PROTOCOLO_EMPALME_CG002_ANIMA.md",
        "fase": 1,
        "fecha_pre_registro": "2026-06-30",
        "aprobado": "Alexis (dale)",
        "sr": SR,
        "seg": SEG,
        "amp_viable_base": AV,
        "estimulos": entries,
        "criterio_pass": "clase 'difieren' → delta_struct > clase 'no_difieren'; null_shannon no colapsa diferencia",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} WAV → {WAV_DIR}")
    print(f"Manifest → {MANIFEST}")


if __name__ == "__main__":
    main()