"""Codificación visual CG001 — protocolo §152.

Color (matiz)  ← H  historia acumulada
Tamaño         ← S  persistencia / viabilidad
Brillo         ← S  viabilidad
Estelas        ← trayectoria por partícula (id estable)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return r, g, b


def display_position(pos: Sequence[float], grid_size: float) -> np.ndarray:
    """Centra el volumen experimental en [0, grid_size]³ sin wrap toroidal."""
    g2 = grid_size * 0.5
    return np.array(
        [float(pos[0]) + g2, float(pos[1]) + g2, float(pos[2]) + g2],
        dtype=np.float32,
    )


def entity_rgba_size(
    entity: dict,
    smax: float,
    hmax: float,
) -> tuple[tuple[float, float, float, float], float]:
    """Devuelve (r,g,b,a) y tamaño de punto para una entidad."""
    h_val = float(entity.get("H") or 0)
    s_val = float(entity.get("S") or 0)
    h_norm = h_val / max(hmax, 1e-9)
    s_norm = max(0.0, s_val) / max(smax, 1e-9)

    # Matiz recorre el espectro a medida que crece la historia
    hue = (0.50 + h_norm * 0.50) % 1.0
    sat = 0.55 + 0.35 * min(1.0, h_norm)
    lit = 0.20 + 0.70 * min(1.0, s_norm)
    r, g, b = hsv_to_rgb(hue, sat, lit)
    # Tamaño en PIXELES (el scatter usa pxMode=True): mar de puntitos diminutos,
    # solo las raras de alta persistencia se agrandan (protocolo §10 punto luminoso / §152 tamaño<-S).
    size = 1.8 + 6.0 * min(1.0, s_norm)
    return (r, g, b, 0.95), size


def encode_batch(
    entities: list[dict],
    grid_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Codifica lote de entidades → posiciones, colores RGBA, tamaños."""
    if not entities:
        z = np.zeros((0, 3), dtype=np.float32)
        return z, z, np.zeros(0, dtype=np.float32)

    smax = max(max(float(e.get("S") or 0) for e in entities), 1e-9)
    hmax = max(max(float(e.get("H") or 0) for e in entities), 1e-9)

    n = len(entities)
    pos = np.zeros((n, 3), dtype=np.float32)
    colors = np.zeros((n, 4), dtype=np.float32)
    sizes = np.zeros(n, dtype=np.float32)

    for i, e in enumerate(entities):
        pos[i] = display_position(e.get("pos", [0, 0, 0]), grid_size)
        rgba, size = entity_rgba_size(e, smax, hmax)
        colors[i] = rgba
        sizes[i] = size

    return pos, colors, sizes