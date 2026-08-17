"""Campo ambiental Env(x,y,z) — registro de constricciones históricas (protocolo §54)."""
from __future__ import annotations

import numpy as np


class Environment:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.history = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        self.stability = np.zeros_like(self.history)
        self.niches = np.zeros_like(self.history, dtype=bool)
        # Stencil isótropo para el gradiente: 26 vecinos, pesos ~1/|δ|. Por simetría
        # Σ w·δδᵀ = c·I, así que grad = (1/c)·Σ w·δ·(f(δ)−f0) es isótropo (no axial).
        offs = np.array(
            [(dx, dy, dz)
             for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if (dx, dy, dz) != (0, 0, 0)],
            dtype=np.float64,
        )
        self._grad_offs = offs
        self._grad_w = 1.0 / np.linalg.norm(offs, axis=1)
        c = float(np.sum(self._grad_w * offs[:, 0] ** 2))
        self._grad_invc = (1.0 / c) if c > 1e-12 else 0.0

    def _idx(self, pos: np.ndarray) -> tuple[int, int, int]:
        g = self.grid_size
        ix = int(np.clip(pos[0] % g, 0, g - 1))
        iy = int(np.clip(pos[1] % g, 0, g - 1))
        iz = int(np.clip(pos[2] % g, 0, g - 1))
        return ix, iy, iz

    def deposit(self, pos: np.ndarray, amount: float) -> None:
        """Depósito SUB-CELDA (trilineal): reparte entre las 8 celdas vecinas con pesos
        fraccionarios → campo suave en posición, sin cuantización a celda (que sesgaría el
        grafo de interacción vía la deriva). Cierra el residuo fino de grilla."""
        g = self.grid_size
        fx = float(pos[0]) % g
        fy = float(pos[1]) % g
        fz = float(pos[2]) % g
        x0 = int(fx); y0 = int(fy); z0 = int(fz)
        dx = fx - x0; dy = fy - y0; dz = fz - z0
        x1 = (x0 + 1) % g; y1 = (y0 + 1) % g; z1 = (z0 + 1) % g
        amt = float(amount)
        for xi, wx in ((x0 % g, 1.0 - dx), (x1, dx)):
            for yi, wy in ((y0 % g, 1.0 - dy), (y1, dy)):
                for zi, wz in ((z0 % g, 1.0 - dz), (z1, dz)):
                    w = wx * wy * wz
                    if w <= 0.0:
                        continue
                    self.history[xi, yi, zi] += amt * w
                    self.stability[xi, yi, zi] += amt * 0.25 * w

    def field_at(self, pos: np.ndarray) -> float:
        ix, iy, iz = self._idx(pos)
        return float(self.history[ix, iy, iz] + self.stability[ix, iy, iz] * 0.1)

    def gradient(self, pos: np.ndarray) -> np.ndarray:
        """Gradiente ISÓTROPO 3D de Env por mínimos cuadrados sobre vecindad esférica
        (stencil de 26 vecinos, pesos ~1/|δ|): NO privilegia los ejes de la grilla — el
        lock residual se cierra EN EL KERNEL, no con difusión. Magnitudes reales, sin
        normalizar. Env plano → gradiente ≈ 0. (C-N2.6 constricción; la única fuente
        legítima de dirección es la huella histórica acumulada.)"""
        g = self.grid_size
        ix, iy, iz = self._idx(pos)
        h = self.history
        f0 = float(h[ix, iy, iz])
        acc = np.zeros(3, dtype=np.float64)
        offs = self._grad_offs
        w = self._grad_w
        for k in range(offs.shape[0]):
            jx = min(max(ix + int(offs[k, 0]), 0), g - 1)
            jy = min(max(iy + int(offs[k, 1]), 0), g - 1)
            jz = min(max(iz + int(offs[k, 2]), 0), g - 1)
            acc += w[k] * offs[k] * (float(h[jx, jy, jz]) - f0)
        return acc * self._grad_invc

    def diffuse(self, lam: float, decay: float) -> None:
        """Difusión ambiental ISÓTROPA (§128: Env += λ∇²Env) + memoria con decaimiento
        (§129: Env·decay). Suaviza los depósitos puntuales → el gradiente por diferencias
        centradas deja de preferir los ejes de la grilla (cierra el lock residual)."""
        if lam > 0.0:
            h = self.history
            lap = (
                np.roll(h, 1, 0) + np.roll(h, -1, 0)
                + np.roll(h, 1, 1) + np.roll(h, -1, 1)
                + np.roll(h, 1, 2) + np.roll(h, -1, 2)
                - 6.0 * h
            )
            self.history = h + np.float32(lam) * lap
        if decay < 1.0:
            self.history *= np.float32(decay)
            self.stability *= np.float32(decay)

    def update_niches(self, h_crit: float) -> int:
        self.niches = self.history > h_crit
        return int(self.niches.sum())

    def total_history(self) -> float:
        return float(self.history.sum())