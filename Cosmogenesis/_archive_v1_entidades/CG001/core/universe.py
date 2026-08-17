"""Universo CG001 — simulación mínima de persistencia diferencial."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from CG001.core.entity import Entity
from CG001.core.environment import Environment
from CG001.metrics.persistence import compute_metrics


def _load_config(path: str | None = None) -> dict:
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    default = Path(__file__).resolve().parents[1] / "config" / "CG001_default.yaml"
    with open(default, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Universe:
    def __init__(self, config: dict | None = None, config_path: str | None = None):
        self.cfg = config or _load_config(config_path)
        u = self.cfg["universe"]
        self.grid_size = int(u["grid_size"])
        self.n_entities = int(u["n_entities"])
        self.initial_radius = float(u["initial_radius"])
        self.expansion_rate = float(u["expansion_rate"])
        self.r_int = float(u["r_interaction"])
        self.persist_cost = float(u["persist_cost"])
        self.s0 = float(u["s0"])
        self.epsilon = float(os.environ.get("CG_EPSILON", u.get("epsilon", 0.0)))
        self.kappa_p = float(u["kappa_p"])
        self.kappa_h = float(u["kappa_h"])
        self.niche_h_crit = float(u["niche_h_crit"])
        # Bloque A — geometría isótropa
        self.sigma_move = float(u.get("sigma_move", 0.3))   # difusión gaussiana por eje
        self.drift_coef = float(u.get("drift_coef", 0.5))   # peso de la deriva por gradiente real de Env
        # Bloque C — S según §118 completo (gana/pierde/redistribuye → S fluctúa, se desacopla de H)
        self.gain = float(u.get("gain", 0.02))              # refuerzo (Δ y S)   [CALIBRAR B]
        self.loss = float(u.get("loss", 0.015))             # cancelación en Δ
        self.s_loss = float(u.get("s_loss", 0.01))          # cancelación en S (§118 C) — rompe acreción
        self.transfer_frac = float(u.get("transfer_frac", 0.05))  # intercambio redistribuye Δ y S
        self.compat_thr = float(u.get("compat_thr", 0.15))
        self.deposit_coef = float(u.get("deposit_coef", 0.05))
        self.niche_s_gain = float(u.get("niche_s_gain", 0.002))
        self.env_diffusion = float(u.get("env_diffusion", 0.05))  # §128 — SOLO ecología (palanca de régimen)
        self.env_decay = float(u.get("env_decay", 0.999))         # §129 memoria ambiental
        # Modo conmutable de la regla de S (hipótesis testeada, NO decidida). Ninguno es default
        # metodológico: el gate corre AMBOS. coupled = §118-completo (H1); persist_only = §8 literal (H2).
        self.s_rule = os.environ.get("CG_S_RULE", str(u.get("s_rule", "coupled")))
        # Ganancia de acoplamiento S↔Δ CONTINUA en [0,1] (banda §118 parcial): escala los
        # canales intercambio+cancelación de S. 0 = persist_only (S no sigue a Δ por esas
        # vías), 1 = coupled (§118 pleno). Si no se da CG_S_GAIN/s_gain, se deriva del modo
        # → g=0 y g=1 reproducen EXACTAMENTE los dos endpoints ya medidos (compat total).
        _g_default = 1.0 if self.s_rule == "coupled" else 0.0
        self.s_gain = float(os.environ.get("CG_S_GAIN", u.get("s_gain", _g_default)))
        self.s_coupled = (self.s_gain > 0.0)
        self.rotate_test = os.environ.get("CG_ROTATE_TEST", str(u.get("rotate_test", 0))) in ("1", "true", "True")
        self.seed = int(os.environ.get("CG_SEED", self.cfg.get("seed", 42)))
        self.experiment_id = os.environ.get("CG_EXPERIMENT_ID", "CG001-A" if self.epsilon == 0 else "CG001-B")

        self.rng = np.random.default_rng(self.seed)
        self.env = Environment(self.grid_size)
        self.entities: list[Entity] = []
        self.t_sim = 0
        self.R = 1.0
        self.events: list[dict] = []
        self._next_id = 0
        self.paused = False
        self._init_entities()

    def _random_in_ball(self, radius: float) -> np.ndarray:
        """Posición isótropa uniforme en una BOLA (el cubo uniform(-r,r) prefiere esquinas)."""
        v = self.rng.normal(size=3)
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
        r = radius * float(self.rng.random()) ** (1.0 / 3.0)
        return v * r

    def _random_rotation(self) -> np.ndarray:
        """Rotación arbitraria (test causal §A): rota la grilla al iniciar."""
        a = self.rng.normal(size=(3, 3))
        q, r = np.linalg.qr(a)
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q

    def _init_entities(self) -> None:
        self.entities.clear()
        self._next_id = 0
        rot = self._random_rotation() if self.rotate_test else None
        for i in range(self.n_entities):
            pos = self._random_in_ball(self.initial_radius)
            if rot is not None:
                pos = rot @ pos
            s = self.s0 + (self.epsilon if i == 0 else 0.0)   # ε SOLO en la S inicial de id=0 (§60/§133)
            delta = self.rng.uniform(0.01, 0.05)
            e = Entity(id=self._next_id, S=s, delta_struct=delta, H=0.0, pos=pos.astype(np.float64), lineage=i)
            self._next_id += 1
            self.entities.append(e)
        self._log_event("inicio", {"n": self.n_entities, "epsilon": self.epsilon, "seed": self.seed, "s_rule": self.s_rule, "rotate_test": self.rotate_test})

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.env = Environment(self.grid_size)
        self.t_sim = 0
        self.R = 1.0
        self.events.clear()
        self._init_entities()

    def _log_event(self, kind: str, payload: dict | None = None) -> None:
        ev = {"t": self.t_sim, "evento": kind, **(payload or {})}
        self.events.append(ev)
        if len(self.events) > 500:
            self.events = self.events[-500:]

    def _alive(self) -> list[Entity]:
        return [e for e in self.entities if e.alive]

    def step(self) -> dict[str, Any]:
        if self.paused:
            return self.snapshot()

        self.t_sim += 1
        self.R += self.expansion_rate

        alive = self._alive()
        if not alive:
            return self.snapshot()

        # Expansión ADITIVA radial (§132: R(t)=R₀+v_exp·t), no multiplicativa desde el origen
        for e in alive:
            r = float(np.linalg.norm(e.pos))
            if r > 1e-9:
                e.pos = e.pos + self.expansion_rate * (e.pos / r)

        # Movimiento ISÓTROPO: difusión gaussiana independiente por eje + deriva por el
        # gradiente REAL de Env. Env plano → deriva ≈ 0 → difusión isótropa pura → cero
        # estructura inducida por el código (la única anisotropía legítima es la huella).
        for e in alive:
            drift = self.drift_coef * self.env.gradient(e.pos)
            e.pos = e.pos + self.rng.normal(0.0, self.sigma_move, 3) + drift

        # Interacciones locales (§52) — bucketing espacial O(n·k)
        cell = max(self.r_int, 0.5)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for i, e in enumerate(alive):
            key = (
                int(e.pos[0] // cell),
                int(e.pos[1] // cell),
                int(e.pos[2] // cell),
            )
            buckets.setdefault(key, []).append(i)
        log_events = os.environ.get("CG_QUIET_EVENTS", "0") != "1"

        def _pair_interact(ei: Entity, ej: Entity) -> None:
            d = float(np.linalg.norm(ei.pos - ej.pos))
            if d >= self.r_int:
                return
            compat = abs(ei.delta_struct - ej.delta_struct) < self.compat_thr
            if compat:
                # Refuerzo (§118 G): ambos ganan en Δ y en S
                ei.delta_struct += self.gain
                ej.delta_struct += self.gain
                ei.S += self.gain
                ej.S += self.gain
                if log_events:
                    self._log_event("refuerzo", {"a": ei.id, "b": ej.id, "d": round(d, 3)})
            elif self.rng.random() < 0.5:
                # Intercambio: Δ SIEMPRE se redistribuye (§124). S sigue con ganancia
                # continua s_gain∈[0,1] (§118 parcial): 0=no redistribuye, 1=plena.
                dt = self.transfer_frac * ei.delta_struct
                ei.delta_struct -= dt
                ej.delta_struct += dt
                st = self.s_gain * self.transfer_frac * ei.S
                ei.S -= st
                ej.S += st
                if log_events:
                    self._log_event("intercambio", {"from": ei.id, "to": ej.id})
            else:
                # Cancelación: Δ SIEMPRE baja. S baja con ganancia continua s_gain (§118 C):
                # 0=persist_only (S solo cae por costo/paso, §8 literal), 1=coupled pleno.
                ei.delta_struct = max(0.0, ei.delta_struct - self.loss)
                ej.delta_struct = max(0.0, ej.delta_struct - self.loss)
                ei.S -= self.s_gain * self.s_loss
                ej.S -= self.s_gain * self.s_loss
                if log_events:
                    self._log_event("cancelacion", {"a": ei.id, "b": ej.id})
            amt = self.deposit_coef * (ei.delta_struct + ej.delta_struct)
            self.env.deposit(ei.pos, amt * 0.5)
            self.env.deposit(ej.pos, amt * 0.5)

        # Δ_struct al INICIO del paso (para historia neta observable por paso, §16.3/§22)
        d0 = {e.id: e.delta_struct for e in alive}

        max_pairs = int(os.environ.get("CG_MAX_PAIRS", "4000"))
        processed: set[tuple[int, int]] = set()
        for key, idxs in buckets.items():
            if max_pairs and len(processed) >= max_pairs:
                break
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        jdxs = buckets.get((key[0] + dx, key[1] + dy, key[2] + dz))
                        if not jdxs:
                            continue
                        for i in idxs:
                            for j in jdxs:
                                if max_pairs and len(processed) >= max_pairs:
                                    break
                                if i == j:
                                    continue
                                a, b = min(i, j), max(i, j)
                                if (a, b) in processed:
                                    continue
                                processed.add((a, b))
                                _pair_interact(alive[a], alive[b])

        # Historia por PASO: solo la variación estructural OBSERVABLE cuenta (§16.3 + §22).
        # Desacopla H y t_hist del conteo de interacciones (y de S).
        for e in alive:
            e.record_history(abs(e.delta_struct - d0[e.id]), self.kappa_h)

        # R₀ + muerte estructural (§53)
        births = 0
        for e in alive:
            field = self.env.field_at(e.pos)
            niche_bonus = self.niche_s_gain if field > self.niche_h_crit else 0.0
            e.S += niche_bonus - self.persist_cost
            if e.S <= self.kappa_p:
                e.alive = False
                self.env.deposit(e.pos, e.H + e.delta_struct)
                self._log_event("colapso", {"entity": e.id, "S": round(e.S, 5)})

        # Fusión compatible ocasional (Regla 3)
        if self.t_sim % 50 == 0 and len(self._alive()) > 10:
            pair = self.rng.choice(self._alive(), size=2, replace=False)
            if abs(pair[0].delta_struct - pair[1].delta_struct) < 0.08:
                pair[0].S += pair[1].S * 0.3
                pair[0].delta_struct += pair[1].delta_struct * 0.5
                pair[0].H += pair[1].H
                pair[1].alive = False
                pair[0].lineage = pair[0].id
                births += 1
                self._log_event("fusion", {"survivor": pair[0].id, "merged": pair[1].id})

        # Difusión + memoria ambiental (§128/§129) — isotropiza el gradiente
        self.env.diffuse(self.env_diffusion, self.env_decay)

        niches = self.env.update_niches(self.niche_h_crit)
        if niches and self.t_sim % 100 == 0:
            self._log_event("nichos", {"count": niches})

        snap = self.snapshot()
        snap["births"] = births
        return snap

    def snapshot(self) -> dict[str, Any]:
        alive = self._alive()
        fast = os.environ.get("CG_FAST_METRICS", "0") == "1"
        metrics = compute_metrics(alive, self.env, self.cfg.get("metrics", {}), fast=fast)
        top = sorted(alive, key=lambda e: e.S, reverse=True)[:8]
        return {
            "ok": True,
            "experiment_id": self.experiment_id,
            "t_sim": self.t_sim,
            "R": round(self.R, 4),
            "epsilon": self.epsilon,
            "s_rule": self.s_rule,
            "seed": self.seed,
            "paused": self.paused,
            "N": len(alive),
            "N0": self.n_entities,
            "metrics": metrics,
            "S_max_entity": top[0].id if top else None,
            "top_entities": [e.to_dict() for e in top],
            "env_H": round(self.env.total_history(), 4),
            "niches": int(self.env.niches.sum()),
            "events_recent": self.events[-12:],
            "ts": time.time(),
        }

    def sample_positions(self, limit: int = 200) -> list[dict]:
        alive = sorted(self._alive(), key=lambda e: e.id)
        if not alive:
            return []
        chosen = alive if len(alive) <= limit else alive[:limit]
        return [e.to_dict() for e in chosen]