#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoAtencionSocial — ATENCIÓN SOCIAL SELECTIVA (Fase 1).

El organismo decide AUTÓNOMAMENTE, sin intervención del usuario:
  · a quién ESCUCHA  (objetivo de atención / input)
  · a quién HABLA    (destinatario dirigido / output, modelo A: dirección intencional)

en una población de vecinos del Observatorio, mediante SESGOS DE ATENCIÓN social
(claves de éxito, similitud, dominancia — prestigio/conformidad/parentesco llegan en
Fase 2/3 vía agregados del Observatorio). Filtra el "ruido de muchas voces" enfocándose
en pocos compañeros.

Capa biológica: atención social selectiva + sesgos de transmisión cultural.
Capa tecnológica intercambiable: pondera señales del roster y elige objetivo (ε-greedy
+ histéresis + libertad funcional de NO atender). NO hace HTTP ni cambia fuentes: es un
organelo observacional/decisor; el host (WebLive) aplica su decisión (conmuta la fuente
de audio y etiqueta la voz saliente con `dirigido_a`).

Diseño canónico del proyecto: un organelo secreta columnas (`observar` → dict) y expone
`COLS_*`. Determinista salvo la exploración (usa random propio, sembrable).
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

# Columnas que este organelo secreta al CSV (para reconstruir la red social dirigida).
COLS_AS = [
    # lado ESCUCHA
    "as_esc_objetivo", "as_esc_nombre", "as_esc_score", "as_esc_sesgo",
    "as_esc_explorando", "as_esc_atender", "as_esc_dwell_s",
    # lado HABLA
    "as_habla_objetivo", "as_habla_nombre", "as_habla_score", "as_habla_sesgo",
    "as_habla_dirige", "as_habla_dwell_s",
    # sesgos del objetivo de escucha (transparencia del porqué)
    "as_b_exito", "as_b_similitud", "as_b_dominancia",
    # meta
    "as_n_candidatos", "as_modo",
]

# Cuáles de esas columnas son NUMÉRICAS. Existe porque el volcado arrancaba con None en todas
# (`out = {c: None for c in COLS_AS}`) y el CSV escribe la CADENA "None": medido sobre las 31.846
# filas del 7-ago, `as_esc_score` y sus ocho hermanas numéricas traían "None" en 14.815 filas
# (46,52 %), que es exactamente el 0 % de organismo enganchado — pero cualquier float() aguas
# abajo revienta ahí, y perfil_columnas las contaba como texto. Una columna numérica ausente vale
# 0,0 (no atendió a nadie: su puntaje de atención es cero), no la palabra "None".
_COLS_AS_NUM = frozenset((
    "as_esc_score", "as_esc_explorando", "as_esc_atender", "as_esc_dwell_s",
    "as_habla_score", "as_habla_dirige", "as_habla_dwell_s",
    "as_b_exito", "as_b_similitud", "as_b_dominancia", "as_n_candidatos",
))

_SESGOS_P1 = ("exito", "similitud", "dominancia")


def _f(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        return v if v == v else d  # descarta NaN
    except (TypeError, ValueError):
        return d


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _norm(x: Any, escala: float) -> float:
    return _clip01(_f(x) / max(1e-9, escala))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _pesos_desde_env(name: str, default: dict[str, float]) -> dict[str, float]:
    """ANIMA_AS_W_ESCUCHA='exito:1,similitud:1,dominancia:2' → dict normalizado luego."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return dict(default)
    out = dict(default)
    for par in raw.split(","):
        if ":" in par:
            k, v = par.split(":", 1)
            k = k.strip().lower()
            if k in default:
                out[k] = _f(v, default[k])
    return out


class OrganoAtencionSocial:
    def __init__(
        self,
        organism_id: str,
        self_attrs: dict[str, Any] | None = None,
        *,
        seed: int | None = None,
    ):
        self.organism_id = organism_id
        self.self_attrs = dict(self_attrs or {})
        # Modo: auto (decide solo) · manual (lo decide el usuario) · off (desactivado)
        self.modo = os.environ.get("ANIMA_ATENCION_SOCIAL", "auto").strip().lower()
        # Pesos por sesgo, SEPARADOS por lado (escuchar suele favorecer dominancia/éxito;
        # hablar suele favorecer similitud/parentesco — perillas por experimento).
        # Escuchar favorece prestigio/dominancia/éxito (atender al respetado/fuerte/exitoso).
        self.w_escucha = _pesos_desde_env("ANIMA_AS_W_ESCUCHA", {
            "exito": 1.0, "similitud": 0.5, "dominancia": 1.0,
            "prestigio": 1.5, "conformidad": 0.5, "parentesco": 0.5})
        # Hablar favorece similitud/parentesco/conformidad (dirigirse al semejante/pariente/mayoría).
        self.w_habla = _pesos_desde_env("ANIMA_AS_W_HABLA", {
            "exito": 0.5, "similitud": 1.5, "dominancia": 0.3,
            "prestigio": 0.5, "conformidad": 1.0, "parentesco": 1.5})
        # Exploración (novedad), compromiso (histéresis) y libertad funcional.
        self.epsilon = _env_float("ANIMA_AS_EPSILON", 0.15)
        self.dwell_min_s = _env_float("ANIMA_AS_DWELL_MIN_S", 8.0)
        self.margen_cambio = _env_float("ANIMA_AS_MARGEN_CAMBIO", 0.12)
        # Escalas de normalización (calibrables).
        self.esc_energia = _env_float("ANIMA_AS_ESC_ENERGIA", 1.0)
        self.esc_edad_s = _env_float("ANIMA_AS_ESC_EDAD_S", 600.0)
        # Estado interno (histéresis por lado).
        self._objetivo = {"escucha": None, "habla": None}
        self._dwell_desde = {"escucha": 0.0, "habla": 0.0}
        self._rng = random.Random(seed)
        # Última decisión (para que el host la aplique).
        self.objetivo_escucha: str | None = None
        self.objetivo_habla: str | None = None

    # ---- claves propias (para similitud), tolerante a fila o self_attrs ---------
    def _mi(self, fila: dict[str, Any], *claves: str) -> Any:
        for c in claves:
            v = self.self_attrs.get(c)
            if v not in (None, "", "-"):
                return v
        for c in claves:
            v = (fila or {}).get(c)
            if v not in (None, "", "-"):
                return v
        return None

    # ---- SESGOS (cada uno en [0,1]) --------------------------------------------
    def _exito(self, v: dict[str, Any]) -> float:
        """Sesgo de éxito: seguir a quien le va bien (alta energía, baja necesidad, alta integración)."""
        e = _norm(v.get("energia"), self.esc_energia)
        n = _clip01(_f(v.get("necesidad")))
        oi = _clip01(_f(v.get("OI")))
        return _clip01(0.5 * e + 0.3 * (1.0 - n) + 0.2 * oi)

    def _similitud(self, v: dict[str, Any], fila: dict[str, Any]) -> float:
        """Sesgo de similitud: atender a pares del mismo tipo/sexo y edad parecida."""
        vg = v.get("genero") or v.get("cara_genero")
        vt = v.get("tono") or v.get("cara_tono")
        mg = self._mi(fila, "genero", "cara_genero")
        mt = self._mi(fila, "tono", "cara_tono")
        s_gen = 1.0 if (vg and mg and str(vg) == str(mg)) else 0.0
        s_ton = 1.0 if (vt and mt and str(vt) == str(mt)) else 0.0
        edad_v = _f(v.get("t"))
        edad_yo = _f(self._mi(fila, "t"))
        cerca = 1.0 - min(1.0, abs(edad_v - edad_yo) / max(1e-9, self.esc_edad_s)) if (edad_v and edad_yo) else 0.0
        return _clip01(0.5 * s_gen + 0.3 * s_ton + 0.2 * cerca)

    def _dominancia(self, v: dict[str, Any]) -> float:
        """Sesgo de dominancia: observar a los fuertes/activos (alto arousal, energía, actividad vocal)."""
        ar = _clip01(_f(v.get("arousal") if v.get("arousal") is not None else v.get("voz_arousal")))
        en = _norm(v.get("energia"), self.esc_energia)
        vocal = 1.0 if str(v.get("voz_emitida") or "").strip() not in ("", "-") else 0.0
        return _clip01(0.5 * ar + 0.3 * en + 0.2 * vocal)

    def _sesgos(self, v: dict[str, Any], fila: dict[str, Any],
               agregados: dict[str, Any] | None) -> dict[str, float]:
        s = {
            "exito": self._exito(v),
            "similitud": self._similitud(v, fila),
            "dominancia": self._dominancia(v),
        }
        # Fase 2: prestigio/conformidad desde el grafo social del Observatorio (agregados).
        oid = v.get("organism_id")
        extra = (agregados or {}).get(oid) if oid else None
        if isinstance(extra, dict):
            for k in ("prestigio", "conformidad"):
                if extra.get(k) is not None:
                    s[k] = _clip01(_f(extra.get(k)))
        # Fase 3: parentesco pairwise (firma de genoma). Inerte hasta que haya herencia real.
        par = self._parentesco(v)
        if par is not None:
            s["parentesco"] = par
        return s

    def _parentesco(self, v: dict[str, Any]) -> float | None:
        """Similitud de firma de genoma (vector) entre yo y el par. None = sin sustrato de herencia."""
        mia = self.self_attrs.get("genoma_firma")
        suya = v.get("genoma_firma")
        if not mia or not suya:
            return None
        try:
            a = [float(x) for x in mia]
            b = [float(x) for x in suya]
            n = min(len(a), len(b))
            if n == 0:
                return None
            d = sum(abs(a[i] - b[i]) for i in range(n)) / n
            return _clip01(1.0 - d)
        except Exception:
            return None

    def _score(self, sesgos: dict[str, float], pesos: dict[str, float]) -> tuple[float, str]:
        num = 0.0
        den = 0.0
        mejor_b, mejor_c = "-", -1.0
        for b, val in sesgos.items():
            w = pesos.get(b, 1.0 if b not in _SESGOS_P1 else 0.0)
            num += w * val
            den += abs(w)
            contrib = w * val
            if contrib > mejor_c:
                mejor_c, mejor_b = contrib, b
        return (num / den if den > 0 else 0.0), mejor_b

    # ---- LIBERTAD FUNCIONAL de NO atender/dirigirse -----------------------------
    def _disposicion(self, fila: dict[str, Any]) -> float:
        """Disposición a enganchar con otros: baja con alta necesidad (auto-foco); sube con hambre social."""
        nec = _clip01(_f((fila or {}).get("necesidad")))
        hs = _clip01(_f((fila or {}).get("hambre_social")))
        return _clip01((1.0 - nec) * (0.5 + 0.5 * hs) + 0.15)

    # ---- DECISIÓN por lado (ε-greedy + histéresis) ------------------------------
    def _decidir(self, lado: str, scores: dict[str, float], engancha: bool,
                 ahora: float) -> tuple[str | None, bool]:
        actual = self._objetivo[lado]
        if not engancha or not scores:
            if actual is not None:
                self._objetivo[lado] = None
                self._dwell_desde[lado] = ahora
            return None, False
        # Compromiso: si sigo dentro del dwell y mi objetivo sigue vivo, no salto salvo margen.
        if actual in scores and (ahora - self._dwell_desde[lado]) < self.dwell_min_s:
            mejor = max(scores, key=scores.get)
            if scores.get(mejor, 0.0) <= scores.get(actual, 0.0) + self.margen_cambio:
                return actual, False
        # Exploración (novedad) vs explotación (mejor puntaje).
        explorando = self._rng.random() < self.epsilon
        if explorando:
            elegido = self._rng.choice(list(scores.keys()))
        else:
            elegido = max(scores, key=scores.get)
        if elegido != actual:
            self._objetivo[lado] = elegido
            self._dwell_desde[lado] = ahora
        return elegido, explorando

    # ---- CICLO ------------------------------------------------------------------
    def observar(self, fila: dict[str, Any], vecinos: list[dict[str, Any]] | None,
                 dt: float = 0.1, agregados: dict[str, Any] | None = None) -> dict[str, Any]:
        ahora = time.time()
        # 0,0 en las numéricas y "" en las de texto (ver _COLS_AS_NUM): el CSV no debe llevar
        # nunca la cadena "None" en una columna que alguien va a leer con float().
        out: dict[str, Any] = {c: (0.0 if c in _COLS_AS_NUM else "") for c in COLS_AS}
        out["as_modo"] = self.modo
        vecinos = [v for v in (vecinos or []) if v.get("organism_id") and v.get("organism_id") != self.organism_id]
        out["as_n_candidatos"] = len(vecinos)

        if self.modo == "off" or not vecinos:
            self.objetivo_escucha = None
            self.objetivo_habla = None
            out["as_esc_atender"] = 0.0
            out["as_habla_dirige"] = 0.0
            return out

        # Puntajes por vecino para cada lado.
        sesgos_por_oid: dict[str, dict[str, float]] = {}
        score_esc: dict[str, float] = {}
        score_hab: dict[str, float] = {}
        sesgo_esc: dict[str, str] = {}
        sesgo_hab: dict[str, str] = {}
        nombres: dict[str, str] = {}
        for v in vecinos:
            oid = v["organism_id"]
            nombres[oid] = v.get("name") or oid
            s = self._sesgos(v, fila, agregados)
            sesgos_por_oid[oid] = s
            score_esc[oid], sesgo_esc[oid] = self._score(s, self.w_escucha)
            score_hab[oid], sesgo_hab[oid] = self._score(s, self.w_habla)

        engancha = self._rng.random() < self._disposicion(fila)

        obj_e, expl_e = self._decidir("escucha", score_esc, engancha, ahora)
        obj_h, _expl_h = self._decidir("habla", score_hab, engancha, ahora)
        self.objetivo_escucha = obj_e
        self.objetivo_habla = obj_h

        # Volcado de columnas.
        out["as_esc_atender"] = 1.0 if obj_e else 0.0
        out["as_habla_dirige"] = 1.0 if obj_h else 0.0
        if obj_e:
            out["as_esc_objetivo"] = obj_e
            out["as_esc_nombre"] = nombres.get(obj_e, obj_e)
            out["as_esc_score"] = round(score_esc.get(obj_e, 0.0), 4)
            out["as_esc_sesgo"] = sesgo_esc.get(obj_e, "-")
            out["as_esc_explorando"] = 1.0 if expl_e else 0.0
            out["as_esc_dwell_s"] = round(max(0.0, ahora - self._dwell_desde["escucha"]), 1)
            s = sesgos_por_oid.get(obj_e, {})
            out["as_b_exito"] = round(s.get("exito", 0.0), 4)
            out["as_b_similitud"] = round(s.get("similitud", 0.0), 4)
            out["as_b_dominancia"] = round(s.get("dominancia", 0.0), 4)
        if obj_h:
            out["as_habla_objetivo"] = obj_h
            out["as_habla_nombre"] = nombres.get(obj_h, obj_h)
            out["as_habla_score"] = round(score_hab.get(obj_h, 0.0), 4)
            out["as_habla_sesgo"] = sesgo_hab.get(obj_h, "-")
            out["as_habla_dwell_s"] = round(max(0.0, ahora - self._dwell_desde["habla"]), 1)
        return out


if __name__ == "__main__":
    # Prueba standalone: población pequeña, sin red.
    yo = {"genero": "masculino", "tono": "celeste", "t": 500.0}
    org = OrganoAtencionSocial("ANIMA_YO", yo, seed=1)
    poblacion = [
        {"organism_id": "A", "name": "Alfa", "energia": 0.9, "necesidad": 0.1, "OI": 0.8,
         "arousal": 0.2, "genero": "femenino", "tono": "blanco", "t": 480, "voz_emitida": "hola"},
        {"organism_id": "B", "name": "Beta", "energia": 0.3, "necesidad": 0.2, "OI": 0.4,
         "arousal": 0.9, "genero": "masculino", "tono": "celeste", "t": 520, "voz_emitida": "grr"},
        {"organism_id": "C", "name": "Gamma", "energia": 0.5, "necesidad": 0.5, "OI": 0.5,
         "arousal": 0.1, "genero": "masculino", "tono": "negro", "t": 900, "voz_emitida": "-"},
    ]
    fila = {"necesidad": 0.2, "hambre_social": 0.7, "genero": "masculino", "tono": "celeste", "t": 500.0}
    print("COLS:", len(COLS_AS))
    for i in range(6):
        r = org.observar(fila, poblacion, dt=0.5)
        print(f"paso {i}: escucha={r['as_esc_objetivo']}({r['as_esc_sesgo']},{r['as_esc_score']}) "
              f"habla={r['as_habla_objetivo']} explor={r['as_esc_explorando']} dwell={r['as_esc_dwell_s']} "
              f"cand={r['as_n_candidatos']}")
        time.sleep(0.05)
