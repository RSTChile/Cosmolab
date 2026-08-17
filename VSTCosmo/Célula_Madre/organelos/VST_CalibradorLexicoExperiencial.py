#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_CalibradorLexicoExperiencial — el ESTATUTO de una palabra emerge de lo que le HACE al organismo.

POR QUÉ EXISTE (corrección anti-Shannon de Alexis, 29-jun-2026)
--------------------------------------------------------------
El órgano vocal asignaba arousal/valence a cada sonido por su ESPECTRO (FFT, centroide, planitud)
en `_afecto_acustico`. Eso es Shannon encubierto: "la palabra significa por cómo suena". PROHIBIDO.
En Cosmosemiótica el sentido NO está en la señal: emerge de la TRANSFORMACIÓN que la señal produce
en un organismo SITUADO. Por eso aquí invertimos el principio:

    NO:  la palabra significa por cómo suena
    SÍ:  la palabra adquiere estatuto por lo que le hace al organismo al escucharla

CÓMO (mismo principio que EIT-3 Térmico: clasificar por el RÉGIMEN que emerge del acoplamiento)
----------------------------------------------------------------------------------------------
    sonido escuchado → respuesta fisiológica (Δvector) → régimen inducido → nombre observacional provisional

Cuatro regímenes (cuadrantes de respuesta sistémica, NO emociones humanas):
    JARDIN_FERTIL  integración/vigor/apertura      (sube ICR/OI/estabilidad/apertura, sin alto coste)
    CERRADO        silencio/espera/reposo           (casi no mueve, o pausa estable; no-apertura)
    COLAPSO        pérdida/bloqueo/agotamiento       (baja integración/energía/H/LF)
    SELVA_HOSTIL   tensión/alerta/riesgo/defensa     (mucha activación, sube IRDE/coste/riesgo)

QUÉ NO HACE (límites estrictos pedidos por Alexis)
--------------------------------------------------
NO decide conducta · NO toca la Libertad Funcional (LF) · NO gobierna OVE (expresión) ni OAO
(aprendizaje) · NO impone semántica. SÓLO observa, clasifica y nombra para el OBSERVADOR. El nombre
es PROVISIONAL y REVISABLE: una palabra puede ser 'Alerta' hoy y 'Vigor' mañana si su efecto histórico
cambia. El nombre nunca es verdad fija ni orden para el organismo.

Es medición pura sobre dos filas (basal/final): no produce los estados, sólo los lee. El script
experimentos/calibrar_repertorio_experiencial.py conduce al organismo y le pasa las filas.
"""
from __future__ import annotations
import os, json, math

# ----------------------------------------------------------------------------- variables del Δvector
# Cada variable canónica con sus posibles claves en la fila (robusto a nombres; sólo usa las presentes).
VARS_ALIAS = {
    "OI":              ["OI"],
    "LF_op":           ["LF_op", "LF", "lf_op", "libertad_funcional"],
    "H":               ["H", "H_homeostasis", "H_real"],
    "RC_total":        ["RC_total", "RC"],
    "ICR":             ["ICR"],
    "IRDE":            ["IRDE"],
    "A_sys_env":       ["A_sys_env", "A_sys-env", "A_sys"],
    "Omega":           ["omega_A", "Omega", "omega"],
    "energia":         ["met_energia", "energia", "E", "act_perm_energia"],
    "necesidad":       ["necesidad", "met_necesidad"],
    "valence":         ["mem_valencia_estado", "voz_valence", "valence"],
    "atencion":        ["act_propuesta_atencional", "act_atencion_L", "atencion", "absorcion"],
    "orientacion":     ["act_orientacion_deg", "orientacion"],
    # presentes sólo al EMITIR o en el lazo completo (ausentes en escucha pasiva): se usan si aparecen.
    "arousal":         ["voz_arousal", "arousal"],
    "expectativa":     ["expectativa", "expectativa_vale_observar"],
    "valor_ecologico": ["valor_ecologico", "voz_otro_valor", "voz_otro_relevancia"],
    "cara_valoracion": ["cara_valoracion"],
}

# Nombres OBSERVACIONALES por régimen (campos semánticos amplios; el nombre concreto es provisional).
REGIMENES = {
    "JARDIN_FERTIL": {"campo": "fertilidad/integración",
                      "nombres": ["Vigor", "Alegría", "Sí", "Confianza", "Apertura", "Compañía", "Crear", "Florecer", "Acuerdo"]},
    "CERRADO":       {"campo": "reposo/espera",
                      "nombres": ["Silencio", "Espera", "Reposo", "No sé", "Quietud", "Pausa", "Retirada", "Cierre", "Dormir"]},
    "COLAPSO":       {"campo": "pérdida/bloqueo",
                      "nombres": ["No", "Dolor", "Bloqueo", "Caída", "Pérdida", "Fatiga", "Ruptura", "Enfermedad", "Fin"]},
    "SELVA_HOSTIL":  {"campo": "tensión/alerta",
                      "nombres": ["Alerta", "Cuidado", "Amenaza", "Enojo", "Peligro", "Defensa", "Tensión", "Huida", "Choque"]},
}

# Umbrales de clasificación (sobre el Δvector; calibrables). Anti-Shannon: son escalas de RESPUESTA,
# no cortes de banda. UMBRAL_QUIETO discrimina "la palabra casi no mueve al organismo" (CERRADO).
UMBRAL_QUIETO = 0.04      # norma del Δvector bajo esto → CERRADO (no-apertura)
UMBRAL_ACTIVACION = 0.06  # activación sobre esto, con beneficio<0 → SELVA_HOSTIL (alerta); si no → COLAPSO


def _leer(fila: dict, canon: str):
    """Primer alias presente en la fila para la variable canónica; None si ninguna está."""
    for k in VARS_ALIAS.get(canon, []):
        if k in fila and fila[k] is not None:
            try:
                return float(fila[k])
            except (TypeError, ValueError):
                return None
    return None


def snapshot_fisiologico(fila: dict) -> dict:
    """Extrae el sub-vector fisiológico de una fila (sólo las variables presentes)."""
    out = {}
    for canon in VARS_ALIAS:
        v = _leer(fila, canon)
        if v is not None:
            out[canon] = v
    return out


def delta_vector(basal: dict, final: dict) -> dict:
    """Δ = final − basal, sólo para las variables presentes en AMBOS snapshots."""
    b = snapshot_fisiologico(basal); f = snapshot_fisiologico(final)
    return {f"d_{k}": round(f[k] - b[k], 6) for k in f if k in b}


def _eje_beneficio(d: dict) -> float:
    """Integración/beneficio (con signo): ¿la palabra integró y abrió, o desacopló y agotó?
    Sube con OI/ICR/H/A_sys_env/valence/energía; baja con IRDE/necesidad."""
    g = lambda k: d.get(k, 0.0)
    return (0.30 * g("d_OI") + 0.25 * g("d_ICR") + 0.15 * g("d_H") + 0.15 * g("d_A_sys_env")
            + 0.15 * g("d_valence") + 0.10 * g("d_energia")
            - 0.25 * g("d_IRDE") - 0.20 * g("d_necesidad"))


def _eje_activacion(d: dict) -> float:
    """Tensión/alarma (sin signo, ≥0): la activación SIMPÁTICA que distingue SELVA (alerta) de COLAPSO
    (agotamiento). Sube con arousal, ruido contextual y riesgo de desacople (IRDE). NO incluye el gasto
    energético: la pérdida de energía YA pesa en el eje de beneficio (la define el COLAPSO, no la alarma)."""
    g = lambda k: d.get(k, 0.0)
    return abs(g("d_arousal")) + 0.6 * max(0.0, g("d_RC_total")) + 0.8 * max(0.0, g("d_IRDE"))


def _norma(d: dict) -> float:
    return math.sqrt(sum(v * v for v in d.values())) if d else 0.0


class EscalaPropia:
    """Cuánto mueve al organismo una palabra CUALQUIERA, según su propia experiencia.

    UMBRAL_QUIETO=0.04 y UMBRAL_ACTIVACION=0.06 eran dos números escritos a mano contra una escala
    que nadie midió. Si el organismo vive en un ambiente donde todas las palabras lo mueven poco,
    todo cae en CERRADO; si vive en uno intenso, nada. El nombre de la palabra terminaba diciendo
    más del umbral que de la palabra.

    Esto lleva la media móvil y la dispersión típica de lo que el organismo REALMENTE experimenta,
    así que "casi no me movió" pasa a significar "menos de lo que suele moverme a mí". Mientras no
    haya observaciones suficientes se usan las constantes históricas, para no clasificar al azar
    durante los primeros segundos de vida.
    """

    def __init__(self, minimo: int = 20):
        self.minimo = minimo
        self.n = 0
        self.mu_norma = 0.0
        self.sd_norma = 0.0
        self.mu_activacion = 0.0

    def observar(self, norma: float, activacion: float) -> None:
        self.n += 1
        a = min(0.05, 1.0 / max(1, self.n))
        self.mu_norma += a * (norma - self.mu_norma)
        self.sd_norma += a * (abs(norma - self.mu_norma) - self.sd_norma)
        self.mu_activacion += a * (activacion - self.mu_activacion)

    @property
    def maduro(self) -> bool:
        return self.n >= self.minimo

    def umbrales(self) -> tuple:
        """(quieto, activacion) en la escala del propio organismo, o las históricas si aún no sabe."""
        if not self.maduro:
            return UMBRAL_QUIETO, UMBRAL_ACTIVACION
        return max(0.0, self.mu_norma - self.sd_norma), self.mu_activacion

    def snapshot(self) -> dict:
        return {"n": self.n, "mu_norma": self.mu_norma, "sd_norma": self.sd_norma,
                "mu_activacion": self.mu_activacion}

    def restore(self, d: dict) -> None:
        if not d:
            return
        self.n = int(d.get("n", 0) or 0)
        self.mu_norma = float(d.get("mu_norma", 0.0) or 0.0)
        self.sd_norma = float(d.get("sd_norma", 0.0) or 0.0)
        self.mu_activacion = float(d.get("mu_activacion", 0.0) or 0.0)


def clasificar(d: dict, escala: "EscalaPropia | None" = None) -> tuple:
    """Régimen inducido + (beneficio, activacion, norma) a partir del Δvector. Reglas (cuadrantes):
       norma pequeña            → CERRADO   (casi no mueve / pausa estable)
       beneficio ≥ 0            → JARDIN_FERTIL (integró, abrió, sin desacoplar)
       beneficio < 0 y activa   → SELVA_HOSTIL (dañó CON activación → tensión/alerta)
       beneficio < 0 y quieto   → COLAPSO   (dañó con baja activación → caída/agotamiento)"""
    beneficio = _eje_beneficio(d); activacion = _eje_activacion(d); norma = _norma(d)
    u_quieto, u_activacion = escala.umbrales() if escala is not None else (UMBRAL_QUIETO, UMBRAL_ACTIVACION)
    if norma < u_quieto:
        reg = "CERRADO"
    elif beneficio >= 0.0:
        reg = "JARDIN_FERTIL"
    elif activacion >= u_activacion:
        reg = "SELVA_HOSTIL"
    else:
        reg = "COLAPSO"
    return reg, {"beneficio": round(beneficio, 5), "activacion": round(activacion, 5), "norma": round(norma, 5)}


def _nombre(regimen: str, voz_id: str) -> str:
    """Nombre observacional estable por (régimen, voz_id): determinista, sin azar, pero cambia si el
    régimen cambia. NO es significado literal; es una etiqueta para que el observador hable de la palabra."""
    nombres = REGIMENES[regimen]["nombres"]
    idx = sum(ord(c) for c in str(voz_id)) % len(nombres)
    return nombres[idx]


def _confianza(norma: float, n_obs: int, coincidencias: int) -> str:
    """Confianza del estatuto: por claridad del efecto (norma) y consistencia entre repeticiones."""
    consistencia = coincidencias / max(1, n_obs)
    if n_obs >= 3 and consistencia >= 0.8 and norma >= 2 * UMBRAL_QUIETO:
        return "alta"
    if consistencia >= 0.6 and norma >= UMBRAL_QUIETO:
        return "media"
    return "baja"


class CalibradorLexicoExperiencial:
    """Observa qué transformación produce cada palabra al ser escuchada y le da un estatuto provisional.
    Mantiene la BIOGRAFÍA de cada palabra (el nombre puede cambiar con la historia). Persistible."""

    def __init__(self, organismo_id=None):
        self.organismo_id = organismo_id
        # voz_id -> {"observaciones": [ {t, regimen, delta, ejes} ... ], "estatuto": {...}}
        self.lexico: dict = {}
        # la escala de lo que a ESTE organismo lo mueve; nace de su experiencia (ver EscalaPropia)
        self.escala = EscalaPropia()

    def observar(self, voz_id: str, basal: dict, final: dict, t: float = 0.0) -> dict:
        """Registra UNA escucha de la palabra: calcula Δvector, régimen y guarda en su biografía.
        Devuelve el registro de esta observación. NO consolida estatuto (eso es consolidar())."""
        d = delta_vector(basal, final)
        # Se aprende ANTES de clasificar para que la palabra actual cuente en la escala con que se
        # la juzga: si no, la primera escucha de un ambiente nuevo se mide contra un mundo anterior.
        self.escala.observar(_norma(d), _eje_activacion(d))
        reg, ejes = clasificar(d, self.escala)
        obs = {"t": round(float(t), 3), "regimen": reg, "delta": d, "ejes": ejes}
        self.lexico.setdefault(str(voz_id), {"observaciones": [], "estatuto": {}})
        self.lexico[str(voz_id)]["observaciones"].append(obs)
        return obs

    def consolidar(self, voz_id: str, ventana: int = 8) -> dict:
        """MATIZ per-organismo: el régimen que ESTE organismo lee para la palabra, por mayoría de sus
        últimas observaciones. NO asigna el nombre público: el nombre es de la CONVENCIÓN compartida
        (establecer_convencion). Aquí sólo se registra cómo responde este organismo (su matiz), para que
        pueda diferir en INTENSIDAD sin romper el léxico común. El nombre común se inyecta con
        aplicar_convencion(). Devuelve el matiz."""
        voz_id = str(voz_id)
        if voz_id not in self.lexico or not self.lexico[voz_id]["observaciones"]:
            return {}
        obs = self.lexico[voz_id]["observaciones"][-ventana:]
        conteo = {}
        for o in obs:
            conteo[o["regimen"]] = conteo.get(o["regimen"], 0) + 1
        reg = max(conteo, key=conteo.get)
        coincid = conteo[reg]
        norma_media = sum(o["ejes"]["norma"] for o in obs if o["regimen"] == reg) / max(1, coincid)
        matiz = {
            "voz_id": voz_id,
            "regimen_propio": reg,                       # cómo responde ESTE organismo (matiz, no autoridad)
            "confianza_propia": _confianza(norma_media, len(obs), coincid),
            "n_observaciones": len(self.lexico[voz_id]["observaciones"]),
            "delta_medio": self._delta_medio(voz_id),
        }
        self.lexico[voz_id]["matiz"] = matiz
        return dict(matiz)

    def aplicar_convencion(self, convencion: dict) -> None:
        """Inyecta el estatuto COMÚN (nombre/régimen de la convención compartida) en este organismo, sin
        tocar su matiz propio. Tras esto, el organismo NOMBRA según la convención, no según su lectura."""
        for voz_id, est in (convencion or {}).items():
            self.lexico.setdefault(str(voz_id), {"observaciones": [], "matiz": {}})
            self.lexico[str(voz_id)]["estatuto_comun"] = dict(est)

    def estatuto(self, voz_id: str) -> dict:
        """Estatuto operativo = el COMÚN (convención) si está aplicado; el matiz propio es secundario."""
        v = self.lexico.get(str(voz_id), {})
        return dict(v.get("estatuto_comun", {})) or dict(v.get("matiz", {}))

    def _delta_medio(self, voz_id: str) -> dict:
        """Δvector medio sobre todas las observaciones (efecto típico de la palabra). Útil para colocar
        la palabra por lo que INDUCE, no por su espectro — pero el Calibrador sólo lo expone, no lo aplica."""
        obs = self.lexico[str(voz_id)]["observaciones"]
        acc, cnt = {}, {}
        for o in obs:
            for k, v in o["delta"].items():
                acc[k] = acc.get(k, 0.0) + v; cnt[k] = cnt.get(k, 0) + 1
        return {k: round(acc[k] / cnt[k], 5) for k in acc}

    def matiz(self, voz_id: str) -> dict:
        """El matiz propio de este organismo (régimen/Δ que ÉL lee), aunque el nombre común sea otro."""
        return dict(self.lexico.get(str(voz_id), {}).get("matiz", {}))

    def tabla(self) -> list:
        """Estatuto OPERATIVO (común si está; matiz si no) de cada palabra (para observatorio/UI/CSV)."""
        return [self.estatuto(vid) for vid in self.lexico if self.estatuto(vid)]

    def snapshot(self) -> dict:
        # la escala tambien viaja: sin ella el organismo reaprende en cada arranque que es
        # "mucho" y que es "poco", y sus primeras palabras del dia quedan mal clasificadas.
        return {"lexico": self.lexico, "escala": self.escala.snapshot()}

    def restore(self, d: dict) -> None:
        if isinstance(d, dict) and isinstance(d.get("lexico"), dict):
            self.lexico = d["lexico"]
        if isinstance(d, dict) and isinstance(d.get("escala"), dict):
            self.escala.restore(d["escala"])


# ===================================================================================================
# CONVENCIÓN LÉXICA COMPARTIDA — una sola fuente de verdad para el NOMBRE (criterio de Alexis)
# ===================================================================================================
# El nombre NO puede ser idiolecto: si A/B/C/D nombraran distinto, se perdería la capacidad de entender
# qué se dicen (mismo código, sentido divergente). La clasificación se calibra en un organismo o COHORTE
# de referencia, se consolida en una CONVENCIÓN, y se PROPAGA a todos. Los organismos guardan matices de
# respuesta propios, pero el nombre común sólo cambia por NUEVA RECALIBRACIÓN COLECTIVA.

def establecer_convencion(calibradores: dict, ventana: int = 8, convencion_previa: dict | None = None) -> dict:
    """Vota sobre la cohorte de referencia un estatuto COMÚN por palabra, con RECHAZO DE MODO COMÚN.
    El sentido es DIFERENCIAL (O-N3.1a, Δ_struct = diferencia): el régimen de una palabra es su posición
    RELATIVA al campo de sonidos, no su respuesta absoluta contra el silencio. La respuesta genérica de
    'llegó un sonido' (onset: RC/IRDE suben para CUALQUIER sonido) es modo común — no porta sentido
    distintivo — y se RESTA. Se clasifica el RESIDUAL (Δ_palabra − Δ_medio_repertorio).
    El nombre se asigna una vez (determinista); recalibración colectiva → historial_nombres.
    Devuelve {voz_id: estatuto_comun}. `calibradores` = {org: CalibradorLexicoExperiencial}."""
    previa = convencion_previa or {}
    voces = sorted({v for cal in calibradores.values() for v in cal.lexico.keys()})
    # 1) Δvector medio por (organismo, palabra)
    deltas = {}
    for org, cal in calibradores.items():
        for voz in voces:
            m = cal.consolidar(voz, ventana=ventana)
            if m and m.get("delta_medio"):
                deltas[(org, voz)] = m["delta_medio"]
    if not deltas:
        return {}
    keys = sorted({k for d in deltas.values() for k in d})
    # 2) MODO COMÚN = Δ medio de todo el repertorio (la señal compartida 'soy un sonido')
    comun = {k: sum(d.get(k, 0.0) for d in deltas.values()) / len(deltas) for k in keys}
    # 3) clasificar el RESIDUAL por (organismo, palabra) y votar por palabra
    convencion = {}
    for voz_id in voces:
        votos, residuales = {}, []
        for org in calibradores:
            d = deltas.get((org, voz_id))
            if d is None:
                continue
            res = {k: round(d.get(k, 0.0) - comun.get(k, 0.0), 6) for k in keys}
            reg, _ = clasificar(res, getattr(self, 'escala', None))
            votos[reg] = votos.get(reg, 0) + 1
            residuales.append(res)
        if not votos:
            continue
        reg = max(votos, key=votos.get)
        n_orgs = sum(votos.values()); acuerdo = votos[reg] / max(1, n_orgs)
        conf = "alta" if (n_orgs >= 2 and acuerdo >= 0.99) else ("media" if acuerdo >= 0.5 else "baja")
        nombre = _nombre(reg, voz_id)
        res_medio = {k: round(sum(r[k] for r in residuales) / len(residuales), 5) for k in keys} if residuales else {}
        est = {"voz_id": voz_id, "regimen_experiencial": reg, "campo_semantico": REGIMENES[reg]["campo"],
               "nombre_observacional": nombre, "confianza": conf, "n_organismos": n_orgs,
               "acuerdo_cohorte": round(acuerdo, 3), "delta_residual": res_medio,
               "historial_nombres": list(previa.get(voz_id, {}).get("historial_nombres", []))}
        prev = previa.get(voz_id, {})
        if prev.get("regimen_experiencial") != reg or prev.get("nombre_observacional") != nombre:
            est["historial_nombres"].append({"regimen": reg, "nombre": nombre, "confianza": conf})
        convencion[voz_id] = est
    return convencion


class ConvencionLexica:
    """El léxico COMÚN propagado a A/B/C/D. Persistible en el repertorio compartido. Sólo cambia por
    recalibración colectiva (establecer_convencion). Lectura barata para UI/observatorio."""

    def __init__(self, ruta: str | None = None):
        self.ruta = ruta
        self.lexico: dict = {}
        if ruta and os.path.isfile(ruta):
            self.cargar(ruta)

    def actualizar(self, convencion: dict) -> None:
        self.lexico.update(convencion or {})

    def nombre(self, voz_id: str) -> str:
        return self.lexico.get(str(voz_id), {}).get("nombre_observacional", "")

    def estatuto(self, voz_id: str) -> dict:
        return dict(self.lexico.get(str(voz_id), {}))

    def tabla(self) -> list:
        return [dict(v) for v in self.lexico.values()]

    def guardar(self, ruta: str | None = None) -> str:
        ruta = ruta or self.ruta
        if not ruta:
            return ""
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        tmp = ruta + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"convencion": self.lexico}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, ruta)
        return ruta

    def cargar(self, ruta: str | None = None) -> None:
        ruta = ruta or self.ruta
        try:
            with open(ruta, encoding="utf-8") as f:
                self.lexico = json.load(f).get("convencion", {})
        except Exception:
            self.lexico = {}
