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
import os
import urllib.request
import urllib.parse
from typing import Optional

from VST_Genoma import organelo_altruismo, beta_crit, Milieu   # noqa: F401  (beta_crit reexport útil)
# Lo habitual de cada magnitud, aprendido de la propia experiencia. UNA sola implementación
# para todo el proyecto: si este patrón se reescribe en cada módulo, en tres meses hay una
# variante por módulo (es literalmente la primera recomendación de la auditoría del 4-ago-2026).
# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel, NEUTRO


# ------------------------------------------------------------------------------
# POR QUÉ COOPERAR ERA ARITMÉTICAMENTE IMPOSIBLE (medido, 5-ago-2026)
# ------------------------------------------------------------------------------
# Corpus: 100.542 pasos reales del organismo ANIMA_5Z934MWHNNRH (44 CSV de fisiología,
# 3–4 ago 2026). `altruismo_coopera` = 0 en 100.542 de 100.542 filas (nuniq = 1).
# `altruismo_atractor` = "mudo" 99.731 veces, "emergiendo" 795, "comunicando" NINGUNA.
#
# El organelo del genoma exige CINCO cosas a la vez. Medidas por separado:
#     supera_umbral (R₂ > β_crit) ........  47,92 %
#     hamilton_ok (A/e_R > 1/S) ..........   0,77 %   ← la traba
#     Ψ_alma > 0 .........................  53,52 %
#     τ > τ_min = 8,0 s ..................   0,06 %   ← la otra traba
#     costo_desacople > 0,30 .............  92,24 %
#     disposicion > 0,50 .................   0,24 %
#     LAS CINCO A LA VEZ .................   0,00 %  (0 de 100.542)
# Y no es mala suerte: `favorable ∧ τ>8` ocurre 0 veces, o sea las dos trabas son DISJUNTAS.
#
# Las tres causas, todas del mismo tipo — magnitudes de escalas distintas comparadas como si
# fueran conmensurables:
#
#  1. UNIDADES. A_sys_env ∈ [0,1638 , 0,9000] (mediana 0,4954) frente a |e_R| con mediana 8,66
#     y máximo 26,29: 17× más grande. Hamilton (r·σ > τ) sólo puede cumplirse si e_R < A·S ≤ 0,90,
#     y |e_R| baja de 0,90 únicamente en su valor-suelo. El beneficio NO PUEDE superar al costo
#     por construcción; no porque cooperar no convenga, sino porque se restan peras y manzanas.
#
#  2. UN INTERRUPTOR DISFRAZADO DE PRECIO. El organelo hace `objetivo *= max(0, 1 − 0,5·costo)`,
#     o sea espera un costo adimensional en [0,1]. Recibía |e_R|. Ese factor tomó DOS valores
#     distintos en 100.542 pasos: 0,0000 (54,89 %) y 0,7500 (45,11 %). Nada en medio.
#
#  3. UN PUNTO MUERTO MUTUO. τ sólo acumula si el OTRO tiene disposición > 0, y se reinicia a 0
#     en cuanto no. Con la disposición propia > 0 sólo el 5,72 % de los pasos, cada uno espera
#     al otro: τ = 0 el 92,66 % del tiempo, racha máxima observada 15,7 s frente a los 8,0 s que
#     hacen falta — pero nunca coincidiendo con `favorable`. Esto NO se arregla desde aquí: τ_min
#     y la regla de reinicio viven en el organelo del genoma (VST_Genoma.py). Queda anotado.
#
# Lo que este archivo corrige es (1) y (2), que son suyos: lo que le ENTREGA al organelo.
#
# ANTES / DESPUÉS — replay offline sobre el mismo corpus (díada simétrica, dos organelos que se
# reciprocan, y la DISPONIBILIDAD DEL PAR tomada del propio corpus). El replay reproduce la
# producción antes de tocar nada —hamilton_ok 0,77 % clavado, disp>piso 1,30 %, psi_alma 54 %—,
# que es lo que le da derecho a decir algo del después:
#                                     ANTES        DESPUÉS
#     e_R que entra al organelo       8,7234       0,3999
#     β_crit = 1,0 (inalcanzable)     49,29 %       1,94 %
#     supera_umbral                   73,54 %      93,35 %
#     hamilton_ok                      0,77 %       3,76 %
#     costo_desacople  (mediana)      0,6433       0,0465     ← ahora es una medida, no un artefacto
#     costo_desacople > 0,30          96,19 %       4,38 %
#     disposición (p95)               0,0002       0,1259
#     disposición > piso              1,30 %       8,65 %
#     coopera                         0,025 %      0,270 %
#     atractor "comunicando"          25           273
# Cooperar deja de ser IMPOSIBLE y pasa a ser DIFÍCIL, que es lo que debe ser una señal costosa.
# Nótese que costo_desacople BAJA: eso no es una regresión, es dejar de aprobar el criterio de
# unidad de orden superior por un artefacto. La díada todavía no es una unidad, y ahora lo dice.
#
# LO QUE SIGUE BLOQUEADO Y NO SE ARREGLA DESDE AQUÍ (queda escrito para que nadie lo busque
# en este archivo):
#   · La bimodalidad de e_R (o 0,5 o ≥5,08, nada en medio) y el decreto A = 0,9 en silencio
#     son del campo — fases 2.1 y 2.2 del plan. Mientras sigan, "el mejor momento para
#     cooperar" seguirá siendo aquel en que el organismo no oye nada.
#   · `tau_min = 8,0` y la regla de reinicio de τ viven en el organelo del genoma.
#   · `costo_desacople_min = 0,30` también: con el contrafáctico ya honesto sólo se cumple el
#     4,38 % de los pasos, así que ahora es él la traba principal y hay que mirarlo.
#   · C_b sólo toma 4 ó 5 por el `epsilon = 1e-3` del Bloque05 (hallazgo 3.1).
# ------------------------------------------------------------------------------

# ORIGEN: número de señales REPRESENTABLES que cuenta C_b en el Bloque05 —
# ["e_R", "orientacion", "delta_struct", "A_sys_env", "presion_desacople"]
# (genoma/VST_Bloque05_ConscienciaFuncional.py:73). No es una escala elegida: es el MÁXIMO
# ESTRUCTURAL del conteo, y el propio Bloque05 ya publica esa misma división como
# `C_b_norm = C_b / max(1, len(REPRESENTABLES))` (:120).
CB_REPRESENTABLES = 5


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
    """Deriva la URL de /comunicacion/estado a partir de la URL de la voz.
    Par LAN: http://ip:puerto/comunicacion/bloque.wav → http://ip:puerto/comunicacion/estado.
    Par por RELAY del Observatorio: .../voz?oid=X → .../comunicacion/estado?oid=X (proxy de estado del par)."""
    p = urllib.parse.urlparse(url_voz)
    base = f"{p.scheme}://{p.netloc}"
    oid = (urllib.parse.parse_qs(p.query).get("oid") or [""])[0].strip()
    if oid and p.path.rstrip("/").endswith("/voz"):
        return f"{base}/comunicacion/estado?oid={urllib.parse.quote(oid, safe='')}"
    return f"{base}/comunicacion/estado"


def _norm_cb(cb: float, escala: float = float(CB_REPRESENTABLES)) -> float:
    """C_b es un CONTEO de señales representables; se lleva a [0,1] para Ψ_alma/S_shared.

    LEGÍTIMA, SE DEJA. El divisor es el máximo estructural del conteo (ver CB_REPRESENTABLES),
    no una escala inventada. MEDIDO sobre 100.542 pasos: C_b ∈ {1,2,3,4,5}, nunca lo desborda.
    Sí es cierto que el resultado casi no discrimina —vale 0,8 ó 1,0 en el 98,8 % de los pasos,
    porque C_b es 4 ó 5 en el 98,8 %— pero eso NO lo causa este divisor: lo causa el
    `epsilon = 1e-3` del Bloque05 (hallazgo 3.1 de la auditoría, gravedad 9, archivo de otro).
    Cambiar aquí el 5 taparía ese defecto y se rompería el día que se corrija aguas arriba.
    """
    try:
        return max(0.0, min(1.0, float(cb) / max(1e-9, escala)))
    except Exception:
        return 0.0


def _cb01(fila: dict) -> float:
    """C_b en [0,1]. Prefiere el `C_b_norm` que el Bloque05 YA calcula (una sola definición en
    el organismo, no dos); sólo si no viene en la fila divide el conteo por su máximo."""
    v = fila.get("C_b_norm")
    if v is not None:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            pass
    return _norm_cb(fila.get("C_b", fila.get("Cb", 0.0)))


def _s_shared(fila_propia: dict, fila_par: dict, esc_dOI: Optional[Escala] = None) -> float:
    """SENTIDO COMPARTIDO (r de Hamilton): requiere que AMBOS tengan capacidad representacional
    (C_b>0 en los dos) y que sus estados estén alineados (|OI propio − OI par| bajo).
    S_shared = min(Cb_propio, Cb_par) · alineación.

    QUÉ ESTABA MAL en `alineacion = 1 − |ΔOI|`: el 1 da por supuesto que ΔOI puede llegar a 1.
    MEDIDO sobre 100.542 pasos: OI vive entre p01 = 0,2736 y p99 = 0,6619, de modo que |ΔOI|
    no pasa de 0,3883 y la alineación NO PUEDE bajar de 0,6117 pase lo que pase. Es decir: el
    módulo venía afirmando que dos organismos cualesquiera están siempre al menos un 61 %
    alineados — una escala que nadie midió, y un término que resta un poco pero no discrimina.

    AUTORREGULADO: ΔOI se compara con la diferencia HABITUAL entre estos dos organismos
    (rel → 0,5 cuando es la de siempre, → 1 cuando divergen como nunca, → 0 cuando coinciden
    como nunca). Recorre [0,1] entero y no tiene ningún parámetro que elegir.
    Es PERCEPCIÓN (cuánto me parezco a este otro ahora mismo), no condición de viabilidad,
    así que relativizar es lo correcto según la advertencia 2 de escala.py.
    """
    cb_p = _cb01(fila_propia)
    cb_o = _cb01(fila_par)
    if cb_o <= 0.0:
        # El par no se pudo leer (o no es sujeto): no hay sentido compartido, y —esto importa—
        # NO se alimenta la escala de ΔOI. Con `par` vacío, `par.get("OI", 0.0)` vale 0,0 y ΔOI
        # saldría ≈ 0,41 (el OI propio) en el 44,70 % de los pasos, que es el porcentaje medido
        # de pasos con el par ilegible: la escala aprendería como "diferencia habitual con el
        # otro" una cifra en la que el otro ni siquiera participa.
        return 0.0
    oi_p = float(fila_propia.get("OI", 0.0)); oi_o = float(fila_par.get("OI", 0.0))
    d_oi = abs(oi_p - oi_o)
    if esc_dOI is None:
        alineacion = 1.0 - NEUTRO                      # sin historia: la escala se abstiene
    else:
        esc_dOI.observar(d_oi)
        alineacion = (1.0 - rel(d_oi, esc_dOI)) if esc_dOI.madura else (1.0 - NEUTRO)
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

    def __init__(self, base_voice_rms: Optional[float] = None, piso_exploracion: float = 0.08,
                 plast: Optional[dict] = None) -> None:
        self.org = organelo_altruismo(plast=plast)   # el locus DESARROLLADO, del genoma
        self.mil = Milieu()
        # QUÉ ESTABA MAL: el defecto era 0,40, una SEGUNDA escala absoluta para una magnitud que
        # el organismo ya tiene definida en otro sitio. El órgano de la voz
        # (organelos/VST_OrganoComunicacion.py:203) lee LA MISMA variable de entorno
        # VST_VOICE_TARGET_RMS pero con defecto 0,15 — y WebLive escribe el `voice_rms` que sale
        # de aquí DIRECTAMENTE sobre ORGANO_COMUNICACION.voice_target_rms
        # (web/VST_CelulaMadre_WebLive_A.py:5472). Son dos números para lo mismo, 2,67× de
        # diferencia, y cuál manda dependía de quién escribiera último.
        # AUTORREGULADO POR COMPARACIÓN (rel_contra en espíritu, advertencia 1 de escala.py):
        # existe otra magnitud del organismo con LAS MISMAS UNIDADES —el nivel de voz del propio
        # órgano de la voz—, así que la díada MODULA ese nivel en vez de inventar uno. Misma
        # variable de entorno y mismo defecto que el órgano ⇒ una sola definición.
        # OJO (no es de este archivo): web/VST_CelulaMadre_WebLive_{A,B,C,D}.py:719 sigue pasando
        # base_voice_rms=env(VST_VOICE_TARGET_RMS, "0.40") explícito. Mientras eso no se unifique,
        # el 0,40 sobrevive en producción aunque aquí ya no exista.
        if base_voice_rms is None:
            base_voice_rms = float(os.environ.get("VST_VOICE_TARGET_RMS", "0.15"))
        self.base_voice_rms = float(base_voice_rms)   # RMS de voz cuando coopera al máximo (señal costosa)
        # PISO DE EXPLORACIÓN — SE DEJA A PROPÓSITO, con su medida y su deuda escritas.
        # MEDIDO antes: `voice_rms` fue exactamente el piso en 99.219 de 100.542 pasos (98,70 %).
        # O sea que el piso no era un piso: ERA la voz. Pero la causa medida de eso no es el 0,08
        # sino la disposición aplastada por los dos defectos de escala de la cabecera.
        # MEDIDO después, con esos dos corregidos y sobre el mismo corpus (replay offline):
        # la disposición supera el piso en el 8,65 % de los pasos en vez del 1,30 %, así que el
        # piso pasa de mandar el 98,70 % del tiempo a mandar el 91,35 %. Sigue siendo mucho, y
        # por eso queda anotado: lo que al 0,08 le falta es un ORIGEN, hoy es un número elegido.
        # NO se toca todavía porque bajarlo o quitarlo deja MUDO al organismo, y su voz es la
        # entrada del experimento de convergencia léxica entre A, B, C y D: primero se corrige
        # la causa (hecho), se vuelve a medir (hecho) y se decide con el investigador qué es un
        # intento de exploración —el guepardo falla el 83 % de las veces y sólo paga el intento—.
        self.piso_exploracion = float(piso_exploracion)  # voz mínima para permitir la IGNICIÓN (probar)
        self._A_solo: Optional[float] = None          # baseline de A medido EN SOLEDAD → costo_desacople
        # Lo habitual de cada magnitud, aprendido de la propia experiencia (escala.py).
        self.esc_eR = Escala()      # cuánto suele valer el error de representación propio
        self.esc_dOI = Escala()     # cuánto suele diferir mi OI del OI del par
        self.esc_A_solo = Escala()  # cuánto suele valer A cuando el par NO se puede leer (solo)
        self.ultimo: dict = {}

    # --- persistencia de las escalas -------------------------------------------------
    def snapshot(self) -> dict:
        """Sin esto el organismo reaprende en cada arranque qué error es "el de siempre", y sus
        primeras decisiones sociales del día quedan tomadas contra una escala vacía."""
        return {"eR": self.esc_eR.snapshot(), "dOI": self.esc_dOI.snapshot(),
                "A_solo": self.esc_A_solo.snapshot(), "_A_solo": self._A_solo}

    def restore(self, d: Optional[dict]) -> "GobernanzaAltruismo":
        if isinstance(d, dict):
            self.esc_eR.restore(d.get("eR") or {})
            self.esc_dOI.restore(d.get("dOI") or {})
            self.esc_A_solo.restore(d.get("A_solo") or {})
            a = d.get("_A_solo")
            if a is not None:
                try:
                    self._A_solo = float(a)
                except (TypeError, ValueError):
                    pass
        return self

    def paso(self, fila_propia: dict, estado_par: Optional[dict],
             dt: float = 0.1, tempo: float = 1.0) -> dict:
        fp = fila_propia or {}
        par = (estado_par or {}).get("fila", {}) if estado_par else {}
        m = self.mil
        A = float(fp.get("A_sys_env", 0.0))
        par_legible = bool(par) and (float(par.get("C_b", par.get("Cb", 0.0)) or 0.0) > 0.0)

        # --- EL ERROR DE REPRESENTACIÓN, EN LA ESCALA DEL PROPIO ORGANISMO ---------------
        # QUÉ ESTABA MAL — es el núcleo de por qué cooperar es imposible hoy. Se entregaba |e_R|
        # EN CRUDO a dos comparaciones que lo tratan como si tuviera las mismas unidades que
        # A_sys_env. No las tiene. MEDIDO sobre 100.977 pasos:
        #     A_sys_env ∈ [0,1638 , 0,9000]   |e_R| mediana 8,66 · máximo 26,29   (17×)
        #   · Hamilton (A/e_R > 1/S_shared): el techo aritmético de A/e_R es
        #     max(A)/min(|e_R|) = 0,9000/0,5000 = 1,8000, y el piso aritmético de 1/S es
        #     1/max(S) = 1,0000. La ventana entera donde Hamilton PUEDE cumplirse es
        #     A/e_R ∈ (1,0 ; 1,8]. Con el par presente, A/e_R tiene mediana 0,0442 — veintitrés
        #     veces por debajo del piso del umbral. hamilton_ok: 779 de 100.977 = 0,77 %.
        #   · Y esos 779 son TODOS de la rama de silencio: |e_R| vale exactamente 0,5 en el
        #     44,92 % de los pasos (y ahí A vale exactamente 0,9000 en el 100 % de ellos), y
        #     fuera de ese suelo el mínimo de |e_R| es 5,0823. NO EXISTE UN SOLO PASO con
        #     |e_R| entre 0,5 y 5,08 en todo el corpus. O sea: la única forma de cooperar era
        #     que el organismo no oyera nada y el par estuviera leíble en ese preciso instante
        #     (coincidieron 780 veces de 45.359 silencios: 1,72 %).
        #   · β_crit = √(1−[A/(A+e_R)]^(1/LF)): con e_R 17× mayor que A el reparto lo domina el
        #     error y β_crit sale 1,0000 —umbral inalcanzable— en el 49,43 % de los pasos.
        # AUTORREGULADO: rel(|e_R|, historia propia) → 0,5 cuando el error es el de siempre.
        # Aquí NO cabe rel_contra: no hay ninguna otra magnitud del organismo con las mismas
        # unidades que e_R contra la que compararlo —que A y e_R no sean conmensurables es
        # justamente la avería—, así que la media móvil es el recurso correcto y no un atajo.
        # Y es INVARIANTE DE ESCALA: cuando la fase 2 del plan corrija e_R en el campo, este
        # mapeo seguirá significando lo mismo sin tocar nada.
        eR_bruto = abs(float(fp.get("e_R", 0.0)))
        self.esc_eR.observar(eR_bruto)
        eR_rel = rel(eR_bruto, self.esc_eR) if self.esc_eR.madura else NEUTRO

        # --- baseline de A "EN SOLITARIO": el contrafáctico de costo_desacople ------------
        # QUÉ ESTABA MAL: se guardaba con min() —un trinquete que sólo puede bajar— y se
        # actualizaba mientras `disposicion < 0.05`, un corte que nadie midió y que se cumple el
        # 94,28 % de los pasos. Resultado: _A_solo converge al MÍNIMO DE LA VIDA de A (0,1638
        # medido) y, con A mediana 0,4954, costo_desacople sale 0,5651 de mediana y supera el
        # 0,30 que exige el genoma en el 92,24 % de los pasos. Es decir: la díada venía
        # "aprobando" el criterio de unidad de orden superior (Boorman 5: subaditividad) por un
        # artefacto de medición. Ese 0,5651 no dice nada del otro organismo: dice sólo que A está
        # un 56 % por encima de su mínimo histórico.
        # AUTORREGULADO: A "en solitario" es A CUANDO EL PAR NO ESTÁ, y eso el módulo lo sabe sin
        # ningún umbral — es si pudo leer al par o no. Media móvil sobre esos pasos: puede subir
        # y bajar, y es el contrafáctico de verdad. Mientras esa escala no tenga historia,
        # _A_solo = A ⇒ costo_desacople = 0: el organismo se abstiene en vez de inventarse un
        # beneficio de estar acompañado (política de arranque única de escala.py).
        # MEDIDO en el corpus: el par fue ilegible el 44,70 % de los pasos, así que la escala
        # tiene de sobra con qué madurar.
        if not par_legible:
            self.esc_A_solo.observar(A)
        self._A_solo = self.esc_A_solo.media if self.esc_A_solo.madura else A

        # --- MAPEO fila → entradas canónicas del organelo (la deducción de Boorman) ---
        m.secretar("delta_struct", float(fp.get("R2", 0.0)))              # readiness social (meta-rep R₂)
        m.secretar("LF", float(fp.get("LF_op", fp.get("LF_struct", 0.0))))  # libertad funcional
        m.secretar("e_R", eR_rel)                                        # error operativo (= costo τ)
        m.secretar("A_sys_env", A)                                       # acoplamiento (beneficio σ)
        m.secretar("A_sys_env_solo", float(self._A_solo))               # contrafáctico para costo_desacople
        m.secretar("ME", _s_shared(fp, par, self.esc_dOI))              # S_shared (r de Hamilton)
        # estado del OTRO (lo que su altruismo y mi Ψ_alma necesitan)
        m.secretar("otro.Cb", float(par.get("C_b", par.get("Cb", 0.0))))
        m.secretar("otro.valencia", float(par.get("disposicion_cooperar", 0.0)))  # ¿el otro quiere? (reciprocidad)
        # QUÉ ESTABA MAL: el organelo hace `objetivo *= max(0, 1 − 0,5·costo_cooperar)`, o sea
        # espera un costo ADIMENSIONAL en [0,1]; recibía |e_R| con mediana 8,66. MEDIDO: ese
        # factor tomó DOS valores distintos en 100.542 pasos — 0,0000 en el 54,89 % (siempre que
        # |e_R| ≥ 2) y 0,7500 en el 45,11 % (el suelo |e_R| = 0,5). Nada en medio, nunca.
        # No era un hándicap graduado: era un INTERRUPTOR que apagaba la disposición en más de
        # la mitad de la vida del organismo, y que además apagaba lo mismo con |e_R| = 2 que
        # con |e_R| = 26.
        # AUTORREGULADO: el mismo rel — comunicar cuesta 0,5 cuando cuesta lo de siempre, sube
        # cuando el organismo representa peor que de costumbre y baja cuando representa mejor.
        # Es PERCEPCIÓN («¿me sale caro hablar AHORA?»), no condición de viabilidad: la condición
        # de vida sigue siendo Hamilton, que permanece absoluta (advertencia 2 de escala.py).
        m.secretar("costo_cooperar", eR_rel)                            # comunicar cuesta (hándicap)
        # SLOT SIN CONECTAR, se deja en 1,0 y se dice por qué: multiplica `objetivo` por 1,0 en el
        # 100 % de los pasos, o sea hoy no decide nada (no viola la norma: no discrimina a nadie).
        # Lo que DEBERÍA llenarlo es la disponibilidad metabólica real del organismo para gastar
        # en cooperar. El candidato natural es `met_energia`, pero MEDIDO tiene mediana 0,0000
        # (p95 = 0,6582) y conectarlo hoy apagaría la cooperación por un defecto ajeno. Queda
        # anotado como dependencia de la fase 4 del plan (metabolismo).
        m.secretar("estado_reproductivo", 1.0)                          # disponibilidad (1=libre)

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
            # QUÉ ESTABA MAL: `disp > 0.15` era un corte sobre una escala de disposición que
            # nadie midió. MEDIDO: "emergiendo" 795 de 100.542 pasos (0,79 %), "mudo" el 99,21 %
            # y "comunicando" CERO veces. Y la etiqueta no es cosmética: sale al CSV como
            # `altruismo_atractor` y a los monitores de la díada.
            # AUTORREGULADO POR COMPARACIÓN con otra magnitud de LAS MISMAS UNIDADES que ya vive
            # en este módulo: el piso de exploración. "emergiendo" pasa a significar algo
            # verificable —la disposición ya manda sobre la voz, en vez de mandar la
            # exploración— y no introduce ningún número nuevo.
            # A PROPÓSITO no se relativiza contra la propia historia: un organismo mudo de forma
            # sostenida leería "emergiendo" la mitad del tiempo, que es exactamente el fallo que
            # la advertencia 2 de escala.py manda evitar. Mudo/emergiendo es condición, no
            # percepción. Y "comunicando" lo sigue decidiendo `coopera`, que es absoluto.
            "atractor": ("comunicando" if self.org.coopera
                         else "emergiendo" if disp > self.piso_exploracion else "mudo"),
            # Publicados para que esto sea AUDITABLE desde el CSV (lote F del plan): sin estas
            # dos columnas no hay forma de ver desde fuera qué error y qué contrafáctico
            # entraron realmente al organelo.
            "e_R_rel": round(eR_rel, 4),
            "A_sys_env_solo": round(float(self._A_solo), 4),
            "par_legible": bool(par_legible),
        }
        return self.ultimo


# ==============================================================================
# SMOKE / TESTS — simula la díada SIN HTTP (dos gobernanzas que se reciprocan)
# ==============================================================================
# QUÉ ESTABA MAL EN ESTE SMOKE. La auditoría lo llamó «una prueba calibrada para pasar» y
# tenía razón: sembraba `e_R = 0.05` cuando la mediana real del corpus es 8,66 —173 veces
# menos— y además aflojaba `tau_min` de 8,0 s a 0,5 s. Con esos dos retoques la cooperación se
# encendía siempre y NINGUNO de los defectos de escala de este archivo podía manifestarse.
# Como era la única red de seguridad del módulo, es la razón por la que llegaron intactos.
#
# Ahora son dos pruebas y ninguna se siembra a ojo:
#   · _smoke_regresion(): usa las CIFRAS MEDIDAS del corpus y comprueba que con ellas el
#     organismo NO coopera. Es el guardián: el día que la fase 2 del plan corrija e_R en el
#     campo, esta prueba cambiará de resultado — y hay que enterarse, no descubrirlo por
#     casualidad seis semanas después.
#   · _smoke_ignicion(): comprueba que el mecanismo SÍ puede encenderse cuando el estado del
#     organismo mejora respecto de SU PROPIO hábito. Sigue aflojando tau_min, pero lo dice en
#     voz alta y explica por qué: τ_min = 8 s son 80 pasos de dt = 0,1 s de mutualismo
#     ininterrumpido, y el test corre en memoria, no en tiempo real.

# ORIGEN: medido sobre 100.992 pasos del organismo ANIMA_5Z934MWHNNRH
# (~/.anima/history/organismo_ANIMA_5Z934MWHNNRH/fisiologia/*.csv, 3–4 ago 2026).
# Son MEDICIONES, no ajustes. Si se cambia el corpus hay que volver a medirlas y anotar el
# corte de época, porque dejan de ser comparables.
CORPUS = dict(
    e_R_med=8.66,            # mediana de |e_R|  (el suelo de la rama de silencio es 0,5)
    e_R_min_sin_silencio=5.08,   # mínimo de |e_R| fuera de ese suelo: no hay nada entre 0,5 y 5,08
    A_med=0.4954, A_max=0.90,    # A_sys_env: mediana y máximo (el máximo es un decreto del campo)
    C_b_med=5.0, OI_med=0.4130, R2_med=0.9878, LF_op_med=0.3109,
    OI_recorrido=0.3883,     # p99 − p01 de OI: TODO lo que |ΔOI| puede llegar a valer de verdad
)


def _fila_corpus(e_R: float, A: float, oi: float = None, cb: float = None) -> dict:
    """Una fila con la forma y las magnitudes REALES del organismo."""
    return {"R2": CORPUS["R2_med"], "LF_op": CORPUS["LF_op_med"],
            "e_R": e_R, "A_sys_env": A,
            "C_b": CORPUS["C_b_med"] if cb is None else cb,
            "OI": CORPUS["OI_med"] if oi is None else oi}


def _smoke_regresion() -> None:
    """Con las cifras MEDIDAS del organismo, la cooperación NO se enciende. Y se documenta por
    qué: el beneficio (A ≤ 0,90) no puede superar al costo (|e_R| mediana 8,66) en la regla de
    Hamilton, cuyo umbral 1/S_shared no baja nunca de 1,0."""
    gob = GobernanzaAltruismo()
    fila = _fila_corpus(CORPUS["e_R_med"], CORPUS["A_med"])
    par = {"fila": _fila_corpus(CORPUS["e_R_med"], CORPUS["A_med"], oi=CORPUS["OI_med"] + 0.05)}
    for _ in range(400):
        r = gob.paso(fila, par)
    assert r["coopera"] is False, ("con las cifras reales del corpus la cooperación no se "
                                   f"enciende; si esto falla, e_R cambió de escala: {r}")
    # La razón exacta, comprobada y no supuesta: el hándicap ya NO es un interruptor.
    assert 0.0 < r["e_R_rel"] < 1.0, f"e_R debe entrar al organelo en [0,1], no en crudo: {r}"
    # Y con el error clavado en su valor habitual, rel devuelve ~0,5 (la definición de 'lo de
    # siempre'), en vez de los 8,66 crudos que anulaban `objetivo` por completo.
    assert abs(r["e_R_rel"] - 0.5) < 0.15, f"error habitual ⇒ ~0,5; medido {r['e_R_rel']}"
    print(f"   regresión (cifras reales) → coopera={r['coopera']} e_R_rel={r['e_R_rel']} "
          f"S_shared={r['S_shared']} atractor={r['atractor']}")


def _smoke_ignicion() -> None:
    """El mecanismo SÍ puede encenderse — pero ahora se enciende por lo que el organismo mide
    de sí mismo, no por una siembra a medida.

    LO QUE ESTA PRUEBA AFLOJA, DICHO EN VOZ ALTA: `tau_min` baja de 8,0 s a 0,5 s. τ_min son
    80 pasos de dt = 0,1 s de mutualismo ININTERRUMPIDO y esta prueba corre en memoria, no en
    tiempo real; con 8,0 s el bucle tendría que dar 80 vueltas sin un solo corte, que es un test
    de otra cosa. NO se afloja nada más: e_R, A y el resto son magnitudes del corpus.

    LO QUE NO SE SIEMBRA: el error arranca en su valor habitual medido (8,66) y BAJA hasta el
    mínimo real observado fuera de la rama de silencio (5,08). Esa bajada es la historia
    biológica del asunto —la relación se construye y representar al otro cuesta menos— y es la
    que `rel` convierte en «hoy me sale más barato hablar que de costumbre». Con el código
    anterior esta misma bajada no encendía nada: 5,08 y 8,66 dan los dos `1 − 0,5·|e_R| = 0`.

    Y LA PRUEBA EMPIEZA EN SOLEDAD, que no es un detalle del test sino la condición del
    criterio: `costo_desacople` compara A acompañado contra A en solitario, así que sin haber
    estado nunca solo el organismo NO PUEDE saber qué gana con el otro, y se abstiene (vale 0).
    Eso es correcto y es nuevo: antes ese contrafáctico se rellenaba con el mínimo histórico de
    A, que daba 0,56 sin haber medido jamás la soledad. En el corpus el par resultó ilegible el
    44,70 % de los pasos, así que material para esa escala sobra.
    """
    gobA = GobernanzaAltruismo(plast=dict(tau_min=0.5))
    gobB = GobernanzaAltruismo(plast=dict(tau_min=0.5))

    def fila(prog, cb=None, oi=None):
        # el error baja del habitual al mejor realmente observado; A sube al acoplarse
        eR = CORPUS["e_R_med"] + (CORPUS["e_R_min_sin_silencio"] - CORPUS["e_R_med"]) * prog
        return _fila_corpus(eR, CORPUS["A_med"] + (CORPUS["A_max"] - CORPUS["A_med"]) * prog,
                            oi=oi, cb=cb)

    # FASE 1 — EN SOLITARIO: el par no se puede leer. Aquí el organismo aprende cuánto vale su
    # error habitual y cuánto vale su acoplamiento cuando está solo.
    for _ in range(150):
        gobA.paso(fila(0.0), None); gobB.paso(fila(0.0), None)

    # FASE 2 — ACOMPAÑADO: aparece el par. Tres cosas mejoran a la vez, y las tres son
    # magnitudes del organismo, no perillas: el acoplamiento sube, el error baja, y los dos OI
    # CONVERGEN desde todo el recorrido real de OI (0,3883 = p99 − p01) hacia el acuerdo.
    # Esa convergencia es la que ejercita la alineación corregida: con el `1 − |ΔOI|` anterior
    # todo este trayecto se movía dentro de [0,61 , 1] y apenas cambiaba nada.
    rA = {"fila": fila(0.0)}; rB = {"fila": fila(0.0, oi=CORPUS["OI_med"] + CORPUS["OI_recorrido"])}
    for k in range(400):
        prog = min(1.0, k / 399.0)
        oi_b = CORPUS["OI_med"] + CORPUS["OI_recorrido"] * (1.0 - prog)
        fa = fila(prog); fb = fila(prog, oi=oi_b)
        a = gobA.paso(fa, rB); b = gobB.paso(fb, rA)
        rA = {"fila": {**fa, "disposicion_cooperar": a["disposicion_cooperar"]}}
        rB = {"fila": {**fb, "disposicion_cooperar": b["disposicion_cooperar"]}}

    assert gobA.ultimo["e_R_rel"] < 0.5, ("representar mejor que de costumbre debe abaratar el "
                                          f"hándicap: {gobA.ultimo}")
    assert gobA.ultimo["costo_desacople"] > 0.30, ("acompañado el acoplamiento sube ⇒ separarse "
                                                   f"cuesta (Boorman 5): {gobA.ultimo}")
    assert gobA.ultimo["disposicion_cooperar"] > 0.5, gobA.ultimo
    assert gobA.ultimo["coopera"] is True and gobB.ultimo["coopera"] is True, "díada recíproca → coopera"
    assert gobA.ultimo["voice_rms"] > gobA.base_voice_rms * 0.5, "voz costosa sube con la disposición"
    assert gobA.ultimo["atractor"] == "comunicando", gobA.ultimo
    print(f"   ignición (τ_min aflojado, magnitudes reales) → disposicion="
          f"{gobA.ultimo['disposicion_cooperar']} e_R_rel={gobA.ultimo['e_R_rel']} "
          f"β_crit={gobA.ultimo['beta_crit']} voz={gobA.ultimo['voice_rms']} "
          f"S_shared={gobA.ultimo['S_shared']} atractor={gobA.ultimo['atractor']}")

    # DESALMAMIENTO: el par NO es sujeto (C_b=0) → no emerge cooperación, voz al piso.
    # Esta prueba estaba bien y se conserva tal cual: comprueba una condición ABSOLUTA
    # (anti-Shannon, O-N3.4b) que no debe relativizarse nunca.
    gobC = GobernanzaAltruismo(plast=dict(tau_min=0.5))
    parNS = {"fila": fila(1.0, cb=0.0)}
    for _ in range(200):
        gobC.paso(fila(1.0), parNS)
    assert gobC.ultimo["psi_alma"] == 0.0 and gobC.ultimo["disposicion_cooperar"] < 0.05, gobC.ultimo
    assert gobC.ultimo["coopera"] is False
    assert abs(gobC.ultimo["voice_rms"] - gobC.base_voice_rms * gobC.piso_exploracion) < 1e-6, "voz al piso"
    print(f"   sin-sujeto → disposicion={gobC.ultimo['disposicion_cooperar']} "
          f"psi_alma={gobC.ultimo['psi_alma']} voz={gobC.ultimo['voice_rms']} "
          f"atractor={gobC.ultimo['atractor']}")


def _smoke() -> None:
    print("VST_DiadaAltruismo — pruebas sobre magnitudes MEDIDAS del organismo:")
    _smoke_regresion()
    _smoke_ignicion()
    print("OK: el handicap ya no es un interruptor; con las cifras reales NO coopera (y se "
          "sabe por que); con el error por debajo de su habito, enciende.")


if __name__ == "__main__":
    _smoke()
