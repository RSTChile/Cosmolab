#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_RC_A - E018 endogeno
==========================

Organelo observacional para O-N1:

    RC = ICR + IRDE

Esta version elimina las mezclas ponderadas manuales. Las magnitudes se forman por
simetria, normalizacion relativa y competencia entre bases endogenas:

  - RC_relacional / RC_externo nacen de union probabilistica de evidencias del canal.
  - ICR_ratio / IRDE_ratio nacen de competencia conversion-vulnerabilidad.
  - atencion / comprension / riesgo usan medias simetricas y pesos relativos del propio RC.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading


COLS_RC = [
    "RC_total", "RC_externo", "RC_relacional",
    "ICR", "IRDE", "ICR_ratio", "IRDE_ratio",
    "RC_delta_salud", "destino_RC",
    "RC_atencion_L", "RC_atencion_R",
    "RC_comprension_L", "RC_comprension_R",
    "RC_ema_comp_L", "RC_ema_comp_R",
    "RC_riesgo_L", "RC_riesgo_R",
    "RC_consenso_orientacion", "RC_confianza_comprension", "RC_freno_riesgo",
    "RC_base_relacional", "RC_base_externo",
    "RC_soporte_conversion", "RC_vulnerabilidad_desviacion",
    "RC_base_ICR", "RC_base_IRDE", "RC_peso_ICR", "RC_peso_IRDE",
    "RC_apertura_desacople",
]


def _num(x, default=0.0):
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if math.isfinite(v) else float(default)


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _media(vals):
    xs = [_clip01(v) for v in vals if math.isfinite(float(v))]
    return _clip01(sum(xs) / max(1, len(xs)))


def _union(*vals):
    y = 1.0
    for v in vals:
        y *= 1.0 - _clip01(v)
    return _clip01(1.0 - y)


def _competir(a, b):
    a = max(0.0, float(a)); b = max(0.0, float(b))
    den = a + b
    if den <= 1e-12:
        return 0.0, 0.0
    return _clip01(a / den), _clip01(b / den)


def _reparto_neutro() -> bool:
    return (os.environ.get("ANIMA_REPARTO_NEUTRO", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _conversion_libre() -> bool:
    return (os.environ.get("ANIMA_CONVERSION_LIBRE", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _stable_seed(text: str) -> int:
    """Mismo idioma que VST_OrganoComunicacion: sorteo estable, reproducible y no manipulable
    desde fuera. Es como el organelo de comunicación implementa P_CREAR/P_EMULAR."""
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def _abrir_por_desacople(peso_icr, peso_irde, Aenv, e_R, oid="ANIMA", seq=0):
    """LIBERTAD FUNCIONAL — el DESACOPLE I↔E abre la conversión.

    S = I↔E. Una diferencia no persiste por lo que la entidad tenga adentro, sino por cómo
    se sostiene su proporción con el afuera. Lo que abre aquí la conversión no puede ser
    entonces una magnitud interior: tiene que ser la relación.

    LA RELACIÓN, LITERAL. El genoma ya la tiene escrita —`A_sys_env` es el acoplamiento
    sistema-entorno, y `presion_desacople = e_R·(1−A_sys_env)` es, en sus palabras, "la
    TENSIÓN de no estar acoplado al medio"—. Se usa esa misma forma, en su versión
    instantánea, con los dos valores que el reparto ya tiene en la mano: no hay magnitud
    prestada de otro organelo ni desfase de un paso.

    QUÉ HACE. Acoplado (A→1) la tensión tiende a 0, el rango se cierra solo y la conversión
    sigue lo que dé la marcha del organismo: el canónico queda intacto sin excepciones.
    Desacoplado, el rango se abre — y se abre justo donde convertir es lo que restauraría
    la relación. No empuja ni obliga: hace posible.

    EL PROBLEMA. `peso_icr` sale de `mejoras`: cuenta cuántas variables de estado SUBEN
    frente a cuántas BAJAN. Convertir es lo único que permite mejorar, y sólo se puede
    convertir si ya se está mejorando — el lazo se cierra sobre sí mismo y no tiene salida
    propia. (La magnitud del atrapamiento está aún por medir: los `RC_peso_ICR` de los
    registros viejos duplicaban `ICR_ratio` por un error de la columna, corregido el
    2-ago-2026, así que `peso_icr` sólo es observable desde entonces.)

    POR QUÉ NO SE ARREGLA CON UNA CONSTANTE. Poner un piso fijo a peso_icr resolvería el
    síntoma violando el principio: el organismo quedaría determinado por un número nuestro.
    La salida tiene que ser un acto suyo.

    RANGO, NO VALOR. Nosotros fijamos el RANGO; el organismo elige dentro. La tensión sólo
    mueve el TECHO: el suelo sigue siendo lo que la marcha del organismo da por sí sola, y
    dónde cae la conversión entre ambos lo resuelve él, con el mismo sorteo estable con que
    P_CREAR/P_EMULAR implementan la libertad en el organelo de comunicación. No hay umbral
    ni probabilidad puestos a mano, y nada de fuera decide el punto.

    LO QUE NO ES. No es prescribir una conducta ni calcular el paso obligatorio siguiente.
    El organismo no marcha: baila. El rango es el espacio donde esa danza cabe — sostener
    el acoplamiento con el entorno y con el propio metabolismo, no ejecutar una jugada
    forzada.

    HISTORIAL. La primera versión abría el rango con el HAMBRE (`met_hambre_prev`). Aunque
    endógena, el hambre es una magnitud del interior y consecuencia del metabolismo, no la
    relación: usarla era volver a poner un parámetro donde va una proporción. Se sustituyó
    por la tensión I↔E el 2-ago-2026; con ello se cayeron también el desfase de un paso y
    la inyección desde el integrador, que eran andamios de aquel error.
    """
    if not _conversion_libre():
        return peso_icr, peso_irde
    tension = _clip01(_num(e_R) * (1.0 - _clip01(_num(Aenv))))
    techo = max(peso_icr, tension)
    if techo - peso_icr <= 1e-12:   # acoplado: la relación no abre nada, rango nulo
        return peso_icr, peso_irde
    u = (_stable_seed(f"{oid}:conversion:{seq}") % 1000000) / 1000000.0
    elegido = _clip01(peso_icr + u * (techo - peso_icr))
    return elegido, _clip01(1.0 - elegido)


def _relativo(x: float, habitual: float) -> float:
    """Dónde cae `x` respecto de LO HABITUAL PARA ESTE ORGANISMO. Sin ningún parámetro libre.

    El problema (medido el 3-ago-2026): `_media` recorta cada término a [0,1], y `e_R` tiene mediana
    ~20. Es decir que el error de representación aportaba 1,0 —el máximo posible de vulnerabilidad—
    en el 86% de los pasos, y `1−Λ_Cos` aportaba 0,998 porque Λ vive en ~0,002. Dos de los seis
    términos de la vulnerabilidad eran constantes clavadas en el techo: no medían, decidían.

    La escala no puede ser un número nuestro, porque cada organismo vive en un ambiente distinto y
    lo que para uno es un error grande para otro es la norma. Aquí la referencia es SU PROPIO valor
    habitual, y el mapeo r/(1+r) da exactamente 0,5 cuando el valor es el de siempre, sube al
    acercarse a 1 cuando es mucho mayor y baja hacia 0 cuando es mucho menor. No hay constante que
    elegir: la escala la pone la historia del organismo.
    """
    if habitual <= 1e-12:
        return 0.5
    r = max(0.0, float(x)) / habitual
    return r / (1.0 + r)


def _soporte_vulnerabilidad(OI, H, Aenv, LF, R2, C_m, XE, e_R, Lambda):
    """Las dos fuerzas que compiten por repartir RC en ICR (convertir) e IRDE (disipar).

    CANÓNICO (histórico) — dos promedios sobre conjuntos DISTINTOS: R2/C_m/XE sólo suman
    al soporte; e_R y 1−Λ sólo a la vulnerabilidad. NO son complementarios: medidos sobre
    58.290 pasos reales de los cuatro organismos del laboratorio dan 0,555 y 0,845, que
    suman 1,40. La competencia arranca ~40/60 en contra de la conversión ANTES de que el
    organismo procese nada, y ese sesgo de entrada es lo que las constantes asimétricas
    del metabolismo (k_ingesta 0,30 vs k_toxico 0,04) venían compensando a mano.

    NEUTRO (ANIMA_REPARTO_NEUTRO=1) — una sola medida de soporte, sobre el mismo conjunto
    y con la MISMA POLARIDAD que ya usa `mejoras` más abajo (Λ bueno cuando sube, e_R bueno
    cuando baja), y la vulnerabilidad como su complemento exacto. Un sonido no trae carga
    previa: reparte 50/50 y es el procesamiento del organismo —no una constante puesta a
    mano— el que lo desplaza hacia alimento o hacia indigestión.

    Requiere ANIMA_REPARTO_NEUTRO también en el metabolismo (k_toxico = k_ingesta): mover
    sólo uno de los dos deja al organismo peor que antes, porque se quita el parche sin
    quitar el sesgo que compensaba.
    """
    if _reparto_neutro():
        soporte = _media([OI, H, Aenv, LF, R2, C_m, XE, Lambda, 1.0 - e_R])
        return soporte, _clip01(1.0 - soporte)
    # e_R y Lambda llegan YA relativizados a lo habitual del organismo (ver _relativo): antes
    # entraban en crudo y el recorte a [0,1] los convertia en constantes.
    return (_media([OI, H, Aenv, LF, R2, C_m, XE]),
            _media([1.0 - OI, 1.0 - H, 1.0 - Aenv, 1.0 - LF, e_R, 1.0 - Lambda]))


class OrganoRC:
    """Metabolismo observacional de RC -> ICR/IRDE, sin ponderaciones manuales."""

    def __init__(self, ema: float = 0.20) -> None:
        self.ema = _clip01(ema)
        self._prev = None
        self._lock = threading.Lock()
        self._ema_rc = 0.0
        # LO HABITUAL PARA ESTE ORGANISMO (3-ago-2026): referencia movil con que se relativizan las
        # magnitudes que no viven en [0,1]. Arranca en None y se llena con la primera observacion,
        # para no imponer un punto de partida. Usa self.ema, la constante de tiempo que el organelo
        # ya tenia declarada: no se introduce ninguna nueva.
        self._hab_eR = None          # error de representacion habitual
        self._hab_lambda = None      # razon cosmosemiotica habitual
        self._hab_estructura = None  # orden habitual del alimento (la voz de los pares ES su dieta)
        self._ema_icr = 0.0
        self._ema_irde = 0.0
        self._ema_comp_l = 0.0     # comprensión sostenida por lado: memoria del LAZO de atención (lo que nutrió atrae la mirada)
        self._ema_comp_r = 0.0
        # TOGGLE del lazo por env ANIMA_RC_LOOP (on/off, def on): permite experimentos limpios sin swap de código.
        self.loop_on = os.environ.get("ANIMA_RC_LOOP", "on").strip().lower() in ("on", "1", "true", "yes")
        self._energy_ref = 1e-9
        # Identidad y paso, para que el sorteo de la Libertad Funcional sea propio del
        # organismo y reproducible (mismo idioma que P_CREAR en el organelo de comunicación).
        self._oid = os.environ.get("VST_ORGANISMO_ID", "ANIMA")
        self._seq = 0

    def reset(self) -> None:
        with self._lock:
            self._prev = None
            self._ema_rc = self._ema_icr = self._ema_irde = 0.0
            self._ema_comp_l = self._ema_comp_r = 0.0
            self._energy_ref = 1e-9

    def _energia_relativa(self, raw_l: float, raw_r: float) -> tuple[float, float]:
        raw_l = max(0.0, _num(raw_l)); raw_r = max(0.0, _num(raw_r))
        pico = max(raw_l, raw_r, self._energy_ref)
        self._energy_ref = max(1e-9, max(pico, self._energy_ref * (1.0 - self.ema)))
        return _clip01(raw_l / (raw_l + self._energy_ref)), _clip01(raw_r / (raw_r + self._energy_ref))

    def observar(self, fila: dict, meta: dict | None = None) -> dict:
        with self._lock:
            self._seq += 1
            prev = dict(self._prev) if self._prev else None

            energia_l, energia_r = self._energia_relativa(fila.get("energia_L", 0.0), fila.get("energia_R", 0.0))
            balance_signed = _num(fila.get("balance_LR", 0.0))
            balance = _clip01(abs(balance_signed))
            coher = _clip01(abs(_num(fila.get("coherencia_biaural", 0.0), 0.0)))

            w_l = _num(fila.get("omega_L", 0.0))
            w_r = _num(fila.get("omega_R", 0.0))
            w_a = _num(fila.get("omega_A", 0.0))
            sal_l = _clip01(abs(w_l - w_a)) * energia_l
            sal_r = _clip01(abs(w_r - w_a)) * energia_r

            novelty_l = novelty_r = novelty = 0.0
            if prev:
                novelty_l = _clip01(abs(w_l - _num(prev.get("omega_L", w_l)))) * energia_l
                novelty_r = _clip01(abs(w_r - _num(prev.get("omega_R", w_r)))) * energia_r
                novelty_terms = [
                    abs(_num(fila.get("Omega")) - _num(prev.get("Omega"))),
                    abs(_num(fila.get("R2")) - _num(prev.get("R2"))),
                    abs(_num(fila.get("LF_op")) - _num(prev.get("LF_op"))),
                    abs(_num(fila.get("C_m")) - _num(prev.get("C_m"))),
                    abs(_num(fila.get("XE")) - _num(prev.get("XE"))),
                    abs(_num(fila.get("H_homeostasis")) - _num(prev.get("H_homeostasis"))),
                ]
                novelty = _media(novelty_terms)

            incoher = 1.0 - coher
            base_rel = _media([energia_l, sal_l, balance * energia_l, novelty_l])
            base_ext = _media([energia_r, sal_r, incoher * energia_r, novelty_r])
            rc_rel = _union(base_rel, novelty_l)
            rc_ext = _union(base_ext, novelty_r)
            rc_total = _union(rc_rel, rc_ext, novelty)

            OI = _clip01(_num(fila.get("OI")))
            H = _clip01(_num(fila.get("H_homeostasis")))
            Aenv = _clip01(_num(fila.get("A_sys_env")))
            LF = _clip01(_num(fila.get("LF_op")))
            R2 = _clip01(_num(fila.get("R2")))
            C_m = _clip01(_num(fila.get("C_m")))
            XE = _clip01(_num(fila.get("XE")))
            e_R = _clip01(abs(_num(fila.get("e_R"))))
            Lambda = _clip01(_num(fila.get("Lambda_Cos")))

            if prev:
                mejoras = [
                    OI - _clip01(_num(prev.get("OI"))),
                    H - _clip01(_num(prev.get("H_homeostasis"))),
                    Lambda - _clip01(_num(prev.get("Lambda_Cos"))),
                    Aenv - _clip01(_num(prev.get("A_sys_env"))),
                    LF - _clip01(_num(prev.get("LF_op"))),
                    C_m - _clip01(_num(prev.get("C_m"))),
                    XE - _clip01(_num(prev.get("XE"))),
                    _clip01(abs(_num(prev.get("e_R")))) - e_R,
                ]
                conversion = sum(max(0.0, x) for x in mejoras)
                desviacion = sum(max(0.0, -x) for x in mejoras)
                peso_icr, peso_irde = _competir(conversion, desviacion)
                delta = conversion - desviacion
                if conversion + desviacion <= 1e-12:
                    soporte_conversion, vulnerabilidad = _soporte_vulnerabilidad(
                        OI, H, Aenv, LF, R2, C_m, XE,
                        _relativo(abs(e_R), self._hab_eR or abs(e_R)),
                        _relativo(Lambda, self._hab_lambda or Lambda))
                    peso_icr, peso_irde = _competir(soporte_conversion, vulnerabilidad)
            else:
                soporte_conversion, vulnerabilidad = _soporte_vulnerabilidad(
                    OI, H, Aenv, LF, R2, C_m, XE,
                    _relativo(abs(e_R), self._hab_eR or abs(e_R)),
                    _relativo(Lambda, self._hab_lambda or Lambda))
                peso_icr, peso_irde = _competir(soporte_conversion, vulnerabilidad)
                delta = 0.0

            # ── APRENDER LO HABITUAL, Y JUZGAR CONTRA ESO (3-ago-2026) ────────────────────
            # `e_R` (mediana ~20) y `Lambda_Cos` (mediana ~0,002) no viven en [0,1]. Al recortarlos
            # aportaban constantes al reparto. Ahora cada uno se compara con SU PROPIO valor
            # habitual, que el organismo va aprendiendo con la constante de tiempo que ya tenia.
            for _n, _v in (("_hab_eR", abs(e_R)), ("_hab_lambda", Lambda)):
                _a = getattr(self, _n)
                setattr(self, _n, float(_v) if _a is None else _a + self.ema * (float(_v) - _a))
            e_R_rel = _relativo(abs(e_R), self._hab_eR)
            Lambda_rel = _relativo(Lambda, self._hab_lambda)
            soporte_conversion, vulnerabilidad = _soporte_vulnerabilidad(
                OI, H, Aenv, LF, R2, C_m, XE, e_R_rel, Lambda_rel)
            # LIBERTAD FUNCIONAL: la tensión I↔E (desacople) abre el rango de conversión
            # aunque el organismo no venga mejorando (ver _abrir_por_desacople). Usa Aenv y
            # e_R, que ya están calculados arriba. Apagado por defecto.
            _peso_antes = peso_icr
            peso_icr, peso_irde = _abrir_por_desacople(
                peso_icr, peso_irde, Aenv, e_R, self._oid, self._seq)
            _apertura = peso_icr > _peso_antes + 1e-12
            # ESTRUCTURA del input (membrana sensorial): el SENTIDO nace del ORDEN. La conversión en sentido
            # (ICR=ICES) sólo RINDE si hay estructura que convertir; el ruido (estructura→0) se DISIPA (IRDE).
            # Default 1.0 → si no llega la señal, comportamiento previo (compat). (Aún no experiencial: directo.)
            estructura = _clip01(_num(fila.get("estructura", 1.0), 1.0))
            # EL ORDEN, CONTRA SU PROPIA DIETA (3-ago-2026). Cuando los organismos estan solos, las
            # palabras de los demas son TODO lo que oyen: esa es su dieta, no un ambiente ruidoso del
            # que rescatan una fraccion. Medido: la estructura de la voz de un par vive entre 0,28 y
            # 0,51. Usada en absoluto, ese 0,39 tipico repartia 40/60 en contra de la conversion
            # ANTES de juzgar nada. Relativizada, la comida normal vale 0,5 —ni buena ni mala— y
            # entonces una palabra especialmente estructurada puntua alto de verdad y el ruido bajo.
            _a = self._hab_estructura
            self._hab_estructura = estructura if _a is None else _a + self.ema * (estructura - _a)
            estructura_rel = _relativo(estructura, self._hab_estructura)
            # EL SOPORTE ENTRA UNA SOLA VEZ. `peso_icr` YA es el resultado de competir soporte contra
            # vulnerabilidad; volver a multiplicar por `soporte_conversion` lo aplicaba dos veces, y
            # lo mismo la vulnerabilidad del otro lado. Como soporte (0,41) y vulnerabilidad (0,87)
            # estan lejos, elevarlos al cuadrado convertia una ventaja de 0,47 en una de 0,22: el
            # doble cobro amplificaba la asimetria en vez de medirla. Es la misma figura que ya
            # habiamos corregido en el metabolismo, aqui arriba.
            base_icr = _clip01(rc_total * peso_icr * estructura_rel)
            base_irde = _clip01(rc_total * (peso_irde + (1.0 - estructura_rel) * peso_icr))
            icr_ratio, irde_ratio = _competir(base_icr, base_irde)

            icr = rc_total * icr_ratio
            irde = rc_total - icr

            # LAZO DE ATENCIÓN: sumar el retorno de la nutrición previa por lado (_ema_comp_*). Arranca en 0
            # → primer paso idéntico al feedforward; el sesgo EMERGE con la historia (sin ponderación manual).
            # Conmutable por ANIMA_RC_LOOP: con loop_on=False es feedforward puro (canónico), para el control 2×2.
            _tl = [sal_l, rc_rel, energia_l, novelty_l]
            _tr = [sal_r, rc_ext, energia_r, novelty_r]
            if self.loop_on:
                _tl.append(self._ema_comp_l); _tr.append(self._ema_comp_r)
            at_l = _media(_tl)
            at_r = _media(_tr)
            rel_sum = rc_rel + rc_ext
            peso_l, peso_r = _competir(rc_rel, rc_ext)
            if rel_sum <= 1e-12:
                peso_l = peso_r = 0.0

            comp_l = _clip01(at_l * icr_ratio * _media([peso_l, soporte_conversion, rc_rel]))
            comp_r = _clip01(at_r * icr_ratio * _media([peso_r, soporte_conversion, rc_ext]))
            riesgo_l = _clip01(at_l * irde_ratio * _media([1.0 - peso_l, vulnerabilidad, rc_rel]))
            riesgo_r = _clip01(at_r * irde_ratio * _media([1.0 - peso_r, vulnerabilidad, rc_ext]))

            consenso = _clip01((comp_r - comp_l + 1.0) / 2.0) * 2.0 - 1.0
            confianza_comp = _clip01(icr_ratio * _media([comp_l + comp_r, soporte_conversion, rc_total]))
            freno_riesgo = _clip01(irde_ratio * _media([riesgo_l + riesgo_r, vulnerabilidad, rc_total]))

            self._ema_rc = (1.0 - self.ema) * self._ema_rc + self.ema * rc_total
            self._ema_icr = (1.0 - self.ema) * self._ema_icr + self.ema * icr
            self._ema_irde = (1.0 - self.ema) * self._ema_irde + self.ema * irde
            # cierre del lazo: recordar cuánto nutrió cada lado ESTE paso (misma constante de tiempo self.ema=0.20)
            self._ema_comp_l = (1.0 - self.ema) * self._ema_comp_l + self.ema * comp_l
            self._ema_comp_r = (1.0 - self.ema) * self._ema_comp_r + self.ema * comp_r

            if rc_total <= 1e-12:
                destino = "silencio"
            elif abs(icr_ratio - irde_ratio) <= 1e-12:
                destino = "mixto"
            else:
                destino = "ICR" if icr_ratio > irde_ratio else "IRDE"

            fila.update({
                "RC_total": round(rc_total, 5),
                "RC_externo": round(rc_ext, 5),
                "RC_relacional": round(rc_rel, 5),
                "ICR": round(icr, 5),
                "IRDE": round(irde, 5),
                "ICR_ratio": round(icr_ratio, 5),
                "IRDE_ratio": round(irde_ratio, 5),
                "RC_delta_salud": round(delta, 6),
                "destino_RC": destino,
                "RC_atencion_L": round(at_l, 5),
                "RC_atencion_R": round(at_r, 5),
                "RC_comprension_L": round(comp_l, 5),
                "RC_comprension_R": round(comp_r, 5),
                "RC_ema_comp_L": round(self._ema_comp_l, 5),   # sesgo de atención emergente por lado (memoria del lazo)
                "RC_ema_comp_R": round(self._ema_comp_r, 5),
                "RC_riesgo_L": round(riesgo_l, 5),
                "RC_riesgo_R": round(riesgo_r, 5),
                "RC_consenso_orientacion": round(consenso, 5),
                "RC_confianza_comprension": round(confianza_comp, 5),
                "RC_freno_riesgo": round(freno_riesgo, 5),
                "RC_base_relacional": round(base_rel, 5),
                "RC_base_externo": round(base_ext, 5),
                "RC_soporte_conversion": round(soporte_conversion, 5),
                "RC_vulnerabilidad_desviacion": round(vulnerabilidad, 5),
                "RC_base_ICR": round(base_icr, 5),
                "RC_base_IRDE": round(base_irde, 5),
                # CORREGIDO 2-ago-2026: estas dos columnas escribían icr_ratio/irde_ratio, o sea
                # DUPLICABAN ICR_ratio/IRDE_ratio en vez de exponer los pesos de la competencia.
                # Con eso, `peso_icr` —que es lo que decide el reparto— era inobservable, y
                # cualquier análisis que las usara razonaba en círculo.
                "RC_peso_ICR": round(peso_icr, 5),
                "RC_peso_IRDE": round(peso_irde, 5),
                # ¿abrió la tensión I↔E la conversión en este paso? (0/1; siempre 0 si el
                # interruptor está apagado). Hace falsable el mecanismo de Libertad Funcional.
                "RC_apertura_desacople": 1 if _apertura else 0,
            })
            self._prev = dict(fila)
            return fila
