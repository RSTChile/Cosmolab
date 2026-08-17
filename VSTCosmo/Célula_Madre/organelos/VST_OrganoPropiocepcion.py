#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoPropiocepcion — el organismo SE SIENTE a sí mismo. La base para CALIFICAR (gustar/disgustar).

POR QUÉ EXISTE (insight de Alexis, 29-jun-2026)
-----------------------------------------------
Veníamos atascados en "¿qué hace que un sonido GUSTE?" sin querer inventar una función arbitraria. La clave
es la PROPIOCEPCIÓN: el organismo califica los sonidos a partir de la SUMA DE SUS ESTADOS — siente su propia
condición total y juzga un sonido por cómo la afecta. Lo que me hace sentir mejor, gusta; lo que me deja
peor, no. El valor no se impone: EMERGE de que el organismo se sienta a sí mismo.

Es PIEZO2 llevado al campo: así como PIEZO2 hace sentir el estiramiento del propio cuerpo (saber dónde están
tus miembros con los ojos cerrados), aquí el organismo siente el "estiramiento" de su propio estado interno.
Fundamento común con la membrana (ver FUNDAMENTO_SENSORIAL.md): un transductor que lee un gradiente —aquí,
el del PROPIO estado— y lo vuelve señal. La membrana mira AFUERA (exterocepción); esto mira ADENTRO.

QUÉ COMPUTA — el BIENESTAR (W), suma integrada de estados, en [0,1]
------------------------------------------------------------------
  prop_bienestar   W = condición global sentida (alto = pleno; bajo = malestar). Suma de lo que sostiene la
                   vida (energía, acople, libertad, homeostasis, integración, cierre Λ) menos lo que la mina
                   (necesidad, error/desacople, riesgo, fatiga, reflejo estapedial = "esto es demasiado").
  prop_vigor       energía + libertad funcional (cuánta capacidad de actuar siente)
  prop_acople      A_sys-env + homeostasis (cuán bien acoplado/regulado se siente)
  prop_malestar    suma de los costos (lo que duele/agota/abruma)
  prop_dW          ΔW respecto al ánimo basal (línea lenta): si SUBE, lo que pasa ahora me MEJORA → base del gusto
  prop_dW_rel      cuán GRANDE es ese ΔW para él (0,5 = un vaivén de los de siempre): la vara de la cara

NO decide conducta. SÓLO se siente. El valor de un sonido (el GUSTO) se aprende afuera comparando W con/sin
ese sonido (lo hace el soma con su memoria perceptual). Persistible (conserva el ánimo basal).
"""
from __future__ import annotations

from escala import Escala, rel as _rel, NEUTRO

COLS_PROP = ["prop_bienestar", "prop_vigor", "prop_acople", "prop_malestar", "prop_dW",
             "prop_dW_rel"]


def _g(fila, *claves, default=0.0):
    """Primer alias presente en la fila (robusto a nombres)."""
    for k in claves:
        if k in fila and fila[k] is not None:
            try:
                return float(fila[k])
            except (TypeError, ValueError):
                pass
    return default

def _c01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


class OrganoPropiocepcion:
    """Siente la SUMA de los estados del organismo → bienestar W. La base de la valoración (gusto)."""

    def __init__(self, organismo_id=None, ema_base: float = 0.02):
        self.organismo_id = organismo_id
        self.ema_base = float(ema_base)       # qué tan lento es el ánimo basal (línea contra la que se compara)
        self.W_base = None                    # ánimo basal (se calibra solo)
        self.esc_desac = Escala()             # presión de desacople habitual (llega en ~105, no en [0,1])
        self.esc_fat = Escala()               # cansancio habitual (llega en ~330: grados recorridos)
        self.esc_eR = Escala()                # error de representación habitual (grados), sólo de reserva
        self.esc_dW = Escala()                # cuánto suele MOVERSE el ánimo: la vara de la cara
        self.ultimo: dict = {}

    def observar(self, fila: dict) -> dict:
        # --- lo que SOSTIENE la vida (suma de estados positivos), cada uno en ~[0,1] ---
        energia = _c01(_g(fila, "met_energia", "energia"))
        A = _c01(_g(fila, "A_sys_env"))
        LF = _c01(_g(fila, "LF_op", "LF"))
        H = _c01(_g(fila, "H_homeostasis", "H"))
        OI = _c01(_g(fila, "OI"))
        Lam = _c01(_g(fila, "Lambda_Cos") / 4.0)          # Λ_Cos canónico (0..4) → 0..1
        # --- lo que la MINA (suma de estados negativos) ------------------------------------
        # TRES DE LOS SEIS SUMANDOS NO MEDÍAN NADA (11-ago-2026). Medido sobre 28.301 filas del
        # 8-ago: `presion_desacople` llega en ~105 y `act_fatiga` en ~330 —son acumuladores sin
        # techo, no fracciones— así que `_c01` los recortaba a 1,0 el 100,0 % y el 99,7 % del
        # tiempo; `e_R/10` es el error CRUDO en grados y vivía en 0,874 con el 16,5 % en el techo;
        # y `mem_reflejo` no lo escribe nadie, así que contaba 0 siempre. Con dos sextos fijos y
        # un tercero casi fijo, `prop_malestar` vivía entre 0,486 y 0,558 —un rango de 0,07 sobre
        # un dominio de 0 a 1— y arrastraba a W a [0,1667 · 0,5695], por debajo del 0,60 que la
        # cara pedía para sonreír: la boca no medía al organismo, medía tres constantes.
        #
        # AHORA cada una se compara con lo habitual EN ELLA MISMA (`rel`, 0,5 = lo de siempre),
        # que es el idioma que VST_Memoria ya usa para estas dos magnitudes exactas (`esc_fat`,
        # `esc_Cb`). No entra ni un número nuevo. Advertencia 2 de escala.py aceptada a
        # conciencia: un organismo crónicamente exhausto leerá «cansancio normal». Es correcto
        # aquí —esto es PERCEPCIÓN, y la percepción se adapta— y la viabilidad sigue siendo
        # absoluta donde debe estarlo: la pérdida real de capacidad vive en `factor_gain`, en el
        # genoma, y no se toca.
        nec = _c01(_g(fila, "necesidad", "met_necesidad"))
        IRDE = _c01(_g(fila, "IRDE"))
        desac_bruto = max(0.0, _g(fila, "presion_desacople"))
        fat_bruto = max(0.0, _g(fila, "act_fatiga", "fatiga_activa"))
        eR_bruto = max(0.0, _g(fila, "e_R"))
        self.esc_desac.observar(desac_bruto)
        self.esc_fat.observar(fat_bruto)
        self.esc_eR.observar(eR_bruto)
        desac = _rel(desac_bruto, self.esc_desac) if self.esc_desac.madura else NEUTRO
        fat = _rel(fat_bruto, self.esc_fat) if self.esc_fat.madura else NEUTRO
        # el malestar del error es el error que NO SE ESTÁ CERRANDO, no el error. Eso ya lo
        # calcula la homeostasis desde el 7-ago (`x_interna_estres`, el exceso sobre lo que este
        # organismo sostiene habitualmente); si el organelo no corre, se cae a su propia escala.
        if fila.get("x_interna_estres") is not None:
            eR = _c01(_g(fila, "x_interna_estres"))
        else:
            eR = _rel(eR_bruto, self.esc_eR) if self.esc_eR.madura else NEUTRO

        vigor = (energia + LF) / 2.0
        acople = (A + H) / 2.0
        # el BIENESTAR es lo SENTIDO/corporal (energía, acople, homeostasis, libertad); OI/Λ (organismicidad
        # abstracta, baja en la célula mínima) entran sólo como matiz leve para no aplastar el sentir.
        sostienen = 0.85 * (energia + A + H + LF) / 4.0 + 0.15 * (OI + Lam) / 2.0
        # sólo promedian los sumandos que EXISTEN: `mem_reflejo` es una columna que nadie escribe,
        # y sumar un cero fantasma en el numerador dividiendo por 6 no es neutro, es diluir.
        minan = [nec, eR, IRDE, fat, desac]
        if fila.get("mem_reflejo") is not None:
            minan.append(_c01(_g(fila, "mem_reflejo")))   # reflejo estapedial: "esto es demasiado"
        malestar = sum(minan) / len(minan)
        placer = _g(fila, "placer_sensorial")              # PLACER SENSORIAL: la armonía física (membrana) gusta
        W = _c01(0.5 + 0.7 * (sostienen - malestar) + 0.32 * placer)   # BIENESTAR integrado [0,1] (+placer de lo bello)

        # ánimo basal (línea lenta) → ΔW = cuánto MEJORA/empeora respecto a mi línea: base causal del gusto
        if self.W_base is None:
            self.W_base = W
        dW = W - self.W_base
        self.W_base += self.ema_base * (W - self.W_base)
        # ¿este movimiento del ánimo es GRANDE para este organismo? La cara necesita una vara, y
        # la única honesta es cuánto suele moverse él: 0,5 = una oscilación de las de siempre.
        # No hay lazo que temer —la cara está declarada sin influencia causal sobre el organismo—
        # así que relativizar aquí no puede convertirse en un trinquete.
        self.esc_dW.observar(abs(dW))
        dW_rel = _rel(abs(dW), self.esc_dW) if self.esc_dW.madura else NEUTRO

        out = {"prop_bienestar": round(W, 4), "prop_vigor": round(_c01(vigor), 4),
               "prop_acople": round(_c01(acople), 4), "prop_malestar": round(_c01(malestar), 4),
               "prop_dW": round(dW, 5), "prop_dW_rel": round(dW_rel, 4)}
        fila.update(out); self.ultimo = out
        return out

    def estado(self) -> dict:
        return dict(self.ultimo)

    _ESCALAS = ("esc_desac", "esc_fat", "esc_eR", "esc_dW")

    def snapshot(self) -> dict:
        d = {"W_base": self.W_base}
        for n in self._ESCALAS:
            d[n] = getattr(self, n).snapshot()
        return d

    def restore(self, d: dict) -> None:
        if not isinstance(d, dict):
            return
        if d.get("W_base") is not None:
            self.W_base = float(d["W_base"])
        for n in self._ESCALAS:
            if isinstance(d.get(n), dict):
                getattr(self, n).restore(d[n])
