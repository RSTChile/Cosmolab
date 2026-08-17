#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Metabolismo — OrganeloMetabolismo: la economía energética del organismo
================================================================================
El órgano que faltaba (veredicto 24-jun-2026): con memoria y necesidad ya vivas, la carencia
se desplazó al METABOLISMO. La necesidad crónica saturada (Cb 8→70) era su huella: un organismo
con hambre y memoria pero SIN cómo comer, digerir ni saciarse.

RE-TRANSCRIBE capacidades PROBADAS y perdidas (no inventa):
  · v034 metabolismo de experiencias — Índice Metabólico: NUTRITIVA / TÓXICA / NEUTRA.
  · v035 preferencia alimentaria      — preferencia aprendida por impacto (qué me alimenta).
  · v069 saciedad diferencial         — saciedad ESPECÍFICA por tipo de alimento (no comer siempre lo mismo).

LAS 4 FASES (las que Alexis nombró):
  · CONSUMO     — el organismo COME su experiencia presente: si es nutritiva, REPONE energía.
  · COSTO       — actuar/abrir membrana/computar GASTA energía (se debita de una reserva FINITA).
  · DEGRADACIÓN — metabolismo basal: la energía se fuga sola con el tiempo (no hay descanso gratis).
  · REPOSICIÓN  — comer experiencias nutritivas repone; las tóxicas dañan (cuestan extra).

QUÉ ES "ALIMENTO" (canónico 2026-07-08 — alineado a Teoría CS, O-N1 / NE23):
  La experiencia se digiere en SENTIDO. Alimento = lo que el organismo YA convirtió
  (ICR / ICES endógeno del OrganoRC), no un umbral externo de "experiencia mayoritariamente
  buena" (duelo ICR>IRDE, legacy). Toxicidad ∝ disipación (IRDE/IDES), no veto de la conversión.
  Doc de equipo: DECISION_alimento_conversion_2026-07-08.md

ANTI-SHANNON: NO hay setpoint de energía ni "comer para que E=X". E EMERGE de ingesta−gasto.
  conversion multiplica magnitudes endógenas (ICR_ratio, ES); no etiqueta mensajes ni impone
  codebook de comidas. Constantes k_* / es_ref son CINÉTICA (ritmo), no significado del estímulo.

CONEXIÓN con [[memoria-organelo-ausente]]/[[act-perm-organo-conativo]]: secreta `met_nutricion`
(calidad de cada bocado) — la memoria la lee para SACIAR la necesidad. Así el lazo
necesidad→comer→saciedad por fin puede CERRARSE (a nivel diagnóstico; sin tocar el soma).
================================================================================
"""
from __future__ import annotations
import math
import os

# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel as _rel


def _c01(v, d=0.0):
    try:
        x = float(v)
    except Exception:
        return d
    return 0.0 if x != x else max(0.0, min(1.0, x))


def _num(v, d=0.0):
    try:
        x = float(v)
        return x if x == x else d
    except Exception:
        return d


COLS_MET = ["met_energia", "met_IM", "met_clase", "met_ingesta", "met_gasto", "met_balance",
            "met_costo_homeostasis",   # lo que cuesta sostener el equilibrio: 0 en calma, sube al defenderse
            "met_hambre", "met_saciedad", "met_estructura", "met_preferencia", "met_nutricion", "met_alimento_modo",
            "met_pref_top", "met_pref_top_val",
            # PALADAR VISIBLE: sin esto la saciedad especifica y el mapa de preferencias vivian
            # dentro del organelo sin salir a la fisiologia, y no habia forma de saber si al
            # organismo le gusta mas la musica que la voz o los tonos puros.
            "met_clave", "met_saciedad_tipo", "met_tipos_n", "met_paladar",
            # IMPACTO CON SIGNO: `met_nutricion` nunca es negativa, asi que quien aprenda de ella
            # solo puede aprender hacia donde IR, nunca de que APARTARSE. Esto si es signado.
            "met_impacto",
            "met_reloj_fase", "met_reloj_deriva", "met_reloj_confianza", "met_tempo_estres"]   # reloj externo GPS/PPS

CLASE = {"toxica": -1, "neutra": 0, "nutritiva": 1}

# LO QUE CUESTA MERAMENTE EXISTIR, un paso. Sale de la firma de `__init__` (era
# `basal: float = 0.003`) a nivel de módulo por dos razones, y el número NO cambia:
#
# 1. PARA QUE OTROS ÓRGANOS PUEDAN ANCLARSE A ÉL. Es la referencia canónica del proyecto
#    cuando hay que ponerle precio a algo: ya se usó así para el costo homeostático
#    («la referencia correcta es `basal`, lo que cuesta MERAMENTE EXISTIR»). Un costo que
#    se declara en otro organelo y nunca se reconcilia con esta escala es exactamente cómo
#    aparecieron los números que valían 26 veces el metabolismo entero.
# 2. PARA QUE EL GUARDIÁN LO VEA. `auditar_constantes.py` sólo reconoce nombres en
#    MAYÚSCULAS a principio de línea: los argumentos por defecto de los constructores le
#    son invisibles, y por eso toda la fisiología de varios organelos estaba fuera de la
#    auditoría. Sacarlo aquí es un número menos escondido, no uno más.
BASAL = 0.003


class OrganeloMetabolismo:
    """Economía energética: come la experiencia, paga el costo de vivir, se degrada y se repone.
    Constantes FISIOLÓGICAS declaradas (no para forzar un nivel de energía)."""

    def __init__(self,
                 E0: float = 0.6,            # energía inicial (reserva ∈ [0,1])
                 basal: float = BASAL,       # DEGRADACIÓN: metabolismo basal por paso (vivir cuesta)
                 k_ingesta: float = 0.30,    # cuánto repone CONVERTIR energía semiótica (ICES); calibrado a ES
                 k_radio: float = 0.10,      # cuánto ALIMENTA la negentropía captada por radio (radio_nutricion): comer ORDEN del espectro (Schrödinger). Ruido→0→hambre
                 k_trabajo: float = 0.008,   # COSTO de actuar/abrir membrana (act_perm) — < intake de buena comida
                 k_tiempo: float = 0.002,    # COSTO de desincronía temporal: propiocepción PPS↔ritmo interno
                 k_toxico: float = 0.04,     # daño extra de comer algo tóxico
                 tau_saciedad: float = 20.0, # cuánto dura la saciedad específica (v069)
                 ema_IM: float = 0.1,        # digestión: se metaboliza el IM SOSTENIDO, no el instantáneo (mata transitorios)
                 ema_pref: float = 0.02,
                 im_piso: float = None) -> None:  # piso del IM para nutrir: max(0, IM - im_piso). None → env ANIMA_MET_IM_PISO (def 0.0)
        self.E0 = E0; self.basal = basal; self.k_ingesta = k_ingesta; self.k_radio = k_radio; self.k_trabajo = k_trabajo
        self.k_tiempo = k_tiempo
        self.k_toxico = k_toxico
        self.tau_saciedad = tau_saciedad; self.ema_pref = ema_pref
        self.ema_IM = ema_IM
        # es_ref ELIMINADO (4-ago-2026). Valía 0,10 con su propia confesión al lado —"RC_total
        # con sonido ~0.26"— mientras el RC real de estos organismos es 0,0247: diez veces menos.
        # La referencia es ahora `self._hab_ES`, el propio valor habitual del organismo.
        # PISO DEL IM (indulgencia nutricional, 2-jul-2026): la nutrición usa max(0, IM - im_piso). Con
        # im_piso=0.0 (canónico) sólo nutre lo que CONVIERTE en sentido (ICR>IRDE). Un piso NEGATIVO da
        # margen a organismos jóvenes/mínimos donde IRDE domina de fábrica (mundo aún poco convertible):
        # les deja extraer algo de sustento de experiencias apenas-riesgosas sin volver nutritivo lo tóxico.
        # Knob de experimento vía env ANIMA_MET_IM_PISO; por defecto 0.0 = comportamiento previo intacto.
        self.im_piso = float(os.environ.get("ANIMA_MET_IM_PISO", "0.0")) if im_piso is None else float(im_piso)
        # MODO DE ALIMENTO (canónico 2026-07-08, alineado a Teoría CS NE23 / O-N1):
        #   "conversion" (DEFAULT) — nutrición = ICR_ratio·es_norm
        #       Alimento = lo que el organismo YA convirtió en sentido (ICES/ICR endógeno del OrganoRC).
        #       NO es Shannon: no hay setpoint de E, no hay umbral de "mensaje válido", no hay codebook
        #       de comidas. Solo se multiplica lo emergente (ICR_ratio, ES) por constantes CINÉTICAS
        #       (k_ingesta, es_ref) que escalan ritmo, no significan el estímulo.
        #       Toxicidad ∝ IRDE_ratio·es_norm (disipar cuesta; no veta la conversión).
        #   "duelo" (histórico/experimental) — nutrición = max(0, IM - im_piso)·es_norm
        #       Exige ICR_ratio > IRDE_ratio: regla EXTERNA de "experiencia mayoritariamente buena".
        #       Eso SÍ se acerca a imposición de validez (más Shannon que conversion).
        # Env: ANIMA_MET_ALIMENTO=conversion|duelo  (default conversion)
        _alim = (os.environ.get("ANIMA_MET_ALIMENTO", "conversion") or "conversion").strip().lower()
        self.alimento_modo = "duelo" if _alim in ("duelo", "majority", "im", "legacy") else "conversion"
        # ── REPARTO NEUTRO (ANIMA_REPARTO_NEUTRO=1) ─────────────────────────────────────────
        # Las constantes de fábrica no son simétricas: k_ingesta=0.30 premia cada unidad
        # CONVERTIDA 7,5 veces más de lo que k_toxico=0.04 castiga cada unidad DISIPADA. Eso
        # pone el punto de equilibrio en ICR≈11,8% en vez de 50%, o sea que el diseño ya venía
        # decidiendo que la experiencia típica es veneno. No era una decisión teórica: era el
        # parche que compensaba el sesgo de entrada del reparto en VST_RC_A (soporte 0,555 vs
        # vulnerabilidad 0,845, medidos sobre 58.290 pasos reales).
        #
        # Con el interruptor, ambas constantes se igualan y el balance semiótico pasa a ser
        #     ingesta − toxicidad = k · (ICR − IRDE) · es_norm = k · IM · es_norm
        # que es CERO exactamente cuando ICR = IRDE = 50%. Un sonido no llega con signo: sólo
        # al procesarlo se sabe si alimentó o dio indigestión.
        #
        # ATENCIÓN: se enciende JUNTO con el de VST_RC_A (misma variable de entorno a propósito).
        # Igualar aquí sin corregir allá deja al organismo PEOR que antes — quita el parche y
        # deja el sesgo.
        if (os.environ.get("ANIMA_REPARTO_NEUTRO", "") or "").strip().lower() in ("1", "true", "yes", "on"):
            self.k_toxico = self.k_ingesta
        self.reset()

    def reset(self) -> None:
        self.E = self.E0
        self.saciedad = {}      # saciedad específica por tipo de alimento (clave)  — v069
        self.preferencia = {}   # preferencia aprendida por tipo                    — v035
        self._IM_ema = 0.0      # IM SOSTENIDO (digestión: filtra transitorios del campo)
        self._bal_ema = 0.0     # BALANCE SOSTENIDO
        self._escala_RC = Escala()   # lo que suele recibir: gobierna el letargo, no la ingesta: el juez de si un bocado alimenta o intoxica
        self._sab_mu = 0.5     # centro movil del ORDEN: que es 'normal' para este organismo
        self._sab_sd = 0.15    # dispersion tipica: a partir de cuanto se sale de lo normal
        self._ES_ema = 0.0      # energía semiótica sostenida (RC_total suavizado)

    # ----- PERSISTENCIA: las reservas y lo aprendido sobreviven al apagón (incremento 1) -----
    def snapshot(self) -> dict:
        """Estado metabólico a guardar: reservas de energía (E), saciedades específicas (v069) y
        preferencias aprendidas por alimento (v035). Las constantes (basal, k_*) vienen del genoma."""
        return {"E": self.E, "saciedad": self.saciedad, "preferencia": self.preferencia,
                "_IM_ema": self._IM_ema, "_ES_ema": self._ES_ema,
                # el juicio sostenido sobre la comida sobrevive al reinicio: sin esto cada arranque
                # volvia a partir de "neutro" y se perdia lo aprendido en la sesion.
                "_bal_ema": self._bal_ema, "_escala_RC": self._escala_RC.snapshot(),
                # el paladar aprendido: sin esto el organismo reaprende que es "mucho orden"
                "_sab_mu": self._sab_mu, "_sab_sd": self._sab_sd}

    def restore(self, d: dict) -> None:
        if not d:
            return
        self.E = float(d.get("E", self.E0))
        self.saciedad = d.get("saciedad", {}) or {}
        self.preferencia = d.get("preferencia", {}) or {}
        self._IM_ema = float(d.get("_IM_ema", 0.0) or 0.0)
        self._bal_ema = float(d.get("_bal_ema", 0.0) or 0.0)
        self._escala_RC.restore(d.get("_escala_RC"))
        self._sab_mu = float(d.get("_sab_mu", 0.5) or 0.5)
        self._sab_sd = max(0.02, float(d.get("_sab_sd", 0.15) or 0.15))
        self._ES_ema = float(d.get("_ES_ema", 0.0) or 0.0)

    def _clave_alimento(self, lat: float, sabor: float):
        """Tipo de 'alimento': DE DÓNDE viene × CUÁNTO ORDEN trae.

        El eje del sabor era `IM = ICR − IRDE` contra un umbral fijo de ±0,15. Medido el 3-ago en
        Abraxas, IM vale ≈−0,69 en TODOS los pasos, así que `sb` daba −1 siempre: el eje estaba
        muerto y música, voz y tonos puros caían en el mismo cubo si llegaban por el mismo lado.
        Cualquier "preferencia por la música" medida con esta clave era un artefacto.

        Ahora el sabor es la ESTRUCTURA que trae el bocado — la medida de localización de la
        membrana, que sí varía (0,00–0,96). Ordena los alimentos como el organismo los recibe:
        un tono puro concentra la vibración en pocos modos (mucho orden), la música y la voz
        reparten en más (orden medio), el ruido la reparte en todos (nada de orden).

        El umbral NO se elige: es la propia historia del organismo. Se comparan el bocado actual
        contra su media móvil y su dispersión típica, así que "mucho" y "poco" significan mucho o
        poco PARA ESTE ORGANISMO en este ambiente. Un organismo que sólo ha oído ruido tiene otro
        cero que uno criado entre música, y ninguno de los dos hereda una constante nuestra.
        """
        lb = -1 if lat < -0.15 else (1 if lat > 0.15 else 0)
        dev = sabor - self._sab_mu
        sb = 1 if dev > self._sab_sd else (-1 if dev < -self._sab_sd else 0)
        return (lb, sb)

    def _aprender_sabor(self, sabor: float) -> None:
        """Actualiza el centro y la dispersión con que el organismo juzga cuánto orden es 'mucho'."""
        self._sab_mu += self.ema_pref * (sabor - self._sab_mu)
        self._sab_sd += self.ema_pref * (abs(sabor - self._sab_mu) - self._sab_sd)
        self._sab_sd = max(0.02, self._sab_sd)   # nunca cero: si no, todo bocado seria "extremo"

    def actualizar(self, fila: dict, dt: float = 0.1) -> dict:
        icr_r = _c01(fila.get("ICR_ratio")); irde_r = _c01(fila.get("IRDE_ratio"))
        act_perm = _c01(fila.get("act_perm"))
        # ESTRUCTURA del alimento: el ORDEN reconocido en lo que entra (membrana sensorial).
        # 1 = sonido con forma (ritmo, estructura); 0 = ruido blanco, estática, caos.
        estructura_alimento = _c01(_num(fila.get("estructura", 1.0), 1.0))
        lat = _num(fila.get("lateralidad"))
        # ENERGÍA SEMIÓTICA en acto (RC ≡ ES, C-N2/NE23.1): RC_total = ES = ICES + IDES. En silencio ~0:
        # no hay nada que convertir ni comer. La absorción se rige por ESTO (estar EN ACTO), NO por A (calma).
        ES = max(0.0, _num(fila.get("RC_total")))
        self._ES_ema = (1.0 - self.ema_IM) * self._ES_ema + self.ema_IM * ES      # ES sostenida (digestión)
        # ── EL RECURSO ES ABSOLUTO; EL COSTO SE ADAPTA (4-ago-2026) ──────────────────
        # Ayer medí el recurso contra el propio habitual del organismo, y el ayuno del 4-ago
        # demostró que estaba mal: corté la entrada, el organismo murió en 32 segundos —bien— y
        # DESPUÉS resucitó. Con la entrada muda (energia_L = 0,00000 en 457 pasos) comió en el 54%
        # de ellos y su reserva subió de 0,26 a 0,58, porque su "habitual" decayó hasta el ruido de
        # fondo del campo —RC ≈ 0,0017, que nunca es exactamente cero— y entonces cualquier miseria
        # volvió a leerse como normal. Inmortalidad por adaptación.
        #
        # La naturaleza no funciona así: un animal en escasez adapta su CONDUCTA (busca más) y su
        # GASTO (letargo), pero NO extrae más calorías del mismo alimento — su digestión no mejora
        # por tener hambre. Lo que Alexis pidió literalmente era «que el costo se mida contra el
        # recurso», y yo lo implementé al revés.
        #
        # RC_total ya es una fracción acotada en [0,1] (verificado: 0 fuera de rango en 3.226
        # pasos), así que el recurso no necesita ninguna referencia. `es_ref` nunca hizo falta.
        es_norm = _c01(self._ES_ema)                 # ¿hay energía semiótica disponible?

        # ── ÍNDICE METABÓLICO (v034): ¿esta experiencia ALIMENTA o INTOXICA? ──────────────────
        # nutritiva = ICR convierte en sentido; tóxica = IRDE/riesgo domina. IM ∈ [−1,1].
        # DIGESTIÓN: se metaboliza el IM SOSTENIDO (EMA), no el instantáneo — un transitorio del campo no
        # es comida ni veneno; sólo lo que PERSISTE alimenta o intoxica. (También limpia la señal en movimiento.)
        IM_inst = icr_r - irde_r
        self._IM_ema = (1.0 - self.ema_IM) * self._IM_ema + self.ema_IM * IM_inst
        IM = self._IM_ema
        # La CLASE ya no se decide aquí: se decidía ANTES de digerir, comparando IM contra dos
        # umbrales fijos (±0,10) en una escala que no es la suya. Con ICR≈0,16 el IM vale ≈−0,69
        # SIEMPRE, así que el organismo calificaba de veneno cada bocado mientras ganaba energía
        # con él. Un sonido no trae carga previa: sólo al procesarlo se sabe si alimentó o
        # indigestó. El veredicto se emite más abajo, después de cerrar el balance.
        # PALADAR UNIFICADO: la clave lleva MODALIDAD (mundo / voz_otro / …). El integrador puede fijar
        # met_modalidad="voz_otro" cuando lo que suena es la voz del par; por defecto "mundo".
        modalidad = str(fila.get("met_modalidad") or "mundo")
        # El SABOR del bocado es el orden que trae (estructura), no el saldo ICR/IRDE.
        self._aprender_sabor(estructura_alimento)
        clave = (modalidad,) + self._clave_alimento(lat, estructura_alimento)

        # ── SACIEDAD diferencial (v069): específica por tipo; decae sola ─────────────────────
        d_sac = math.exp(-dt / self.tau_saciedad)
        for k in self.saciedad:
            self.saciedad[k] *= d_sac
        sac = self.saciedad.get(clave, 0.0)

        # ── CONSUMO = CONVERSIÓN DE ENERGÍA SEMIÓTICA EN SENTIDO (ICES) ──────────────────────
        # COMER es estar EN ACTO convirtiendo (enérgeia), NO estar calmo (por eso ya NO va A).
        # La saciedad reduce el rendimiento pero no lo anula (sac_max): se puede vivir de un alimento.
        # ── SACIEDAD = LA RESERVA (corregido 3-ago-2026) ────────────────────────────────
        # No hay nada que fijar. Estar saciado no es un contador aparte: es TENER RESERVA.
        # Tengo hambre cuando el gasto ya consumió lo que comí; quedo satisfecho cuando como
        # más de lo que gasto y me sobra para un rato. Eso es exactamente `self.E`, que ya
        # integra (ingesta − gasto) — y de hecho el hambre ya se calculaba como 1 − E.
        #
        # Lo anterior era un mecanismo paralelo que duplicaba lo mismo con cuatro constantes
        # puestas a mano (tau_saciedad, base_saciedad, sac_max) y una regla binaria: subía a
        # tasa fija con `if ingesta > 0`, sin mirar CUÁNTO se comía. Una miga saciaba igual
        # que un banquete. Medido en Abraxas el 3-ago: saciedad 0,907 en un organismo que
        # jamás había comido, recortando su ingesta al 40% y volviéndolo inviable.
        #
        # Ahora el lazo se cierra solo y sin setpoint: lleno ⇒ ingiere poco; vacío ⇒ ingiere
        # a pleno rendimiento. Homeostasis emergente, no un objetivo impuesto.
        sac_ef = _c01(self.E)
        if self.alimento_modo == "conversion":
            # TEORÍA CS: alimento = lo convertido en sentido (ICR/ICES), no el duelo ICR>IRDE.
            # nutrición ∝ fracción convertida × ES disponible. Si ICR_ratio>0 y hay ES, hay comida
            # aunque IRDE_ratio sea mayor (la disipación se paga como toxicidad, no como veto).
            # EL ORDEN ENTRA UNA SOLA VEZ (3-ago-2026). `icr_r` YA viene filtrado por la estructura:
            # VST_RC_A reparte el bocado con `base_icr = RC · soporte · peso_ICR · estructura`, y lo
            # que a la estructura le falta engorda `base_irde`. Ahí es donde el orden decide qué se
            # vuelve sentido y qué se disipa. Volver a multiplicar aquí por `estructura` cobraba el
            # peaje DOS VECES al orden y una sola al desorden: medido el 3-ago en Abraxas, ICR_ratio
            # cayó de 0,341 a 0,132 y el organismo colapsó a energía 0 en tres minutos. La firma del
            # doble cobro era que `estructura` oscilaba entre 0,37 y 0,68 mientras ICR_ratio no se
            # movía (0,132 → 0,121): encontrar mejor comida no mejoraba la conversión.
            #
            # DOS APETITOS, DOS MEMORIAS (3-ago-2026). `sac_ef` es la reserva: cuánta hambre tengo
            # EN GENERAL. `sac` es la saciedad de ESTE tipo de alimento, que viene de los pasos
            # anteriores — memoria corta de lo que acabo de comer. Se multiplican porque son
            # independientes: puedo tener hambre y aun así no querer más de lo mismo. Es la
            # saciedad sensorial específica de v069, que empuja a VARIAR la dieta sin que nadie
            # lo ordene: comer mucho de un tipo lo vuelve menos rendidor y otro tipo pasa a rendir
            # más. Ninguna constante nueva: `sac` se mide contra el propio gasto (ver más abajo).
            nutricion = max(0.0, icr_r) * es_norm * (1.0 - sac_ef) * (1.0 - _c01(sac))
        else:
            # DUELO (histórico): max(0, IM - im_piso)·es_norm — sólo nutre si ICR_ratio>IRDE_ratio
            # (o por encima del piso experimental).
            nutricion = max(0.0, IM - self.im_piso) * es_norm * (1.0 - sac_ef)
        foto_aporte = max(0.0, _num(fila.get("foto_aporte")))
        # COMER ORDEN: la negentropía captada por radio (radio_nutricion = enganchado × estructura
        # coherente) alimenta como la luz. Ruido → estructura≈0 → radio_aporte≈0 → el organismo
        # pasa hambre por sintonizar ruido. Es el anti-Shannon hecho metabolismo (falsable).
        radio_aporte = self.k_radio * max(0.0, _num(fila.get("radio_nutricion")))
        ingesta = self.k_ingesta * nutricion + foto_aporte + radio_aporte
        # ── SACIEDAD PROPORCIONAL A LO COMIDO (corregido 3-ago-2026) ────────────────────
        # ANTES: `if ingesta > 0` con una tasa fija. La saciedad no dependía de CUÁNTO se
        # comía, sino de que se comiera ALGO: una miga saciaba igual que un banquete. Con
        # base_saciedad=0,03 el organismo subía ~5% por paso rozando migajas de 0,0045 y se
        # clavaba en 0,907 sin haber comido nunca de verdad — un organismo permanentemente
        # saciado y permanentemente hambriento a la vez. Medido el 3-ago en Abraxas.
        #
        # AHORA: la saciedad es la fracción de una RACIÓN que el organismo lleva ingerida,
        # promediada con la misma constante de tiempo que ya tenía (tau_saciedad). Y la
        # ración no la elegimos: es LO QUE LE CUESTA VIVIR, su propio gasto del paso
        # anterior. Comer lo que se gasta deja saciedad ≈ 1 bocado completo; comer la mitad
        # deja la mitad; no comer no sacia nada —sin necesidad de ningún `if`—. El estado
        # estacionario es exactamente ingesta/gasto, así que la saciedad pasa a MEDIR la
        # suficiencia de la dieta en vez de contar visitas a la mesa.
        #
        # `sac` ya viene multiplicado por d_sac arriba, así que sumar el término nuevo
        # ponderado por (1−d_sac) completa la media exponencial.

        # ── COSTO + DEGRADACIÓN: vivir y actuar gastan; DISIPAR energía semiótica (IDES) daña ──
        if self.alimento_modo == "conversion":
            # ── INDIGESTIÓN SOBRE EL REPARTO, CON COMPUERTA DE APETITO (3-ago-2026) ─────
            # El orden entra UNA SOLA VEZ, en VST_RC_A, que parte el bocado en ICR (lo que se
            # vuelve sentido) e IRDE (lo que se disipa). El metabolismo no vuelve a juzgar la
            # estructura: consume ese reparto. Se alimenta de ICR y paga por IRDE.
            #
            # Probé la alternativa —cobrar `k_ingesta · (1 − estructura)` como porción sin forma
            # del bocado— y colapsó el organismo en tres minutos: al aplicar la estructura aquí y
            # en RC_A a la vez, el orden pagaba peaje dos veces y el desorden una sola.
            #
            # Lo único que se conserva de aquel intento es la COMPUERTA DEL APETITO: la toxicidad
            # lleva el mismo factor (1 − sac_ef) que la ingesta, porque no se puede intoxicar con
            # lo que no se comió. Sin ella, un organismo lleno seguía envenenándose sin probar
            # bocado — la ingesta tendía a cero y la toxicidad se quedaba entera.
            #
            # `k_toxico = 0,04` queda como estaba. La auditoría del 2-ago lo marcó por no estar
            # derivado, y sigue sin estarlo; pero ahora se entiende POR QUÉ es pequeño: la
            # estructura ya hizo su trabajo aguas arriba, y esto sólo cobra la disipación
            # residual. Derivarlo es tarea aparte, no de este cambio.
            # ── FALLAR CUESTA EL INTENTO, NO UNA SEGUNDA CUENTA (4-ago-2026) ─────────────
            # Medido sobre 43.607 pasos: esta multa era el 27% del gasto total del organismo.
            # Cobraba `k_toxico · IRDE_ratio`, es decir un cargo proporcional a lo que NO supo
            # convertir. Con RC = ICR + IRDE siendo una ley de conservacion, eso paga dos veces
            # por el mismo fracaso: no recibe esa energia Y ademas se le cobra por no haberla
            # recibido.
            #
            # La naturaleza no cobra asi. Un arbol no paga por el 92% de luz que no aprovecha:
            # simplemente no la recibe. Un guepardo falla 83 de cada 100 cacerias y solo paga la
            # carrera, no una penalizacion por el fallo. El costo del intento ya esta cobrado en
            # `k_trabajo * act_perm`; esto era el doble cobro, en el ultimo sitio donde quedaba.
            #
            # Indigestarse sigue siendo posible: lo que entra sin forma no nutre (la estructura ya
            # reparte ICR/IRDE en VST_RC_A). Lo que desaparece es la MULTA por no convertir.
            toxicidad = 0.0
        else:
            toxicidad = self.k_toxico * max(0.0, -IM) * es_norm  # disipación de la energía presente (silencio no intoxica)
        # COSTE DE ACUÑAR: si el organismo creó una palabra propia con el aparato fonador, paga aquí su gasto
        # energético (más caro que reutilizar el banco). Lo inyecta el bucle vía met_costo_extra (una vez por
        # creación). Así "crear cuesta de verdad": sólo se acuña cuando hay energía y la necesidad lo amerita.
        costo_voz = max(0.0, _num(fila.get("met_costo_extra")))
        # ── RELOJ GPS/PPS COMO PROPIOCEPCIÓN METABÓLICA ─────────────────────────────
        # OrganoLocalizacion compara el tempo interno con el PPS externo y secreta loc_pps_deriva:
        # 0 = sincronía; + = E corre adelantado al cielo; - = atrasado. El metabolismo no usa la
        # hora como mandato, sino como costo suave de desincronía cuando el reloj externo es confiable.
        reloj_fase = _num(fila.get("loc_reloj_fase"), 0.0)
        reloj_deriva = _num(fila.get("loc_pps_deriva"), 0.0)
        reloj_confianza = 1.0 if fila.get("gps_pps_count") is not None else 0.0
        tempo_estres = min(1.0, abs(reloj_deriva)) * reloj_confianza
        costo_tiempo = self.k_tiempo * tempo_estres

        # ── LETARGO: gastar menos cuando hay menos (4-ago-2026) ─────────────────────
        # Lo que sí se adapta a la escasez. `basal` NO entra en el letargo: es el costo
        # irreducible de existir, y es lo que mantiene la muerte posible — un animal en letargo
        # profundo sigue consumiendo y sigue muriendo si el ayuno se alarga. Lo que se reduce es
        # lo que el organismo GASTA EN HACER: actuar, abrir la membrana, cantar.
        #
        # El factor se mide contra lo que ESTE organismo suele recibir, que es la comparación
        # legítima aquí: no infla lo que entra, modula lo que se hace con ello. Vale 1 cuando el
        # recurso es el habitual, baja cuando escasea y sube cuando abunda.
        self._escala_RC.observar(_num(fila.get("RC_total")))
        _letargo = 2.0 * _rel(_num(fila.get("RC_total")), self._escala_RC) if self._escala_RC.madura else 1.0
        # CORREGIDO 5-ago-2026: aquí puse `max(0.05, min(1.5, ...))`, dos cotas sin origen —
        # exactamente lo que esta auditoría persigue, cometido por mí. No hacen falta: `rel`
        # ya devuelve [0,1] por construcción, así que el factor vive en [0,2] sin recortes.
        # Que pueda llegar a 0 es correcto y es lo que mantiene la muerte posible: `basal` no
        # entra en el letargo, así que el gasto irreducible sigue drenando aunque lo demás se
        # apague. Es lo que hace un animal en letargo profundo.
        # ── LO QUE CUESTA SOSTENER EL EQUILIBRIO (6-ago-2026) ───────────────────────────
        # No puede costar lo mismo estar en calma que defenderse de una perturbación: con frío
        # se queman más calorías para sostener la temperatura. Hasta hoy el organelo de
        # homeostasis cobraba un `costo_base` fijo y proporcional a la MASA (Kleiber), ciego a
        # cuánto estaba trabajando — flotar y pelear salían igual.
        #
        # EL ESFUERZO es el contra-empuje que la regulación aplica de verdad: en el Daisyworld
        # `x` se mueve por (estrés − κ·b + κ·w), así que en calma las dos margaritas se
        # equilibran y el neto es ~0, y bajo perturbación una desplaza a la otra. Por eso vale
        # |b − w|, que el organelo ya publica como `x_interna_esfuerzo`.
        #
        # LA ESCALA SALE DEL PROPIO ORGANISMO, y hubo que medirla antes de cobrar nada: el
        # `costo_base = 1.0` del organelo NO está en las unidades de este metabolismo — MEDIDO
        # en vivo, cobrarlo tal cual daba 0,2714 contra un `met_gasto` de 0,0105, veintiséis
        # veces el metabolismo entero. La referencia correcta es `basal`, lo que cuesta
        # MERAMENTE EXISTIR: defensa máxima cuesta otro tanto que estar vivo, y calma no cuesta
        # nada extra. Ninguna constante nueva.
        #
        # Y SÍ ENTRA EN EL LETARGO, a diferencia de `basal`: un animal sin reservas deja de
        # termorregular antes que de existir. Eso es el torpor, y es la conducta correcta —
        # defenderse es de lo primero que se apaga cuando no hay con qué.
        costo_homeo = self.basal * max(0.0, min(1.0, _num(fila.get("x_interna_esfuerzo"), 0.0)))
        gasto = (self.basal
                 + (self.k_trabajo * act_perm + costo_voz + costo_homeo + costo_tiempo) * _letargo
                 + toxicidad)
        balance = ingesta - gasto
        self.E = max(0.0, min(1.0, self.E + balance))
        hambre = 1.0 - self.E

        # ── EL VEREDICTO SALE DE LA DIGESTIÓN (3-ago-2026) ──────────────────────────────
        # Alimenta lo que deja al organismo con más reserva de la que tenía; intoxica lo que lo
        # deja con menos. No hay nada que declarar de antemano ni ninguna escala ajena: el juez
        # es el propio balance, que ya integra lo que entró, lo que costó convertirlo y lo que
        # costó vivir el paso.
        #
        # Se juzga el balance SOSTENIDO, no el instantáneo, por la misma razón por la que el IM
        # se promediaba: un transitorio del campo no es comida ni veneno. Reusa `ema_IM`, que es
        # justamente la constante de tiempo de la digestión.
        #
        # La banda neutra tampoco se elige: es `basal`, lo que cuesta MERAMENTE EXISTIR. Un bocado
        # que mueve la reserva menos que respirar no se distingue de no haber comido. Así los dos
        # umbrales puestos a mano (±0,10 sobre IM) desaparecen y quedan reemplazados por una
        # magnitud del propio organismo. `umbral_nut`/`umbral_tox` quedan sin uso.
        # ── MEMORIA CORTA: DE QUÉ ESTOY LLENO (restaurado 3-ago-2026) ───────────────────
        # Al reescribir la saciedad como reserva perdí la línea que alimentaba `self.saciedad`:
        # el diccionario sólo decaía y nunca se asignaba, así que la saciedad ESPECÍFICA (v069)
        # había quedado como estado muerto y el organismo no recordaba de qué venía comiendo.
        #
        # Vuelve, pero medida contra el propio organismo en vez de subir a tasa fija: la RACIÓN
        # es lo que costó vivir este paso, así que la saciedad de un tipo es qué fracción de una
        # ración llevo comida DE ESE TIPO. Comer lo que se gasta llena un bocado; comer la mitad,
        # media; no comer no llena nada — sin necesidad de ningún `if ingesta > 0`, que era lo que
        # hacía que una miga saciara igual que un banquete. El decaimiento (tau_saciedad) ya se
        # aplicó arriba; esto completa la media exponencial con el término del paso.
        racion = max(0.0, min(1.0, ingesta / max(gasto, self.basal)))
        self.saciedad[clave] = max(0.0, min(1.0, self.saciedad.get(clave, 0.0) + (1.0 - d_sac) * racion))

        self._bal_ema = (1.0 - self.ema_IM) * self._bal_ema + self.ema_IM * balance
        clase = ("nutritiva" if self._bal_ema > self.basal
                 else "toxica" if self._bal_ema < -self.basal
                 else "neutra")

        # ── PALADAR UNIFICADO (v035 + radio): UN mapa de preferencia sobre TODAS las modalidades ──
        # Cada modalidad aporta su PROPIA nutrición: semiosis por IM; radio por su estructura (negentropía
        # espectral). Aprender qué me alimentó mejor, venga de donde venga → un solo gusto.
        for k in self.preferencia:
            self.preferencia[k] *= (1.0 - 0.2 * self.ema_pref)   # olvido competitivo suave
        # APRENDER DE LO QUE LA COMIDA HIZO, NO DE LO QUE PROMETÍA (3-ago-2026)
        # Antes: `pref += ema_pref · (IM − pref)`. Con ICR≈0,16 el IM vale ≈−0,69 en TODOS los
        # pasos, así que todos los alimentos convergían al mismo número: el organismo no aprendía
        # que algo le cayó mal, aprendía que todo le cae mal por igual — que es no aprender nada.
        #
        # Ahora el maestro es el IMPACTO REAL del bocado: qué fracción de lo que costó vivir ese
        # paso alcanzó a pagar. +1 = se pagó a sí mismo con creces; 0 = salió a mano; −1 = no
        # aportó nada y hubo que poner la reserva. La referencia es el gasto DEL PROPIO ORGANISMO
        # en ese paso, no una escala externa; por eso no hace falta ninguna constante nueva.
        impacto = max(-1.0, min(1.0, balance / max(gasto, self.basal)))
        pref = self.preferencia.get(clave, 0.0)
        pref += self.ema_pref * (impacto - pref)
        self.preferencia[clave] = max(-1.0, min(1.0, pref))
        # RADIO entra al MISMO paladar: sabor = su frecuencia (MHz); su 'IM' = la negentropía captada
        # (radio_nutricion), escalada a rango comparable con el IM semiótico.
        radio_nut = max(0.0, _num(fila.get("radio_nutricion")))
        if radio_nut > 0.0:
            crad = ("radio", int(round(_num(fila.get("radio_freq_dom_hz")) / 1e6)))
            pr = self.preferencia.get(crad, 0.0)
            self.preferencia[crad] = max(-1.0, min(1.0, pr + self.ema_pref * (min(1.0, radio_nut * 5.0) - pr)))
        # el FAVORITO del paladar, entre todas las modalidades (lo que este organismo más gusta de comer)
        if self.preferencia:
            top_k = max(self.preferencia, key=self.preferencia.get)
            top_v = self.preferencia[top_k]
        else:
            top_k, top_v = ("-",), 0.0

        return {
            "met_energia": round(self.E, 4), "met_IM": round(IM, 4), "met_clase": CLASE[clase],
            "met_ingesta": round(ingesta, 5), "met_gasto": round(gasto, 5), "met_balance": round(balance, 5),
            "met_costo_homeostasis": round(costo_homeo * _letargo, 6),   # ya con el torpor aplicado
            "met_hambre": round(hambre, 4), "met_saciedad": round(sac_ef, 4),
            # ORDEN del alimento tal como lo ve el metabolismo. Ya no entra en el cálculo —lo
            # aplica VST_RC_A al repartir ICR/IRDE— pero se publica para poder VERLO: si sube
            # la estructura y no sube ICR_ratio, el orden se está cobrando dos veces.
            "met_estructura": round(estructura_alimento, 4),
            "met_preferencia": round(self.preferencia[clave], 4), "met_nutricion": round(nutricion, 4),
            "met_alimento_modo": self.alimento_modo,  # "conversion" (canónico) | "duelo" (legacy)
            "met_pref_top": "|".join(str(x) for x in top_k), "met_pref_top_val": round(top_v, 4),
            # QUE estoy comiendo ahora y cuan lleno estoy DE ESO (no de todo).
            "met_impacto": round(impacto, 4),
            "met_clave": "|".join(str(x) for x in clave),
            "met_saciedad_tipo": round(_c01(self.saciedad.get(clave, 0.0)), 4),
            "met_tipos_n": len(self.preferencia),
            # el paladar entero, ordenado de lo que mas gusta a lo que menos: tipo=valor;tipo=valor
            "met_paladar": ";".join(
                "%s=%.3f" % ("|".join(str(x) for x in k), v)
                for k, v in sorted(self.preferencia.items(), key=lambda kv: -kv[1])[:8]),
            "met_reloj_fase": round(reloj_fase, 4),
            "met_reloj_deriva": round(reloj_deriva, 6),
            "met_reloj_confianza": round(reloj_confianza, 4),
            "met_tempo_estres": round(tempo_estres, 4),
        }


if __name__ == "__main__":
    met = OrganeloMetabolismo()
    nutr = {"A_sys_env": 0.8, "ICR_ratio": 0.8, "IRDE_ratio": 0.2, "act_perm": 0.3, "lateralidad": 0.0}
    tox = {"A_sys_env": 0.3, "ICR_ratio": 0.2, "IRDE_ratio": 0.8, "act_perm": 0.6, "lateralidad": 0.0}
    for _ in range(60):
        rn = met.actualizar(nutr)
    print("tras comer NUTRITIVO:", rn)
    for _ in range(60):
        rt = met.actualizar(tox)
    print("tras comer TÓXICO:   ", rt)
