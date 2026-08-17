#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Memoria — OrganeloMemoria: la historia interna del organismo (un órgano, 6 capas)
================================================================================
"Sin memoria no hay futuro: no se recuerda el pasado, sólo hay devenir" (Alexis, 24-jun-2026).

RE-TRANSCRIBE capacidades PROBADAS y perdidas del linaje (no inventa): cada capa hereda de
un experimento validado. Encapsulado: percibir→metabolizar→secretar sobre el milieu/fila.

CAPAS (procedencia):
  1. BUFFER CORTO      — deque de estados recientes (RegistroRepresentaciones v180).
  2. PERSISTENCIA      — mantiene el último estímulo en ausencia; confianza exp(−t/τ),
                         τ CRECE con la vida vivida (MemoriaAusencia v180/V155).
  3. VALOR/AFECTIVA    — valencia[estado] con DOBLE escala: corto (τ≈8s) y largo (τ≈240s)
                         = hábito vs IDENTIDAD (ValenciaLocal v180 + v035 + valor_estructural v051).
  4. EPISÓDICA         — eventos {t,clave,tipo,intensidad,valencia}; recall por similitud CON
                         COSTO; + el recall explícito que faltaba (MemoriaEpisódica v180, 0/50).
  5. ESTRUCTURAL       — LEE (no reconstruye) la memoria implícita del soma (W hebbiana +
                         Phi_int_historia, v074/v80h) → familiaridad/novedad consultables.
  6. RELACIONAL        — confianza hacia el OTRO (Hebb social, v182); plena sólo en díada A+B.

CONEXIÓN Cb→act_perm: necesidad = act_perm·(1+Cb_norm), con Cb = presion_desacople (integrador
con fuga YA vivo). Convierte la DISPOSICIÓN instantánea (act_perm) en NECESIDAD con historia,
y añade SACIEDAD/REFRACTARIEDAD tras el re-acople (lo que a act_perm le faltaba).

ANTI-SHANNON: nada de setpoints ni "recordar para que pase X". Saliencia, consolidación y olvido
EMERGEN de sorpresa·valencia·energía. BIOENERGÉTICA: buffers acotados; no se graba sin energía;
escalas τ fisiológicas DECLARADAS. NO cierra el lazo conductual (no toca soma ni orientación).
================================================================================
"""
from __future__ import annotations
from collections import deque
import math

# ESCALA COMPARTIDA (4-ago-2026). La auditoría de 168 constantes ilegítimas dejó escrito que
# el patrón "comparar contra lo habitual" se escriba UNA vez y se importe, no 168. Aquí se
# importa: `rel(x, esc)` vale 0,5 cuando x es lo de siempre para ESTE organismo, sin ningún
# número elegido a mano; `clasificar` devuelve "indeterminado" mientras no haya historia, que
# es la política de arranque única del proyecto (abstenerse, no inventar).
# `escala` vive en celula_madre/; esto permite importar el organelo suelto (pruebas y smokes)
# además de dentro del organismo. Unificado el 5-ago-2026: la revisión encontró CUATRO
# variantes del mismo arranque, que es el problema contra el que existe el módulo compartido.
import os as _os, sys as _sys
_RAIZ_CM = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _RAIZ_CM not in _sys.path:
    _sys.path.insert(0, _RAIZ_CM)
from escala import Escala, rel as _rel, clasificar as _clasificar, NEUTRO


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


COLS_MEM = ["mem_familiaridad", "mem_novedad", "mem_carga_estructural", "mem_valencia_estado",
            "mem_persistencia", "mem_recall", "mem_recall_tipo", "mem_recall_costo",
            "mem_episodios_n", "Cb_integrado", "necesidad", "necesidad_efectiva",
            "mem_saciedad", "mem_relacional_confianza"]

# Códigos de tipo de episodio (numéricos para CSV/gráficos)
TIPO = {"ninguno": 0, "amenaza": 1, "logro": 2, "novedad": 3, "neutro": 4}


class OrganeloMemoria:
    """La historia interna del organismo. Driven a nivel fila por actualizar(); internamente
    respeta percibir→metabolizar→secretar. Constantes FISIOLÓGICAS declaradas (no para forzar)."""

    _NOMBRES_ESCALA = ("esc_div", "esc_dA", "esc_eR", "esc_lat", "esc_irde",
                       "esc_H", "esc_nov", "esc_sal", "esc_fat", "esc_Cb")

    def __init__(self,
                 buffer_n: int = 200,            # ventana de corto plazo (v180 deque)
                 tau_val_corto: float = 8.0,     # decay del valor de corto plazo (hábito)
                 tau_val_largo: float = 240.0,   # decay del valor de largo plazo (identidad)
                 eta_corto: float = 0.10,         # aprendizaje rápido del valor
                 eta_largo: float = 0.02,         # consolidación lenta corto→largo
                 max_episodios: int = 64,        # capacidad episódica (olvido selectivo si se llena)
                 tau_persist_base: float = 3.0,  # permanencia base (s); crece con la vida
                 k_persist_vida: float = 0.5,    # cuánto alarga τ la historia vivida
                 tau_saciedad: float = 6.0,      # cuánto dura la refractariedad tras saciarse
                 olvido_relacional: float = 0.99) -> None:
        # RETIRADOS DE LA FIRMA (4-ago-2026), porque decidían qué le pasa al organismo contra
        # una escala que nadie midió. Cada uno tiene su medición en el sitio donde se usaba:
        #   · umbral_saliencia = 0.40  → ahora: destacar sobre la propia saliencia habitual.
        #   · escala_dA        = 0.05  → ahora: la escala del propio |ΔA| (mediana medida 0,0007).
        #   · escala_fam       = 0.5   → ahora: la escala de la propia divergencia (media 0,0012).
        # Ningún llamador les pasaba valor (los cinco sitios construyen `OrganeloMemoria()` a secas),
        # así que quitarlos no rompe ninguna llamada.
        self.buffer_n = buffer_n
        self.tau_val_corto = tau_val_corto; self.tau_val_largo = tau_val_largo
        self.eta_corto = eta_corto; self.eta_largo = eta_largo
        self.max_episodios = max_episodios
        self.tau_persist_base = tau_persist_base; self.k_persist_vida = k_persist_vida
        self.tau_saciedad = tau_saciedad; self.olvido_relacional = olvido_relacional
        self.reset()

    def reset(self) -> None:
        self.buffer = deque(maxlen=self.buffer_n)   # capa 1
        self.valencia = {}                           # capa 3: clave -> {"corto","largo"}
        self.episodios = []                          # capa 4: lista de dicts
        self._A_prev = None                          # para ΔA (sorpresa)
        self._clave_presente = None; self._t_ausencia = 0.0   # capa 2
        self.confianza_persist = 0.0
        self.saciedad = 0.0                          # refractariedad de la necesidad
        self.necesidad_prev = 0.0
        self.confianza_otro = 0.0                    # capa 6
        self.t = 0.0; self.n_visitas = {}
        # LO HABITUAL DE CADA MAGNITUD, aprendido de la propia experiencia. Sustituye a los
        # ocho números que antes decidían qué es "mucho" en esta memoria. Una escala por
        # magnitud: mezclarlas sería volver a inventar una escala común que nadie midió.
        self.esc_div = Escala()    # divergencia Φ ↔ Phi_int_historia (familiaridad/novedad)
        self.esc_dA = Escala()     # |ΔA_sys_env|: cuánto suele cambiar el acople de un paso a otro
        self.esc_eR = Escala()     # e_R: el error de representación habitual
        self.esc_lat = Escala()    # |lateralidad|: cuán descentrada suele estar la fuente
        self.esc_irde = Escala()   # IRDE_ratio: el desequilibrio habitual
        self.esc_H = Escala()      # acople_sostenido: la homeostasis habitual
        self.esc_nov = Escala()    # novedad: cuán nuevo suele parecerle el mundo
        self.esc_sal = Escala()    # saliencia: qué tan llamativo suele ser un paso de su vida
        self.esc_fat = Escala()    # fatiga: el cansancio habitual
        self.esc_Cb = Escala()     # presión de desacople: la presión habitual

    # ----- PERSISTENCIA: la historia interna sobrevive al apagón (incremento 1) -----
    def snapshot(self) -> dict:
        """Estado VIVO a guardar (la historia, no las constantes del genoma): valencias aprendidas,
        episodios, vínculo con el otro (confianza_otro), vida vivida (t). Todo JSON-serializable."""
        # Las ESCALAS también son historia, no genoma: sin ellas el organismo reaprende en cada
        # arranque qué es "mucho" y qué "poco", y sus primeras decisiones del día quedan tomadas
        # contra una escala vacía (es la advertencia literal del docstring de escala.py).
        return {"valencia": self.valencia, "episodios": self.episodios,
                "confianza_otro": self.confianza_otro, "confianza_persist": self.confianza_persist,
                "saciedad": self.saciedad, "necesidad_prev": self.necesidad_prev,
                "t": self.t, "n_visitas": self.n_visitas, "buffer": list(self.buffer),
                "_clave_presente": self._clave_presente, "_t_ausencia": self._t_ausencia,
                "_A_prev": self._A_prev,
                "escalas": {n: getattr(self, n).snapshot() for n in self._NOMBRES_ESCALA}}

    def restore(self, d: dict) -> None:
        """Reconstituye la historia interna desde un snapshot (al despertar tras un reinicio)."""
        if not d:
            return
        self.valencia = d.get("valencia", {}) or {}
        self.episodios = d.get("episodios", []) or []
        self.confianza_otro = float(d.get("confianza_otro", 0.0) or 0.0)
        self.confianza_persist = float(d.get("confianza_persist", 0.0) or 0.0)
        self.saciedad = float(d.get("saciedad", 0.0) or 0.0)
        self.necesidad_prev = float(d.get("necesidad_prev", 0.0) or 0.0)
        self.t = float(d.get("t", 0.0) or 0.0)
        self.n_visitas = d.get("n_visitas", {}) or {}
        self.buffer = deque([tuple(x) for x in (d.get("buffer", []) or [])], maxlen=self.buffer_n)
        self._clave_presente = d.get("_clave_presente")
        self._t_ausencia = float(d.get("_t_ausencia", 0.0) or 0.0)
        self._A_prev = d.get("_A_prev")
        esc = d.get("escalas") or {}
        for n in self._NOMBRES_ESCALA:
            getattr(self, n).restore(esc.get(n))

    # --------------------------------------------------------------- utilidades
    def _clave(self, lat: float, A: float):
        """Firma discreta de la situación: dónde está la fuente × qué tan acoplado estoy.

        ANTES: `lb = -1 if lat < -0.15 else (1 if lat > 0.15 else 0)`. Dos problemas medidos
        sobre 99.668 pasos (3–4 ago 2026, ANIMA_5Z934MWHNNRH):
          · El 0,15 es un número puesto a mano sobre una magnitud cuya |mediana| es 0,166 y cuyo
            p75 es 0,985: partía la lateralidad casi exactamente por la mitad (50,37 % por encima)
            por casualidad, no por medida. AHORA el corte es lo que ESTE organismo suele tener
            descentrada la fuente (`esc_lat.media`): "más lateral de lo que me es habitual".
          · La rama izquierda (lat < −0,15) se cumplió 0 veces en 99.668 pasos: hoy la lateralidad
            que produce el campo NUNCA es negativa. NO la borro. La rama no está mal escrita: le
            falta signo AGUAS ARRIBA (fase 1.1 del plan devuelve el forzamiento bipolar al campo).
            Borrarla ahora sería tirar en silencio la mitad izquierda del mundo justo antes de que
            el organismo recupere la capacidad de distinguirla.
        """
        u = self.esc_lat.media if self.esc_lat.madura else 0.0
        lb = -1 if lat < -u else (1 if lat > u else 0)          # izq / centro / der
        ab = int(min(4, max(0, round(A * 4))))                  # 0..4 niveles de acople (A∈[0,1])
        return (lb, ab)

    def _familiaridad_soma(self, soma):
        """CAPA 5 (sólo lectura): convierte la memoria IMPLÍCITA del campo (W + Phi_int_historia)
        en señales consultables. familiaridad = ¿el campo actual se parece a su propia historia?

        ANTES: `familiaridad = exp(−div / escala_fam)` con `escala_fam = 0.5`. Medido sobre 99.668
        pasos: la divergencia real Φ↔historia tiene MEDIA 0,0012 y p99 0,0279. El 0,5 era 415 VECES
        la divergencia media del organismo, así que la exponencial devolvía 1,0000 siempre:
        mem_familiaridad mediana 1,0000 (p01 = 0,9457) y mem_novedad MÁXIMO HISTÓRICO 0,1339.
        Consecuencia: la categoría de episodio "novedad" se disparó 0 veces en 99.668 pasos —
        un organismo estructuralmente incapaz de encontrar nada nuevo.

        AHORA la escala es la propia divergencia habitual: familiaridad = 1 − rel(div, esc_div),
        que vale 0,5 cuando el campo se parece a su historia lo de siempre, sube hacia 1 cuando se
        le parece MÁS de lo habitual y baja hacia 0 cuando se le parece menos. Es autorregulada:
        cuando la fase 1 arregle el forzamiento del campo y la divergencia cambie de orden de
        magnitud, la escala se reaprende sola; un 0,5 escrito a mano habría vuelto a mentir.
        Es una PERCEPCIÓN (¿esto se parece a lo que conozco?), no una condición de viabilidad:
        relativizarla no puede dejar al organismo "crónicamente mal sin enterarse".
        """
        if soma is None:
            return None
        import numpy as np
        divs, cargas = [], []
        for h in ("L", "R", "BL", "BR"):
            hemi = getattr(soma, h, None)
            if hemi is None or not hasattr(hemi, "Phi_int_historia"):
                continue
            hist = hemi.Phi_int_historia
            if float(np.sum(np.abs(hist))) < 1e-9:        # historia aún vacía
                divs.append(0.0)
            else:
                divs.append(float(np.sqrt(np.mean((hemi.Phi - hist) ** 2))))   # RMS por nodo
            if hasattr(hemi, "W"):
                cargas.append(float(np.mean(np.abs(hemi.W))))
        if not divs:
            return None
        div = sum(divs) / len(divs)
        self.esc_div.observar(div)
        # Arranque: sin historia de divergencias, ni familiar ni nuevo (NEUTRO). Abstenerse es la
        # política de arranque única del proyecto; inventar un 1,0 sería declarar todo conocido.
        familiaridad = (1.0 - _rel(div, self.esc_div)) if self.esc_div.madura else NEUTRO
        carga = (sum(cargas) / len(cargas)) if cargas else 0.0
        return familiaridad, carga

    # --------------------------------------------------------------- ciclo
    def actualizar(self, fila: dict, dt: float = 0.1, milieu=None, soma=None) -> dict:
        self.t += dt
        # ---- PERCIBIR (de la fila + milieu + soma) ----
        A = _c01(fila.get("A_sys_env"))
        lat = _num(fila.get("lateralidad"))
        e_R = _num(fila.get("e_R"))
        icr_r = _c01(fila.get("ICR_ratio")); irde_r = _c01(fila.get("IRDE_ratio"))
        act_perm = _c01(fila.get("act_perm"))
        H_real = _c01(fila.get("acople_sostenido"))
        S_shared = max(_c01(fila.get("altruismo_S_shared")), _c01(fila.get("ME")))
        disp = _c01(fila.get("disposicion_cooperar"))
        Cb = _num(milieu.leer("presion_desacople", 0.0)) if milieu is not None else _num(fila.get("presion_desacople"))
        fat = _num(milieu.leer("fatiga_activa", 0.0)) if milieu is not None else _num(fila.get("act_fatiga"))
        vida = _num(milieu.leer("historia", 0.0)) if milieu is not None else self.t

        # ---- LO HABITUAL: se observa ANTES de decidir nada con ello ----
        self.esc_eR.observar(e_R); self.esc_lat.observar(abs(lat))
        self.esc_irde.observar(irde_r); self.esc_H.observar(H_real)
        self.esc_fat.observar(fat); self.esc_Cb.observar(Cb)

        # ANTES: `fat_norm = fat/(1+fat)`. Ese 1,0 declara que la fatiga típica del organismo vale
        # 1. Medida sobre 99.668 pasos, la fatiga tiene MEDIANA 327,5 (p05 111, p95 339): el mapeo
        # devolvía 0,9970 de mediana y la "energía disponible" para consolidar un recuerdo quedaba
        # en 0,0030 (p05 0,0029, p75 0,0036). Es decir: la memoria episódica estaba multiplicada
        # por ~0,003 SIEMPRE, y por eso no grababa (ver más abajo).
        # AHORA se compara la fatiga contra la fatiga habitual de este organismo: 0,5 cuando está
        # como siempre. Es una PERCEPCIÓN moduladora (¿estoy más cansado que de costumbre?), no la
        # condición de vida — la condición de vida es met_energia, la reserva, y ésa no se toca aquí.
        fat_norm = _rel(fat, self.esc_fat) if self.esc_fat.madura else NEUTRO
        if self._A_prev is None:
            self._A_prev = A
        dA = A - self._A_prev; self._A_prev = A
        self.esc_dA.observar(abs(dA))

        # ---- CAPA 5: memoria estructural implícita del soma → familiaridad/novedad ----
        fam_carga = self._familiaridad_soma(soma)
        if fam_carga is not None:
            familiaridad, carga_estructural = fam_carga
        else:
            familiaridad, carga_estructural = NEUTRO, 0.0
        # Sin soma, la novedad es cuánto se desvía el acople respecto de su cambio habitual.
        # ANTES ese respaldo era `|ΔA| / 0.05`; el 0,05 es 71 veces la mediana medida de |ΔA|
        # (0,0007) — declaraba que un cambio típico vale 1/71 de "novedad total".
        novedad = (1.0 - familiaridad) if fam_carga is not None else _rel(abs(dA), self.esc_dA)
        self.esc_nov.observar(novedad)

        # ---- CAPA 1: buffer corto ----
        self.buffer.append((round(self.t, 2), round(A, 3), round(lat, 3), round(H_real, 3), round(act_perm, 3)))

        clave = self._clave(lat, A)
        self.n_visitas[clave] = self.n_visitas.get(clave, 0) + 1

        # ---- CAPA 3: valencia (doble escala) ----
        # ANTES: `bondad = H_real`, y H_real ∈ [0,1]. Con un maestro que nunca baja de cero, la
        # valencia NO PODÍA SER NEGATIVA: medido el 3-ago en Abraxas, mem_valencia_estado vivía
        # entre 0,0001 y 0,2207. El organismo podía recordar que algo estuvo bien o que fue
        # indiferente, jamás que le cayó mal. Sin signo no hay aversión, y sin aversión no hay
        # nada que lo aparte de lo que lo daña.
        #
        # AHORA el maestro es el IMPACTO METABÓLICO: qué fracción de lo que costó vivir ese paso
        # alcanzó a pagar el bocado. Tiene signo por construcción, está normalizado contra el gasto
        # DEL PROPIO ORGANISMO, y es la definición material de "esto me sentó bien o mal".
        #
        # H_real queda FUERA a propósito: su mediana medida es 0,0136 sobre un rango [0,1], así que
        # centrarlo para darle signo dejaría la valencia clavada en ≈−1 y el organismo aborrecería
        # todo por igual — exactamente el defecto que se acaba de corregir en el paladar, invertido.
        # Un índice de viabilidad que vive pegado a cero es sospechoso y merece su propia auditoría;
        # mezclarlo aquí sin entenderlo sólo trasladaría el problema.
        #
        # MEDIDO EL 4-AGO Y NO SE CORRIGE AQUÍ (a propósito): met_impacto está en su suelo −1,0 en
        # más de la mitad de las 99.628 filas (p01 −1,0 · mediana −0,9909 · p75 −0,0657), y con eso
        # mem_valencia_estado quedó en mediana −0,3212 con p05 −0,8896: el organismo aborrece casi
        # todo. NO se relativiza met_impacto contra su propia historia, aunque lo arreglaría de un
        # plumazo, porque es una CONDICIÓN DE VIABILIDAD y no una percepción: dice si el bocado pagó
        # lo que costó vivir el paso, y tiene que poder quedarse mal. Un consigo-mismo aquí haría
        # que un organismo crónicamente desnutrido leyera "impacto normal" — el error que ya hubo
        # que revertir una vez. La escala de met_impacto la debe arreglar el metabolismo, que es
        # quien lo produce; esta memoria sólo hereda su signo. Anotado con la cifra para que quien
        # audite la valencia sepa que el sesgo no nace aquí.
        # Aguas abajo importa: mem_valencia_estado lo lee VST_CalibradorLexicoExperiencial, que es
        # con lo que el organismo aprende los NOMBRES de sus propios regímenes de experiencia.
        bondad = max(-1.0, min(1.0, _num(fila.get("met_impacto", 0.0))))
        d_corto = math.exp(-dt / self.tau_val_corto); d_largo = math.exp(-dt / self.tau_val_largo)
        for k, v in self.valencia.items():                 # olvido de lo no visitado (dos τ)
            v["corto"] *= d_corto; v["largo"] *= d_largo
        vc = self.valencia.setdefault(clave, {"corto": 0.0, "largo": 0.0})
        vc["corto"] += self.eta_corto * (bondad - vc["corto"])     # aprende rápido
        vc["largo"] += self.eta_largo * (vc["corto"] - vc["largo"])  # consolida lento
        valencia_estado = max(-1.0, min(1.0, 0.5 * vc["corto"] + 0.5 * vc["largo"]))

        # ---- CAPA 4: episódica. Saliencia EMERGENTE (sorpresa·intensidad·energía) ----
        # SORPRESA. El término de ΔA era `|ΔA| / 0.05`. Medido sobre 99.668 pasos: |ΔA| tiene
        # mediana 0,0007 y media 0,019, así que ese término valía 0,014 la mitad del tiempo y
        # saturaba en 1 el 8,9 %: no era una medida de sorpresa, era un interruptor. AHORA es
        # rel(|ΔA|, esc_dA): 0,5 cuando el acople cambia lo que suele cambiar.
        sorpresa = max(_c01(novedad),
                       _rel(abs(dA), self.esc_dA),
                       _rel(irde_r, self.esc_irde))
        # INTENSIDAD. ANTES `e_R/(1+e_R)`: ese 1,0 declara que el error de representación típico
        # vale 1. Medido: e_R tiene mediana 8,54 (p75 9,93, máx 26,3), de modo que el mapeo daba
        # 0,895 de mediana con el 46,8 % de los pasos por encima de 0,9 — una constante disfrazada
        # de medida. AHORA rel(e_R, esc_eR) vale 0,5 con el e_R de siempre. Además sobrevive al
        # "corte de época" de la fase 2 del plan: cuando e_R cambie de escala, esto se reaprende solo.
        intensidad = _c01(0.5 * _rel(e_R, self.esc_eR) + 0.5 * act_perm)
        energia = 1.0 - fat_norm
        saliencia = sorpresa * intensidad * energia
        self.esc_sal.observar(saliencia)

        # TIPO DE EPISODIO. ANTES: amenaza pedía `irde_r > 0.5` y logro pedía `H_real > 0.5`.
        # Medido: IRDE_ratio supera 0,5 el 68,86 % del tiempo y acople_sostenido sólo el 15,65 %.
        # Con la misma exigencia aparente ("medio"), la puerta de lo malo estaba 4,4 veces más
        # abierta que la de lo bueno: reconstruido sobre los 99.668 pasos, amenaza 4,53 % contra
        # logro 1,20 % (3,8×). La biografía del organismo estaba sesgada a lo aversivo por dos
        # números, no por su vida. AHORA cada puerta se abre cuando SU PROPIA magnitud está alta
        # PARA ESTE ORGANISMO (clasificar → "alto" = por encima de su media más su dispersión).
        # Al ser el mismo criterio sobre la escala de cada una, la asimetría desaparece por
        # construcción: lo bueno y lo malo se cuentan con la misma vara.
        # Y `novedad > 0.6` (0 casos en 99.668 pasos, máximo histórico 0,1339) pasa por la MISMA
        # vara que las otras dos: es novedad cuando la novedad está alta para este organismo.
        # Probado en replay: con el criterio laxo "novedad > familiaridad" la categoría se comía
        # el 48 % de los recuerdos; con la misma vara que amenaza y logro, las tres compiten en
        # igualdad y el reparto lo decide la vida del organismo, no el orden de los elif.
        cambio = self.esc_dA.madura and abs(dA) > self.esc_dA.media
        alto_irde = _clasificar(irde_r, self.esc_irde) == "alto"
        alto_H = _clasificar(H_real, self.esc_H) == "alto"
        alto_nov = _clasificar(novedad, self.esc_nov) == "alto"
        if cambio and dA < 0 and alto_irde:               tipo = "amenaza"
        elif cambio and dA > 0 and alto_H:                tipo = "logro"
        elif alto_nov:                                    tipo = "novedad"
        else:                                             tipo = "neutro"

        # ¿SE GRABA? ANTES: `saliencia > 0.40`. Reconstruida la saliencia sobre los 99.668 pasos
        # reales, cruzaba 0,40 el 0,090 % de las veces (mediana 0,00093) porque la "energía"
        # estaba multiplicando por 0,003. Resultado medido en el CSV: mem_episodios_n pegado a la
        # capacidad 64 el 67,93 % del tiempo con p01 = 38 — la memoria se llenó una vez y se
        # congeló —, y mem_recall_costo llegó a 1,97, que con 0,05+0,03·n significa que las 64
        # ranuras guardaban LA MISMA clave. Sesenta y cuatro copias de una situación no son una
        # biografía.
        #
        # AHORA hay dos condiciones, ninguna con número elegido:
        #  (a) que el paso destaque sobre la saliencia habitual DE SU PROPIA VIDA (clasificar);
        #      mientras no haya historia devuelve "indeterminado" y no se graba nada — un recién
        #      nacido no tiene con qué saber que algo es memorable, y abstenerse es la respuesta.
        #  (b) que, con la memoria llena, supere al episodio más débil que ya guarda. Es una
        #      comparación entre dos magnitudes del organismo con las MISMAS unidades (saliencia
        #      contra saliencia), que es la forma preferida por la auditoría frente a la media
        #      móvil. Y es lo que hace un archivo acotado de verdad: el listón sube solo a medida
        #      que la memoria se llena de cosas importantes, y baja solo cuando se vacía de ellas.
        destaca = _clasificar(saliencia, self.esc_sal) == "alto"
        if destaca:
            # Peso = intensidad descontada por antigüedad. ANTES el descuento era exp(−edad/600):
            # 600 s es un número sin origen escrito. AHORA la vara es el ALCANCE de su propia
            # memoria episódica (de lo más viejo que recuerda hasta ahora): un episodio en el
            # extremo de lo que alcanza a recordar pesa 1/e de uno recién vivido con la misma
            # intensidad. No pude medirlo en el CSV —los tiempos de cada episodio no se publican—,
            # así que en vez de calibrar el 600 lo derivo de una magnitud que el organismo tiene.
            alcance = max(1e-6, self.t - min((e["t"] for e in self.episodios), default=self.t))
            peso = lambda e: e["intensidad"] * math.exp(-(self.t - e["t"]) / alcance)
            hay_sitio = len(self.episodios) < self.max_episodios
            if hay_sitio or saliencia > min(peso(e) for e in self.episodios):
                self.episodios.append({"t": self.t, "clave": clave, "tipo": tipo,
                                       "intensidad": round(saliencia, 3),
                                       "valencia": round(valencia_estado, 3)})
                if len(self.episodios) > self.max_episodios:   # OLVIDO SELECTIVO por el mismo peso
                    self.episodios.sort(key=peso)
                    self.episodios.pop(0)

        # ---- CAPA 4 (lectura): RECALL explícito por similitud de clave, con COSTO ----
        coincidencias = [e for e in self.episodios if e["clave"] == clave]
        if coincidencias:
            ep = max(coincidencias, key=lambda e: e["intensidad"])
            mem_recall = 1.0; mem_recall_tipo = TIPO[ep["tipo"]]
            # ANTES: `0.05 + 0.03 · n`. Dos números sin origen que fingían segundos de latencia y
            # que nadie cobra: el mapa dice que mem_recall_costo no lo consume ningún organelo.
            # Medido, saturaba en 1,97 (las 64 ranuras con la misma clave). AHORA es la fracción
            # de la propia memoria que hay que revisar para evocar: dos magnitudes del organismo
            # con las mismas unidades (episodios), sin escala inventada.
            mem_recall_costo = round(len(coincidencias) / float(len(self.episodios)), 3)
        else:
            mem_recall = 0.0; mem_recall_tipo = TIPO["ninguno"]; mem_recall_costo = 0.0

        # ---- CAPA 2: persistencia (permanencia del objeto). τ crece con la vida vivida ----
        # ANTES: `(e_R > 0.5) or (|lat| > 0.15)`. Medido sobre 99.668 pasos: e_R > 0,5 el 54,49 %
        # (el 25 % de las filas trae e_R clavado en su suelo exacto de 0,5) y |lat| > 0,15 el
        # 50,37 %; su UNIÓN se cumple el 99,31 %. Y eso es exactamente lo que se ve en el CSV:
        # mem_persistencia == 1,0000 en el 99,31 % de las filas. La capa 2 entera estaba muerta,
        # y con ella tau_persist, k_persist_vida y la vida vivida que los alarga: el organismo
        # nunca llegaba a echar de menos nada porque nunca registraba una ausencia.
        # AHORA "hay algo ahí fuera" es que el error de representación O el descentramiento estén
        # por encima de lo habitual PARA ESTE ORGANISMO. Ojo con la advertencia de la auditoría:
        # esto NO borra un estado sostenido, porque en un silencio permanente la media converge al
        # propio valor y rel() da exactamente 0,5, que no supera el 0,5 — el organismo no se
        # inventa un estímulo por costumbre; sólo deja de estarlo cuando lo de fuera es lo de siempre.
        hay_estimulo = (_rel(e_R, self.esc_eR) > NEUTRO) or (_rel(abs(lat), self.esc_lat) > NEUTRO)
        tau_persist = self.tau_persist_base + self.k_persist_vida * math.log1p(max(0.0, vida))
        if hay_estimulo:
            self._clave_presente = clave; self._t_ausencia = 0.0; self.confianza_persist = 1.0
        else:
            self._t_ausencia += dt
            self.confianza_persist = math.exp(-self._t_ausencia / max(1e-6, tau_persist))

        # ---- CAPA 6: relacional (Hebb social). Plena sólo en díada (S_shared>0) ----
        # CAPA MUERTA, MEDIDA, Y NO POR CULPA DE NINGUNA CONSTANTE DE AQUÍ: mem_relacional_confianza
        # vale 0,0000 en 99.628 de 99.628 filas (un solo valor distinto). No es el olvido_relacional:
        # replayando ESTE mismo código sobre el CSV, la confianza sí se mueve (llega a 0,2527), así
        # que los valores existen en la fisiología pero la `fila` que recibe este organelo en vivo
        # no trae `altruismo_S_shared` ni `disposicion_cooperar`. Es un problema de cañería, no de
        # calibración, y se arregla donde se arma la fila. Dejado escrito para no volver a buscarlo.
        reciproco = S_shared * disp
        self.confianza_otro = self.olvido_relacional * self.confianza_otro + (1.0 - self.olvido_relacional) * reciproco
        self.confianza_otro = max(0.0, min(1.0, self.confianza_otro))

        # ---- NECESIDAD: Cb→act_perm (disposición × presión acumulada) + SACIEDAD/REFRACTARIEDAD ----
        # ANTES: `Cb_norm = Cb/(1+Cb)`. Ese 1,0 declara que la presión de desacople típica vale 1.
        # Medida sobre 99.668 pasos: presion_desacople tiene MEDIANA 110,47 (p01 1,06, p95 176,5),
        # así que Cb_norm valía 0,991 de mediana y `1 + Cb_norm` era el número 1,99 disfrazado de
        # historia: necesidad = 1,99·act_perm siempre. Se comprueba en el CSV sin salir de él —
        # necesidad mediana 0,2334 contra act_perm mediana 0,1207, razón 1,93. La presión acumulada,
        # que es justamente lo que convierte una disposición instantánea en NECESIDAD con historia,
        # no estaba aportando ninguna información.
        # AHORA rel(Cb, esc_Cb) vale 0,5 con la presión de siempre, sube cuando el desacople aprieta
        # más de lo habitual y baja cuando afloja. Lo que NO se relativiza es act_perm: sigue siendo
        # el factor absoluto, y si vale 0 la necesidad es 0 pase lo que pase con la presión. Ésa es
        # la parte que debe poder quedarse mal (condición), y la presión sólo la MODULA (percepción).
        Cb_norm = _rel(Cb, self.esc_Cb) if self.esc_Cb.madura else NEUTRO
        necesidad = max(0.0, min(1.0, act_perm * (1.0 + Cb_norm)))
        # SATISFACCIÓN: la necesidad se sacia por re-acople (A sube con H alta) O por COMER bien
        # (met_nutricion, del OrganeloMetabolismo) — así el lazo necesidad→comer→saciedad CIERRA.
        # El re-acople se mide otra vez contra el cambio habitual de A y no contra el 0,05 de antes.
        comer = _c01(fila.get("met_nutricion"))
        satisfaccion = max(_rel(max(0.0, dA), self.esc_dA) * H_real, comer) * self.necesidad_prev
        d_sac = math.exp(-dt / self.tau_saciedad)
        self.saciedad = max(0.0, min(1.0, d_sac * self.saciedad + (1.0 - d_sac) * satisfaccion))
        necesidad_efectiva = necesidad * (1.0 - self.saciedad)
        self.necesidad_prev = necesidad

        # ---- SECRETAR (al milieu, si está; y devolver columnas para la fila) ----
        out = {
            "mem_familiaridad": round(familiaridad, 4), "mem_novedad": round(_c01(novedad), 4),
            "mem_carga_estructural": round(carga_estructural, 5), "mem_valencia_estado": round(valencia_estado, 4),
            "mem_persistencia": round(self.confianza_persist, 4), "mem_recall": mem_recall,
            "mem_recall_tipo": mem_recall_tipo, "mem_recall_costo": mem_recall_costo,
            "mem_episodios_n": float(len(self.episodios)), "Cb_integrado": round(Cb, 4),
            "necesidad": round(necesidad, 4), "necesidad_efectiva": round(necesidad_efectiva, 4),
            "mem_saciedad": round(self.saciedad, 4), "mem_relacional_confianza": round(self.confianza_otro, 4),
        }
        if milieu is not None:
            milieu.secretar("necesidad", out["necesidad"])
            milieu.secretar("mem_novedad", out["mem_novedad"])
            milieu.secretar("mem_valencia_estado", out["mem_valencia_estado"])
        return out


if __name__ == "__main__":
    # SMOKE. El anterior repetía 50 veces LA MISMA fila inventada (e_R = 4,0 cuando la mediana
    # real es 8,54; presion_desacople = 2,0 cuando la mediana real es 110,5; act_fatiga = 1,0
    # cuando la mediana real es 327,5). Una prueba calibrada para pasar es peor que no tener
    # prueba: con una fila constante NADA puede ser sorprendente y el aparato episódico entero
    # queda sin ejercitar. Ahora se siembra con filas REALES del historial si las hay, y sólo
    # cae a la fila sintética —avisando— cuando no hay historial que leer.
    import csv as _csv, glob as _glob
    mem = OrganeloMemoria()
    _fuente = sorted(_glob.glob(r"C:\Users\adale\.anima\history\*\fisiologia\*.csv"))
    filas = []
    if _fuente:
        with open(_fuente[-1], newline="", encoding="utf-8", errors="replace") as _fh:
            for i, _row in enumerate(_csv.DictReader(_fh)):
                filas.append(_row)
                if i >= 2000:
                    break
    if not filas:
        print("AVISO: sin historial; smoke degradado sobre una fila sintética (no prueba nada).")
        filas = [{"A_sys_env": 0.6, "lateralidad": 0.3, "e_R": 8.5, "ICR_ratio": 0.6,
                  "IRDE_ratio": 0.78, "act_perm": 0.12, "acople_sostenido": 0.07,
                  "presion_desacople": 110.0, "act_fatiga": 327.0}]
    r = {}
    for f in filas:
        r = mem.actualizar(f, dt=0.1)
    print("smoke sobre", len(filas), "filas reales:", r)
    tipos = {}
    for e in mem.episodios:
        tipos[e["tipo"]] = tipos.get(e["tipo"], 0) + 1
    print("episodios:", len(mem.episodios), "por tipo:", tipos,
          "| claves de valencia:", len(mem.valencia))
    print("escalas:", {n: getattr(mem, n).n for n in OrganeloMemoria._NOMBRES_ESCALA})
