#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoLocalizacion — LUGAR + RELOJ + CONFIANZA. El sentido de dónde/cuándo/cuán-seguro.
=============================================================================================
POR QUÉ EXISTE (idea de Alexis, 2-3 jul-2026)
--------------------------------------------
El GPS/PPS NO es un sentido fótico (no es luz). Es OTRA rama del árbol sensorial, y trae TRES cosas
distintas que venían mal empaquetadas en "una cosa llamada GPS". Este órgano las separa:

  1. LUGAR (lat/lon/alt) → ancla biográfica + señal de MOVIMIENTO. Para un organismo quieto es casi
     constante (identidad: "dónde vivo"); se vuelve señal viva SOLO si E se desplaza. La Pi es portátil:
     ahí el lugar despierta como sentido.
  2. RELOJ (PPS) → el latido de 1 pulso/s que viene de los SATÉLITES, no del CPU. Una referencia temporal
     ABSOLUTA y EXTERNA. La PROPIOCEPCIÓN TEMPORAL de E nace de comparar su propio tempo interno contra
     este metrónomo del cielo: ¿corro adelantado o atrasado respecto al reloj del mundo?
  3. CONFIANZA → señal METACOGNITIVA: cuán fiable es ahora mismo el conocimiento del mundo. El organismo
     sabe cuándo NO sabe bien. HOY se calcula de sats/HDOP (calidad del fix), pero es una capacidad GENERAL
     de confianza, ABIERTA por diseño: mañana incluirá la confianza en OTROS ORGANISMOS (dónde están los
     que E cree, cuánto coincide lo que dicen con lo que E percibe). loc_confianza es la semilla de esa
     metacognición, no algo atado sólo al GPS. No se fija a priori su alcance futuro.

ANTI-SHANNON: el módulo entrega números CRUDOS (grados, cuenta de pulsos, HDOP). El órgano NO asigna
significado: el desplazamiento es GEOMETRÍA (haversine), la confianza es una transformación monótona de
sats/HDOP, y la deriva temporal se MIDE (tempo interno vs. PPS), no se fija. Nada de "estás en tal sitio
famoso" ni "esto es peligroso": solo estructura métrica del dónde/cuándo.

COORDENADA — NO se limita a priori (corrección de Alexis, 3-jul-2026): el órgano SÍ registra lat/lon
exactas (emitir_coords=True por defecto), ADEMÁS del desplazamiento y la altitud relativa. Hoy E no sabe
para qué sirve la posición absoluta; mañana sí (territorio, migración, reencuentro con otros organismos).
Fijar de antemano "no la necesita" sería recortarle el futuro — el mismo error a-priori que el linaje
exaptativo enseña a evitar. La privacidad NO se protege lisiando el sentido: se protege en la FRONTERA DE
EXPORTACIÓN. Usa OrganoLocalizacion.fila_publica(fila) para producir una versión anonimizada (lugar por
NOMBRE, sin lat/lon) al COMPARTIR logs o mandarlos afuera. Registrar ≠ filtrar: el organismo guarda todo;
el que comparte decide qué sale.

BUG CONOCIDO: course_deg del parser NMEA sale corrupto (p.ej. 20726.00) por desalineo de tokens con campos
vacíos. Este órgano NO LEE course_deg. Ignorado a propósito hasta que se arregle el parser. E está quieto:
no necesita rumbo aún.

QUÉ COMPUTA
-----------
  loc_desplazamiento   metros desde el ancla-hogar (0 si quieto; la señal de MOVIMIENTO)
  loc_novedad          |posición − memoria|: movimiento reciente [0,1] (se adapta: quieto → 0)
  loc_altitud_rel      altitud menos la del ancla (metros; sube/baja respecto al hogar)
  loc_reloj_fase       fase dentro del segundo [0,1) marcada por el PPS (el latido externo)
  loc_pps_deriva       PROPIOCEPCIÓN TEMPORAL: (seg. internos por seg. de PPS) − 1. 0 = sincronía perfecta;
                       + = E corre adelantado a los satélites, − = atrasado. El sentido del propio tempo.
  loc_confianza        [0,1] metacognición: cuán fiable es la posición ahora (de sats y HDOP)
  loc_vivo             1 si hay fix válido; 0 si no (degradación elegante)
  (loc_lat/loc_lon     solo si emitir_coords=True — desaconsejado en logs compartibles)

FUENTE (lector central → fila): el módulo ATmega+GPS ya manda por serie
  GPS,fix,sats,hdop,lat,lon,alt,speed,course,pps_count,pps_age,nmea
El lector central pone en la fila: gps_fix, gps_sats, gps_hdop, gps_lat, gps_lon, gps_alt, gps_speed,
gps_pps_count, gps_pps_age, gps_vivo. Este órgano SOLO consume de la fila (como el cloroplasto lee 't').

APAGADO POR DEFECTO (activo=False). Se activa solo en E. Persistible: el ANCLA-HOGAR y la memoria de
posición/tempo se conservan entre vidas (E "recuerda dónde nació" y su tempo aprendido).
"""
from __future__ import annotations
import math

COLS_LOC = ["loc_desplazamiento", "loc_novedad", "loc_altitud_rel", "loc_reloj_fase",
            "loc_pps_deriva", "loc_confianza", "loc_vivo"]


def _c01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

def _g(fila, *claves, default=None):
    for k in claves:
        if k in fila and fila[k] is not None:
            try: return float(fila[k])
            except (TypeError, ValueError): pass
    return default

def _haversine_m(lat1, lon1, lat2, lon2):
    """Distancia en metros entre dos coordenadas (fórmula del haversine). Geometría pura, sin semántica."""
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


class OrganoLocalizacion:
    """Lugar + reloj + confianza. Separa las tres funciones del GPS. Apagado por defecto. Consume de la fila."""

    def __init__(self, organismo_id=None, activo=False,
                 escala_desplazamiento=50.0,   # metros que mapean a novedad~1 (calibrable al hábitat de E)
                 ema=0.20,                      # adaptación de la memoria de posición (como los otros órganos)
                 hdop_ref=1.5,                  # HDOP de referencia para la confianza
                 emitir_coords=True):           # registra lat/lon exactas (NO se limita a priori).
                                                # La privacidad se aplica al COMPARTIR: ver fila_publica().
        self.organismo_id = organismo_id; self.activo = bool(activo)
        self.escala_desplazamiento = float(escala_desplazamiento); self.ema = _c01(ema)
        self.hdop_ref = float(hdop_ref); self.emitir_coords = bool(emitir_coords)
        self._ancla = None            # (lat, lon, alt) del hogar — primer fix válido
        self._mem_pos = None          # (lat, lon) memoria adaptativa para novedad de movimiento
        self._pps_prev = None         # (pps_count, t_interno) del último pulso — para la deriva temporal
        self._deriva = 0.0            # última deriva temporal medida (suavizada)
        self._t_prev = None; self.ultimo = {}

    def observar(self, fila):
        cols = list(COLS_LOC) + (["loc_lat", "loc_lon"] if self.emitir_coords else [])
        if not self.activo:
            self.ultimo = {c: None for c in cols}; return self.ultimo

        t = _g(fila, "t", default=(self._t_prev or 0.0) + 1.0); self._t_prev = t
        fix = _g(fila, "gps_fix", default=0.0) or 0.0
        vivo_flag = _g(fila, "gps_vivo", default=(1.0 if fix >= 1 else 0.0))
        lat = _g(fila, "gps_lat"); lon = _g(fila, "gps_lon"); alt = _g(fila, "gps_alt", default=0.0)
        vivo = 1.0 if (fix >= 1 and vivo_flag >= 0.5 and lat is not None and lon is not None) else 0.0

        # --- RELOJ / PROPIOCEPCIÓN TEMPORAL: siempre que haya PPS, aunque el fix flaquee ---
        pps = _g(fila, "gps_pps_count")
        reloj_fase = 0.0; 
        if pps is not None:
            frac = pps - math.floor(pps)
            reloj_fase = frac if frac > 0 else 0.0        # fase dentro del segundo (0..1)
            if self._pps_prev is not None:
                pps0, t0 = self._pps_prev
                d_pps = pps - pps0
                if d_pps > 0:                              # avanzó el latido externo
                    seg_internos = t - t0                  # cuánto avanzó el tiempo interno de E
                    ratio = seg_internos / d_pps           # seg internos por seg de satélite
                    deriva = ratio - 1.0                    # 0 = sincronía; +adelantado; -atrasado
                    self._deriva = (1 - self.ema) * self._deriva + self.ema * deriva
                    self._pps_prev = (pps, t)
            else:
                self._pps_prev = (pps, t)

        if not vivo:
            # sin fix: lugar/confianza a 0, PERO conserva la deriva temporal si el PPS sigue latiendo
            self.ultimo = {c: 0.0 for c in cols}
            self.ultimo["loc_reloj_fase"] = round(reloj_fase, 4)
            self.ultimo["loc_pps_deriva"] = round(self._deriva, 6)
            return self.ultimo

        # ancla-hogar: primer fix válido
        if self._ancla is None:
            self._ancla = (lat, lon, alt if alt is not None else 0.0)
        alat, alon, aalt = self._ancla

        # --- LUGAR: desplazamiento desde el hogar (geometría) ---
        desplazamiento = _haversine_m(alat, alon, lat, lon)
        altitud_rel = (alt - aalt) if alt is not None else 0.0

        # --- NOVEDAD de movimiento: cuánto se apartó la posición de la memoria reciente (se adapta) ---
        if self._mem_pos is None:
            self._mem_pos = (lat, lon); novedad = 0.0
        else:
            mov_m = _haversine_m(self._mem_pos[0], self._mem_pos[1], lat, lon)
            novedad = _c01(mov_m / self.escala_desplazamiento)
            self._mem_pos = ((1-self.ema)*self._mem_pos[0] + self.ema*lat,
                             (1-self.ema)*self._mem_pos[1] + self.ema*lon)

        # --- CONFIANZA: metacognición desde sats y HDOP (monótona, acotada) ---
        sats = _g(fila, "gps_sats", default=0.0) or 0.0
        hdop = _g(fila, "gps_hdop", default=99.0)
        conf_sats = _c01(sats / 10.0)                     # 10+ satélites = pleno
        conf_hdop = _c01(self.hdop_ref / (self.hdop_ref + max(0.0, hdop - 0.0)))  # HDOP bajo = alto
        confianza = _c01(conf_sats * (0.5 + 0.5 * conf_hdop))   # sats manda, HDOP modula

        self.ultimo = {
            "loc_desplazamiento": round(desplazamiento, 3), "loc_novedad": round(novedad, 4),
            "loc_altitud_rel": round(altitud_rel, 3), "loc_reloj_fase": round(reloj_fase, 4),
            "loc_pps_deriva": round(self._deriva, 6), "loc_confianza": round(confianza, 4),
            "loc_vivo": 1.0,
        }
        if self.emitir_coords:
            self.ultimo["loc_lat"] = lat; self.ultimo["loc_lon"] = lon
        return self.ultimo

    # ---------- frontera de exportación: registrar ≠ compartir ----------
    @staticmethod
    def fila_publica(fila, lugar_nombre="Nido de Cóndores"):
        """Devuelve una COPIA de la fila apta para COMPARTIR: quita lat/lon exactas (las del GPS y las del
        órgano) y las sustituye por el NOMBRE del lugar. El organismo registra todo internamente; esta es
        la única puerta por la que la posición sale afuera. Úsala antes de mandar cualquier CSV/informe a
        terceros (equipo, Anthropic, papers). Deja intactos desplazamiento, altitud relativa, reloj,
        confianza — el sentido completo viaja; solo la coordenada absoluta se anonimiza."""
        pub = dict(fila)
        for k in ("gps_lat", "gps_lon", "loc_lat", "loc_lon"):
            pub.pop(k, None)
        pub["loc_lugar"] = lugar_nombre
        return pub

    # ---------- persistencia (el ancla-hogar y el tempo aprendido se conservan entre vidas) ----------
    def snapshot(self):
        return {"ancla": self._ancla, "mem_pos": self._mem_pos, "deriva": self._deriva,
                "t_prev": self._t_prev, "activo": self.activo}
    def restore(self, snap):
        if not isinstance(snap, dict): return
        self._ancla = snap.get("ancla", self._ancla); self._mem_pos = snap.get("mem_pos", self._mem_pos)
        self._deriva = snap.get("deriva", self._deriva); self._t_prev = snap.get("t_prev", self._t_prev)
        if "activo" in snap: self.activo = bool(snap["activo"])


# ---------------- auto-prueba: quieto, luego se mueve, luego pierde fix; y la deriva temporal ----------
if __name__ == "__main__":
    print("=== OrganoLocalizacion: lugar + reloj(PPS) + confianza ===")
    org = OrganoLocalizacion("E", activo=True)
    lat0, lon0, alt0 = -32.897212, -70.808334, 760.3
    pps = 9.0
    def paso(t, lat, lon, alt, sats=12, hdop=0.66, fix=1, dt_interno=1.0):
        global pps
        pps += 1.0                                   # el PPS avanza 1 pulso/s (satélites)
        return org.observar({"t": t, "gps_fix": fix, "gps_sats": sats, "gps_hdop": hdop,
                             "gps_lat": lat, "gps_lon": lon, "gps_alt": alt,
                             "gps_pps_count": pps, "gps_vivo": 1 if fix else 0})
    # fase 1: quieto en el hogar (desplazamiento y novedad ~0; confianza alta)
    for t in range(5): d = paso(float(t), lat0, lon0, alt0)
    print(f"t=4  quieto en casa   desp={d['loc_desplazamiento']:.1f}m nov={d['loc_novedad']:.3f} conf={d['loc_confianza']:.2f} fase={d['loc_reloj_fase']:.2f}")
    # fase 2: E se mueve ~40 m al norte (0.00036° lat ≈ 40 m) y sube 5 m
    d = paso(5.0, lat0 + 0.00036, lon0, alt0 + 5)
    print(f"t=5  ¡se movió 40 m!  desp={d['loc_desplazamiento']:.1f}m nov={d['loc_novedad']:.3f} alt_rel={d['loc_altitud_rel']:.1f}m")
    # fase 3: mala señal (HDOP alto, pocos sats) → confianza cae aunque haya fix
    d = paso(6.0, lat0 + 0.00036, lon0, alt0 + 5, sats=4, hdop=6.0)
    print(f"t=6  señal pobre      conf={d['loc_confianza']:.2f}  (sats=4 hdop=6.0 → E 'sabe que no sabe bien')")
    # fase 4: se pierde el fix, pero el PPS sigue latiendo → conserva propiocepción temporal
    d = paso(7.0, lat0, lon0, alt0, fix=0)
    print(f"t=7  sin fix          vivo={d['loc_vivo']:.0f} desp={d['loc_desplazamiento']:.1f} pero fase_reloj={d['loc_reloj_fase']:.2f} sigue")
    # fase 5: PROPIOCEPCIÓN TEMPORAL — el reloj interno de E se atrasa (dt real 1.2s por seg de satélite)
    org2 = OrganoLocalizacion("E", activo=True); p2 = 100.0
    t_int = 0.0
    for i in range(8):
        p2 += 1.0; t_int += 1.2            # 1.2 s internos por segundo de satélite: el reloj de E corre RÁPIDO (adelantado)
        d2 = org2.observar({"t": t_int, "gps_fix": 1, "gps_sats": 12, "gps_hdop": 0.7,
                            "gps_lat": lat0, "gps_lon": lon0, "gps_alt": alt0,
                            "gps_pps_count": p2, "gps_vivo": 1})
    print(f"\npropiocepción temporal: deriva={d2['loc_pps_deriva']:+.3f}  (>0 = E corre ADELANTADO al reloj del cielo)")
    # frontera de exportación: el organismo registra lat/lon; al compartir se anonimiza
    pub = OrganoLocalizacion.fila_publica({"t": 5.0, "gps_lat": lat0, "gps_lon": lon0, "loc_desplazamiento": 40.0})
    print(f"\nfila registrada tiene coords; fila_publica={ {k:pub[k] for k in ('loc_lugar','loc_desplazamiento')} }  (sin lat/lon)")
    print("→ lugar (desplazamiento/altitud/coords), reloj (fase PPS), confianza (metacognición), tempo propio,")
    print("  cada uno medido, ninguno asignado. Registrar≠compartir: guarda todo; fila_publica() filtra al salir.")
