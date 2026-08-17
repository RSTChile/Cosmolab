#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoCloroplasto — la HOJA: captura de energía de la luz. El primer organismo AUTÓTROFO.
=============================================================================================
POR QUÉ EXISTE (idea de Alexis, 1-jul-2026)
-------------------------------------------
Hasta hoy los organismos eran HETERÓTROFOS: la energía aparecía por reposición metabólica algo
abstracta. Este organelo les da AUTOTROFIA — sostener su persistencia (S>0) capturando un gradiente
CÓSMICO real, la LUZ, y guardándolo—. Es la historia cosmogenética en miniatura: la vida enganchándose
al sol. Emula una HOJA con hardware fotovoltaico real (que Alexis tiene en físico):

  · panel solar   WYM 84.3×56 mm · 5 VDC · 0,5 W
  · batería LiPo  503035 · 3,7 V · 500 mAh  (≈ 1,85 Wh ≈ 6660 J de reserva)
  · controlador de carga solar (corta al llenar; corte por baja tensión = "brownout")

NOMBRE — por qué "Cloroplasto" y no "panel": lo nombramos por su FUNCIÓN (capturar luz → energía), que
es la del cloroplasto. Estructuralmente hoy es fotovoltaico, NO fotosíntesis (para eso haría falta la
cianobacteria endosimbionte). Pero ESTE locus es, a propósito, donde se alojará el cloroplasto REAL
cuando llegue la endosimbiosis eucariota. Es la semilla de ese salto.

ANTI-SHANNON: la luz aquí es RECURSO, no señal. El organismo NO lee significado en la luz ni la
decodifica — se ALIMENTA de ella. Es metabolismo, no semiosis. Limpio.

QUÉ COMPUTA
-----------
  cloro_luz        intensidad de luz [0,1] del ciclo día/noche (horario que fijamos NOSOTROS, no la Tierra)
  cloro_dia        1 de día, 0 de noche (continuo en el crepúsculo)
  cloro_potencia   potencia captada por el panel (∝ luz)
  cloro_carga      estado de carga de la batería (SoC) [0,1] — la RESERVA de vida
  cloro_flujo      flujo neto de energía (carga − gasto): + de día, − de noche
  cloro_energia    energía DISPONIBLE para el metabolismo (autótrofo puro: es su ÚNICA fuente)
  cloro_deficit    1 si SoC bajo el corte (brownout: la vida se ralentiza; de noche, sin sol, es fatal)

AUTÓTROFO PURO: cuando está ACTIVO, la energía del organismo viene SOLO de la batería (carga con luz,
drena con el gasto metabólico). No hay reposición heterótrofa. De noche vive de lo guardado → de ahí
puede EMERGER un ritmo día/noche (callar de noche, porque vocalizar cuesta energía real) sin programarlo.

APAGADO POR DEFECTO (activo=False): capacidad RESERVADA. Solo se activa en el organismo-planta E.
Persistible (la batería conserva su carga entre vidas).
"""
from __future__ import annotations
import math

COLS_CLORO = ["cloro_luz", "cloro_dia", "cloro_potencia", "cloro_carga", "cloro_flujo",
              "cloro_energia", "cloro_deficit"]
COLS_FOTO = ["foto_v_panel", "foto_v_lipo", "foto_luz_norm", "foto_aporte", "foto_sensor_vivo"]

# --- specs del hardware real (documentación; el modelo interno es NORMALIZADO a [0,1]) ---
PANEL_W = 0.5           # W pico del panel WYM
BAT_WH = 3.7 * 0.5      # 1,85 Wh = batería LiPo 503035
BAT_J = BAT_WH * 3600   # ≈ 6660 J de reserva real


def _c01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

def _g(fila, *claves, default=0.0):
    for k in claves:
        if k in fila and fila[k] is not None:
            try: return float(fila[k])
            except (TypeError, ValueError): pass
    return default


class OrganoCloroplasto:
    """La hoja: luz → panel → controlador → batería → energía. Autótrofo puro. Apagado por defecto."""

    def __init__(self, organismo_id=None, activo: bool = False,
                 modo: str = "simulado",        # "simulado" (batería SoC) | "real" (sensor ATmega → foto_aporte)
                 k_foto: float = 0.03,          # aporte fotosintético aditivo por paso (modo real)
                 v_oscuro: float = 0.15,        # calibración panel: voltaje de noche
                 v_pleno: float = 4.20,         # calibración panel: voltaje a pleno sol
                 periodo_s: float = 600.0,      # duración de UN ciclo día+noche completo (NUESTRA hora; ajustable)
                 frac_dia: float = 0.5,         # fotoperiodo: fracción del ciclo que es DÍA (0.5 = 12/12)
                 soc_inicial: float = 0.5,      # carga inicial de la batería
                 tasa_carga: float = 0.9,       # eficiencia de carga (luz→batería) por unidad de tiempo
                 tasa_gasto: float = 1.0,       # cuánto drena el gasto metabólico la batería
                 gasto_basal: float = 0.010,    # coste de estar vivo (drena aun en reposo), por segundo-sim
                 soc_corte: float = 0.05):      # bajo esto = brownout (corte por baja tensión del controlador)
        self.organismo_id = organismo_id
        self.activo = bool(activo)
        self.modo = str(modo).lower()
        self.k_foto = float(k_foto)
        self.v_oscuro = float(v_oscuro); self.v_pleno = float(v_pleno)
        self.periodo_s = float(periodo_s); self.frac_dia = _c01(frac_dia)
        self.tasa_carga = float(tasa_carga); self.tasa_gasto = float(tasa_gasto)
        self.gasto_basal = float(gasto_basal); self.soc_corte = float(soc_corte)
        self.soc = _c01(soc_inicial)           # estado de carga (la reserva de vida)
        self._t_prev = None                    # último reloj visto (para dt)
        self.ultimo: dict = {}

    def _luz_normalizada(self, v_panel: float) -> float:
        x = (v_panel - self.v_oscuro) / max(1e-6, self.v_pleno - self.v_oscuro)
        return _c01(x)

    def _observar_real(self, fila: dict) -> dict:
        """Sensor ATmega: luz modula met_energia vía foto_aporte (aditivo, anti-Shannon)."""
        vivo = _g(fila, "atmega_vivo", default=0.0) >= 0.5
        vp = _g(fila, "v_panel", "v_fuente", default=0.0)
        vl = _g(fila, "v_lipo", default=0.0)
        luz = self._luz_normalizada(vp) if vivo else 0.0
        aporte = self.k_foto * luz if vivo else 0.0
        self.ultimo = {
            "foto_v_panel": round(vp, 4), "foto_v_lipo": round(vl, 4),
            "foto_luz_norm": round(luz, 4), "foto_aporte": round(aporte, 5),
            "foto_sensor_vivo": 1.0 if vivo else 0.0,
        }
        return self.ultimo

    # ---------- luz del ciclo día/noche (horario propio, no el de la Tierra) ----------
    def _luz(self, t: float) -> float:
        """Intensidad de luz [0,1] en el instante t. Día = medio-seno suave (amanece, cenit, atardece);
        noche = 0. El fotoperiodo (frac_dia) fija cuánto dura el día dentro del ciclo."""
        if self.periodo_s <= 0:
            return 1.0
        fase = (t % self.periodo_s) / self.periodo_s        # 0..1 dentro del ciclo
        if self.frac_dia <= 0:
            return 0.0
        if fase >= self.frac_dia:
            return 0.0                                       # noche
        x = fase / self.frac_dia                             # 0..1 dentro del día
        return math.sin(math.pi * x) ** 0.8                  # amanecer→cenit→atardecer, suave

    def observar(self, fila: dict) -> dict:
        """Avanza la hoja un paso: calcula luz, carga el panel, drena el gasto, actualiza la batería.
        Reloj: usa el tiempo de simulación 't' de la fila (reproducible). Si está APAGADO, no hace nada
        (energía = la del metabolismo heterótrofo habitual)."""
        if not self.activo:
            cols = COLS_FOTO if self.modo == "real" else COLS_CLORO
            self.ultimo = {c: None for c in cols}
            return self.ultimo
        if self.modo == "real":
            return self._observar_real(fila)
        t = _g(fila, "t", default=(self._t_prev or 0.0) + 1.0)
        dt = 0.0 if self._t_prev is None else max(0.0, min(5.0, t - self._t_prev))
        self._t_prev = t

        luz = self._luz(t)
        potencia = luz                                       # panel normalizado: P ∝ luz (P_pico=PANEL_W real)
        carga = self.tasa_carga * potencia                   # entrada de energía (SoC/seg) mientras hay sol
        # GASTO: coste basal de vivir + gasto metabólico real de este paso (vocalizar/acuñar cuestan)
        gasto = self.gasto_basal + self.tasa_gasto * _g(fila, "met_gasto", "met_energia_gasto", default=0.0)
        flujo = carga - gasto
        self.soc = _c01(self.soc + flujo * dt)               # el controlador clampa [0,1] (corte al llenar)

        deficit = 1.0 if self.soc <= self.soc_corte else 0.0
        # energía DISPONIBLE (autótrofo puro): la batería ES la energía. En brownout, casi nada.
        energia = 0.0 if deficit else self.soc

        self.ultimo = {
            "cloro_luz": round(luz, 4), "cloro_dia": round(_c01(luz * 4), 3),
            "cloro_potencia": round(potencia, 4), "cloro_carga": round(self.soc, 4),
            "cloro_flujo": round(flujo, 5), "cloro_energia": round(energia, 4),
            "cloro_deficit": deficit,
        }
        return self.ultimo

    # ---------- persistencia (la batería conserva su carga entre vidas) ----------
    def snapshot(self) -> dict:
        return {"soc": self.soc, "t_prev": self._t_prev, "activo": self.activo, "modo": self.modo}
    def restore(self, snap: dict) -> None:
        if not isinstance(snap, dict): return
        self.soc = _c01(snap.get("soc", self.soc))
        self._t_prev = snap.get("t_prev", self._t_prev)
        if "activo" in snap: self.activo = bool(snap["activo"])
        if "modo" in snap: self.modo = str(snap["modo"]).lower()


# ---------------- auto-prueba: simula ciclos día/noche y muestra la batería ----------------
if __name__ == "__main__":
    print("BATERÍA OrganoCloroplasto — 3 ciclos, periodo=120s, fotoperiodo 0.5 (60s día / 60s noche)")
    org = OrganoCloroplasto("E", activo=True, periodo_s=120.0, frac_dia=0.5, soc_inicial=0.6)
    # escenario: gasto metabólico moderado y constante (vive y a veces vocaliza)
    N = 360; hitos = []
    for i in range(N):
        t = float(i)                                          # 1 paso = 1 s-sim
        d = org.observar({"t": t, "met_gasto": 0.006})        # gasto real de este paso
        if i % 30 == 0:
            fase = "DÍA " if d["cloro_luz"] > 0 else "noche"
            hitos.append(f"  t={i:3d}s {fase} luz={d['cloro_luz']:.2f} SoC={d['cloro_carga']:.3f} "
                         f"flujo={d['cloro_flujo']:+.4f}{' ⚠brownout' if d['cloro_deficit'] else ''}")
    print("\n".join(hitos))
    print(f"\nreserva real emulada: {BAT_J:.0f} J ({BAT_WH:.2f} Wh) · panel {PANEL_W} W")
    print("→ la carga sube de día, baja de noche; si el gasto supera la carga diaria, colapsa (autótrofo puro).")
