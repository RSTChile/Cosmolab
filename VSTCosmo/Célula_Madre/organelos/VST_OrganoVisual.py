#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VST_OrganoVisual — el OJO: un banco de RETINAS en paralelo + una corteza que las funde.
=============================================================================================
POR QUÉ EXISTE (idea de Alexis, 2-3 jul-2026)  ·  Nivel 2 del linaje fótico
--------------------------------------------------------------------------
El linaje fótico (ver linajes_sensoriales_exaptativos_ANIMA.md):
  · Nivel 0 — Cloroplasto : luz → energía (RECURSO). El estómago fótico.
  · Nivel 1 — Proto-ojo   : leer la VARIACIÓN de la luz (fotorrecepción temporal).
  · Nivel 2 — ESTE órgano : resolver ESPACIO y BANDAS. El ojo. SEÑAL, no recurso.

UN SOLO ÓRGANO, MUCHAS RETINAS (correcciones de Alexis, 3-jul-2026)
-------------------------------------------------------------------
1) El ojo NO se desacopla en "lector" + "órgano": la retina física y la corteza son EL MISMO órgano,
   como el nervio óptico no es un ser aparte.
2) Y el ojo NO tiene UNA retina intercambiable: tiene MUCHAS retinas A LA VEZ. Como la araña saltarina
   con ocho ojos de tipos distintos, o el insecto con facetados Y ocelos simultáneos. La cámara, el
   proto-ojo del panel, y mañana un lidar o un térmico, son retinas que coexisten y se FUNDEN en una
   sola percepción. Añadir un "ojo" nuevo = añadir una retina al banco, sin tocar la corteza.

       Retina cámara  ┐
       Retina panel   ├─►  CORTEZA  ─►  novedad por banda + saliencia FUNDIDA + banda dominante
       Retina lidar   ┘    (memoria/adaptación por retina; el sentido = desviación, no valor absoluto)
       Retina térmico …

ANTI-DOBLE-CONTEO: ninguna retina alimenta met_energia. Luz-como-alimento = cloroplasto;
luz/mundo-como-vista = aquí. Percepción pura.

ANTI-SHANNON: las retinas entregan magnitudes CRUDAS. La corteza NO etiqueta objetos ni reconoce caras.
Mide NOVEDAD = desviación de la memoria de CADA retina (adaptación: lo constante deja de notarse). La
banda DOMINANTE (hacia dónde "mira" el ojo) EMERGE de cuál retina es ahora más saliente — no se asigna.
El significado pleno emerge del binding cross-modal (¿lo visto coincide con lo oído?), en la atención.

COLUMNAS
--------
  Globales (siempre):
    vis_saliencia    'cuánto vale mirar' [0,1] — FUSIÓN de todas las retinas (candidato al lazo de atención)
    vis_dominante    nombre de la retina más saliente ahora (hacia dónde mira el ojo; EMERGENTE)
    vis_novedad      novedad de la banda dominante [0,1]
    vis_vivo         1 si AL MENOS una retina reporta datos frescos; 0 si todas cayeron
    vis_n_retinas    cuántas retinas están vivas ahora
  Por retina <n> (según el banco configurado), p.ej. la cámara aporta color y contraste:
    vis_<n>_intensidad  escalar primario de esa banda [0,1] (brillo, cercanía-lidar, calor-térmico…)
    vis_<n>_novedad     |intensidad − memoria_<n>| [0,1]
    vis_<n>_movimiento  cambio espacial/temporal [0,1] (si la retina lo provee)
    vis_<n>_saliencia   contribución de esa retina a la saliencia [0,1]
    (cámara además) vis_cam_tono_r/g/b, vis_cam_contraste

APAGADO POR DEFECTO (activo=False): capacidad RESERVADA, se activa solo en E. Persistible (la memoria de
adaptación de cada retina se conserva entre vidas).
"""
from __future__ import annotations
import threading, time

def _c01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

def _g(fila, *claves, default=0.0):
    for k in claves:
        if k in fila and fila[k] is not None:
            try: return float(fila[k])
            except (TypeError, ValueError): pass
    return default


def reducir_frame(pixels_gris, rgb_medios, prev_gris):
    """Imagen decodificada → magnitudes crudas (brillo, contraste, movimiento, color). Sin numpy."""
    n = len(pixels_gris)
    if n == 0:
        return {"brillo": 0.0, "contraste": 0.0, "movimiento": 0.0,
                "r": 0.0, "g": 0.0, "b": 0.0}, prev_gris
    media = sum(pixels_gris) / n
    contraste = (sum((p - media) ** 2 for p in pixels_gris) / n) ** 0.5
    if prev_gris is not None and len(prev_gris) == n:
        movimiento = sum(abs(a - b) for a, b in zip(pixels_gris, prev_gris)) / n
    else:
        movimiento = 0.0
    r, g, b = rgb_medios
    return {"brillo": media, "contraste": contraste, "movimiento": movimiento,
            "r": r, "g": g, "b": b}, list(pixels_gris)


# =========================================================================================
#  RETINAS — cada una es un "ojo" del banco. Interfaz común: nombre, iniciar/detener, leer().
#  leer() -> (crudo:dict|None, edad_s|None).  columnas(crudo) -> dict de columnas normalizadas de ESA
#  retina, incluyendo SIEMPRE 'intensidad' [0,1] y opcionalmente 'movimiento' [0,1].
# =========================================================================================
class _RetinaBase:
    nombre = "base"
    def iniciar(self): pass
    def detener(self): pass
    def leer(self): return None, None
    def columnas(self, crudo): return {"intensidad": 0.0}


class _RetinaCamaraESP32(_RetinaBase):
    """Retina visible espacial: ESP32-CAM por WiFi, en su propio hilo (timeout+watchdog). Aporta brillo,
    color, contraste y movimiento. Requiere Pillow+requests (solo en la Pi); imports perezosos.

    URL: si url es None/""/"auto", usa VST_CamaraESP32Locator (descubre en LAN + caché + re-rastreo).
    Si se pasa una URL fija, se usa como preferida pero ante fallos también re-descubre."""
    nombre = "cam"
    def __init__(self, url=None, periodo_s=0.5, thumb=(32,24),
                 timeout_s=2.0, configurar=True, escala_brillo=255.0, escala_mov=64.0, escala_contraste=128.0):
        self.url = url  # capture URL o None/auto
        self.periodo_s=float(periodo_s); self.thumb=thumb; self.timeout_s=float(timeout_s)
        self.configurar=configurar
        self.escala_brillo=escala_brillo; self.escala_mov=escala_mov; self.escala_contraste=escala_contraste
        self._crudo=None; self._t=None; self._prev=None
        self._lock=threading.Lock(); self._run=False; self._hilo=None
        self._locator = None
        try:
            from VST_CamaraESP32Locator import get_locator  # type: ignore
            self._locator = get_locator()
        except Exception:
            try:
                from organelos.VST_CamaraESP32Locator import get_locator  # type: ignore
                self._locator = get_locator()
            except Exception:
                self._locator = None

    def _capture_url(self) -> str:
        u = (self.url or "").strip()
        if u and u.lower() not in ("auto", "discover", "http://auto", "http://discover"):
            if not u.endswith("/capture"):
                u = u.rstrip("/") + "/capture"
            return u
        if self._locator is not None:
            return self._locator.capture_url() or ""
        return ""

    def _config(self):
        try:
            import requests
            cap = self._capture_url()
            if not cap:
                return
            base = cap.rsplit("/", 1)[0]
            for var, val in [("framesize", 5), ("aec", 0), ("agc", 0), ("awb", 0), ("aec2", 0), ("gainceiling", 0)]:
                try:
                    requests.get(f"{base}/control", params={"var": var, "val": val}, timeout=self.timeout_s)
                except Exception:
                    pass
        except Exception:
            pass

    def iniciar(self):
        # descubrir en background al arrancar (no bloquea el resto de organelos más de un momento)
        if self._locator is not None:
            try:
                self._locator.discover(force=False)
            except Exception:
                pass
        self._run = True
        self._hilo = threading.Thread(target=self._loop, daemon=True)
        self._hilo.start()

    def detener(self):
        self._run = False

    def _loop(self):
        try:
            import requests, io
            from PIL import Image
        except Exception:
            return
        configured_for = ""
        while self._run:
            t0 = time.time()
            try:
                cap = self._capture_url()
                if not cap and self._locator is not None:
                    self._locator.discover(force=True)
                    cap = self._capture_url()
                if not cap:
                    time.sleep(max(self.periodo_s, 1.0))
                    continue
                if self.configurar and configured_for != cap:
                    self.url = cap  # preferir la URL viva en siguientes vueltas
                    self._config()
                    configured_for = cap
                resp = requests.get(cap, timeout=self.timeout_s)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize(self.thumb)
                px = list(img.getdata())
                m = len(px)
                rgb = (sum(p[0] for p in px) / m, sum(p[1] for p in px) / m, sum(p[2] for p in px) / m)
                gris = [0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2] for p in px]
                crudo, self._prev = reducir_frame(gris, rgb, self._prev)
                with self._lock:
                    self._crudo = crudo
                    self._t = time.time()
                if self._locator is not None:
                    self._locator.note_success(cap)
            except Exception:
                if self._locator is not None:
                    self._locator.note_failure()
            dt = time.time() - t0
            if dt < self.periodo_s:
                time.sleep(self.periodo_s - dt)
    def leer(self):
        with self._lock:
            if self._crudo is None: return None,None
            return dict(self._crudo), (None if self._t is None else time.time()-self._t)
    def columnas(self, c):
        suma=c["r"]+c["g"]+c["b"]
        tr,tg,tb=(c["r"]/suma,c["g"]/suma,c["b"]/suma) if suma>1e-6 else (1/3,1/3,1/3)
        return {"intensidad":_c01(c["brillo"]/self.escala_brillo),
                "movimiento":_c01(c["movimiento"]/self.escala_mov),
                "cam_contraste":round(_c01(c["contraste"]/self.escala_contraste),4),
                "cam_tono_r":round(tr,4),"cam_tono_g":round(tg,4),"cam_tono_b":round(tb,4)}


class _RetinaPanelProtoOjo(_RetinaBase):
    """Retina no-espacial: el PROTO-OJO — el mismo panel del cloroplasto, leído como VISTA (su variación),
    no como alimento. Un solo fotorreceptor: intensidad, sin espacio. Consume v_fuente de la fila (que el
    cloroplasto ya recibe del ATmega). Es la mancha ocular de Euglena: el ojo naciendo del estómago."""
    nombre = "panel"
    def __init__(self, v_oscuro=0.15, v_pleno=4.20):
        self.v_oscuro=v_oscuro; self.v_pleno=v_pleno; self._fila=None
    def alimentar_fila(self, fila): self._fila=fila            # la corteza le pasa la fila del paso
    def leer(self):
        if self._fila is None or "v_fuente" not in self._fila or self._fila["v_fuente"] is None:
            return None,None
        return {"v_fuente":float(self._fila["v_fuente"])}, 0.0
    def columnas(self, c):
        x=(c["v_fuente"]-self.v_oscuro)/max(1e-6,(self.v_pleno-self.v_oscuro))
        return {"intensidad":_c01(x)}                          # sin movimiento: un solo receptor


# =========================================================================================
#  EL ÓRGANO — banco de retinas + corteza que las funde. Un solo órgano.
# =========================================================================================
class OrganoVisual:
    """El ojo: un banco de retinas en paralelo + corteza que extrae el sentido y lo funde. Apagado por
    defecto. retinas: lista de instancias _Retina* (cada una un 'ojo'). Vacía = consume vision_* de la fila."""

    def __init__(self, organismo_id=None, activo=False, retinas=None,
                 ema=0.20, w_novedad=0.6, w_movimiento=0.4, caducidad_s=5.0):
        self.organismo_id=organismo_id; self.activo=bool(activo)
        self.retinas=list(retinas) if retinas else []
        self.ema=_c01(ema); self.w_novedad=float(w_novedad); self.w_movimiento=float(w_movimiento)
        self.caducidad_s=float(caducidad_s)
        self._mem={}                    # memoria de adaptación por retina: nombre -> intensidad recordada
        self._t_prev=None; self.ultimo={}
        if self.activo:
            for r in self.retinas: r.iniciar()

    def columnas_declaradas(self):
        """Lista de columnas que este órgano emite, dado su banco de retinas (para registrar el schema)."""
        cols=["vis_saliencia","vis_dominante","vis_novedad","vis_vivo","vis_n_retinas"]
        for r in self.retinas:
            p=f"vis_{r.nombre}"
            cols += [f"{p}_intensidad", f"{p}_novedad", f"{p}_movimiento", f"{p}_saliencia"]
            if r.nombre=="cam":
                cols += ["vis_cam_contraste","vis_cam_tono_r","vis_cam_tono_g","vis_cam_tono_b"]
        return cols

    def observar(self, fila):
        if not self.activo:
            self.ultimo={c:None for c in self.columnas_declaradas()}; return self.ultimo
        t=_g(fila,"t",default=(self._t_prev or 0.0)+1.0); self._t_prev=t

        out={}; saliencias={}; novedades={}; vivas=0
        for r in self.retinas:
            # el proto-ojo lee de la fila; las retinas con hilo propio ya tienen su frame
            if hasattr(r,"alimentar_fila"): r.alimentar_fila(fila)
            crudo,edad=r.leer()
            p=f"vis_{r.nombre}"
            if crudo is None or (edad is not None and edad>self.caducidad_s):
                # retina caída: sus columnas a 0, su memoria se congela (no se corrompe)
                out[f"{p}_intensidad"]=0.0; out[f"{p}_novedad"]=0.0
                out[f"{p}_movimiento"]=0.0; out[f"{p}_saliencia"]=0.0
                if r.nombre=="cam":
                    out["vis_cam_contraste"]=0.0
                    out["vis_cam_tono_r"]=out["vis_cam_tono_g"]=out["vis_cam_tono_b"]=0.0
                continue
            vivas+=1
            col=r.columnas(crudo)
            inten=_c01(col.get("intensidad",0.0)); mov=_c01(col.get("movimiento",0.0))
            # NOVEDAD = desviación de la memoria de ESTA retina (adaptación por banda)
            if r.nombre not in self._mem:
                self._mem[r.nombre]=inten; nov=0.0
            else:
                nov=abs(inten-self._mem[r.nombre])
                self._mem[r.nombre]=(1-self.ema)*self._mem[r.nombre]+self.ema*inten
            sal=_c01(self.w_novedad*nov + self.w_movimiento*mov)
            out[f"{p}_intensidad"]=round(inten,4); out[f"{p}_novedad"]=round(nov,4)
            out[f"{p}_movimiento"]=round(mov,4); out[f"{p}_saliencia"]=round(sal,4)
            for k,v in col.items():
                if k.startswith("cam_"): out["vis_"+k]=v
            saliencias[r.nombre]=sal; novedades[r.nombre]=nov

        # --- FUSIÓN: saliencia global y banda DOMINANTE (hacia dónde mira el ojo; emergente) ---
        if saliencias:
            dominante=max(saliencias, key=saliencias.get)
            vis_sal=_c01(max(saliencias.values()))      # la banda más urgente gobierna la atención
            vis_nov=novedades.get(dominante,0.0)
        else:
            dominante="ninguna"; vis_sal=0.0; vis_nov=0.0
        out["vis_saliencia"]=round(vis_sal,4); out["vis_dominante"]=dominante
        out["vis_novedad"]=round(vis_nov,4); out["vis_vivo"]=1.0 if vivas>0 else 0.0
        out["vis_n_retinas"]=float(vivas)
        self.ultimo=out; return out

    def snapshot(self):
        return {"mem":dict(self._mem),"t_prev":self._t_prev,"activo":self.activo}
    def restore(self, snap):
        if not isinstance(snap,dict): return
        self._mem=dict(snap.get("mem",self._mem)); self._t_prev=snap.get("t_prev",self._t_prev)
        if "activo" in snap: self.activo=bool(snap["activo"])
    def cerrar(self):
        for r in self.retinas: r.detener()


# ---------------- auto-prueba: DOS retinas a la vez (cámara simulada + proto-ojo del panel) ----------
if __name__ == "__main__":
    # retina cámara en modo 'fila' (sin red): le inyectamos crudo a mano vía una retina de prueba
    class _RetinaCamPrueba(_RetinaBase):
        nombre="cam"; escala_brillo=255.0; escala_mov=64.0; escala_contraste=128.0
        def __init__(self): self._c=None
        def poner(self,b,mov,r,g,bl): self._c={"brillo":b,"movimiento":mov,"contraste":40.0,"r":r,"g":g,"b":bl}
        def leer(self): return (dict(self._c),0.0) if self._c else (None,None)
        columnas=_RetinaCamaraESP32.columnas
    cam=_RetinaCamPrueba(); panel=_RetinaPanelProtoOjo()
    ojo=OrganoVisual("E", activo=True, retinas=[cam,panel])

    print("=== DOS retinas en paralelo: cámara + proto-ojo(panel). ¿Quién domina la atención? ===")
    def paso(t, b, mov, vf):
        cam.poner(b,mov,130,130,130)
        return ojo.observar({"t":t,"v_fuente":vf})
    for t in range(6): d=paso(t,130,0, 2.0)                       # todo estable
    print(f"t=5  todo estable      dom={d['vis_dominante']:<6} sal={d['vis_saliencia']:.3f}  cam_nov={d['vis_cam_novedad']:.3f} panel_nov={d['vis_panel_novedad']:.3f}")
    d=paso(6,210,0, 2.0)                                          # cambio SÓLO en cámara (luz de escena)
    print(f"t=6  cambia la CÁMARA   dom={d['vis_dominante']:<6} sal={d['vis_saliencia']:.3f}  cam_nov={d['vis_cam_novedad']:.3f} panel_nov={d['vis_panel_novedad']:.3f}")
    for t in range(7,12): d=paso(t,210,0, 2.0)
    d=paso(12,210,0, 3.8)                                         # cambio SÓLO en panel (nube se abre)
    print(f"t=12 cambia el PANEL    dom={d['vis_dominante']:<6} sal={d['vis_saliencia']:.3f}  cam_nov={d['vis_cam_novedad']:.3f} panel_nov={d['vis_panel_novedad']:.3f}")
    d=paso(13,210,45, 3.8)                                        # movimiento en cámara
    print(f"t=13 MOVIMIENTO cámara  dom={d['vis_dominante']:<6} sal={d['vis_saliencia']:.3f}  cam_mov={d['vis_cam_movimiento']:.2f}")
    print(f"\nretinas vivas={d['vis_n_retinas']:.0f}  ·  columnas del órgano: {len(ojo.columnas_declaradas())}")
    print("→ la atención salta a la retina que trae la novedad; cada banda se adapta por separado.")
    print("  Un ojo, varias retinas, una atención que emerge de cuál banda importa AHORA.")
