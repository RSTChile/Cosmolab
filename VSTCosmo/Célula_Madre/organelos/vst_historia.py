#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_HISTORIA — BIOGRAFÍA LONGITUDINAL DEL ORGANISMO  (infraestructura, NO cerebro)
================================================================================
QUIÉN SOY
  Los organismos viven 24/7 en Docker. Yo dejo su BIOGRAFÍA en disco externo de forma
  permanente, organizada y analizable: fisiología continua, comunicación A↔B, voz,
  eventos, snapshots periódicos, e índice maestro.

  NO decido nada. NO toco RC/metabolismo/memoria/homeostasis/comunicación. Sólo REGISTRO.
  Escritura NO BLOQUEANTE: una cola + un hilo escritor; el lazo vivo nunca espera al disco.
  Si el disco falla, registro el error y sigo — jamás mato al organismo.

ESTRUCTURA (bajo VST_HISTORY_DIR):
  organismo_<id>/{fisiologia,comunicacion,voz,eventos,memoria,metabolismo,sesiones,snapshots}
  diada/{comunicacion,eventos,sincronias,conversaciones,sesiones,resumenes}
  index.jsonl

CONFIG (entorno):
  VST_HISTORY_DIR · VST_HISTORY_ENABLE · VST_HISTORY_FORMAT (csv) · VST_HISTORY_ROTATE_SECONDS
  VST_SNAPSHOT_SECONDS · VST_RECORD_VOICE_WAV · VST_RECORD_EXTERNAL_AUDIO · VST_LIFE_MODE
================================================================================
"""
from __future__ import annotations
import os, sys, json, time, threading, queue, csv as _csv, io


def _verdad(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _ahora():
    return time.time()


def _stamp(ts=None, fmt="%Y-%m-%d_%H-%M-%S"):
    return time.strftime(fmt, time.localtime(ts if ts is not None else _ahora()))


def _jsonsafe(o):
    """JSON-seguro para el REGISTRO (one-way): claves-tupla→str, numpy→nativo. Los organelos indexan
    por clave-tupla (valencia…) que json.dump no admite; aquí es para LEER, no restaurar."""
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): _jsonsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonsafe(x) for x in o]
    try:
        import numpy as _np
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return o


# Subcarpetas por organismo y de la díada (se crean al iniciar).
_SUB_ORG = ("fisiologia", "comunicacion", "voz", "eventos", "memoria", "metabolismo", "sesiones", "snapshots")
_SUB_DIADA = ("comunicacion", "eventos", "sincronias", "conversaciones", "sesiones", "resumenes")


class Historiador:
    """Escritor histórico de UN organismo (o de la díada con organismo_id='diada'). No bloqueante."""

    def __init__(self, organismo_id: str, base_dir: str | None = None):
        self.organismo_id = str(organismo_id)
        self.base_dir = base_dir or os.environ.get("VST_HISTORY_DIR", "/history")
        self.enable = _verdad(os.environ.get("VST_HISTORY_ENABLE", "true"))
        self.formato = (os.environ.get("VST_HISTORY_FORMAT", "csv") or "csv").lower()
        self.rotate_s = float(os.environ.get("VST_HISTORY_ROTATE_SECONDS", "3600"))
        self.modo_vida = os.environ.get("VST_LIFE_MODE", "continuous")
        self.rec_voz = _verdad(os.environ.get("VST_RECORD_VOICE_WAV", "true"))
        self.es_diada = (self.organismo_id == "diada")
        self.dir = os.path.join(self.base_dir, "diada" if self.es_diada else f"organismo_{self.organismo_id}")
        self._q: "queue.Queue" = queue.Queue(maxsize=20000)   # tope: no crece RAM sin límite
        self._fis_fh = None; self._fis_path = None; self._fis_cols = None; self._fis_bucket = None
        self._errores = 0; self._escritos = {"fila": 0, "evento": 0, "com": 0, "snapshot": 0, "voz": 0}
        self._stop = False
        if self.enable:
            self._crear_dirs()
            self._hilo = threading.Thread(target=self._writer_loop, daemon=True)
            self._hilo.start()

    # ----------------------------------------------------------------- dirs
    def _crear_dirs(self):
        try:
            subs = _SUB_DIADA if self.es_diada else _SUB_ORG
            for s in subs:
                os.makedirs(os.path.join(self.dir, s), exist_ok=True)
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception as e:
            sys.stderr.write(f"[historia] no pude crear dirs: {e}\n")

    # --------------------------------------------------------- API (no bloqueante)
    def registrar_fila(self, fila: dict, modo: str | None = None):
        """Una fila de fisiología (estado completo). NO bloquea: encola y sigue."""
        if not self.enable:
            return
        self._poner(("fila", dict(fila), modo or self.modo_vida))

    def evento(self, tipo: str, detalle: str = "", extra: dict | None = None, t_vida=None):
        if not self.enable:
            return
        ev = {"ts": round(_ahora(), 3), "stamp": _stamp(fmt="%Y-%m-%d %H:%M:%S"),
              "organismo": self.organismo_id, "tipo": tipo, "detalle": detalle, "t_vida": t_vida}
        if extra:
            ev["estado"] = extra
        self._poner(("evento", ev))

    def comunicacion(self, registro: dict):
        if not self.enable:
            return
        registro = dict(registro); registro.setdefault("ts", round(_ahora(), 3))
        self._poner(("com", registro))

    def voz(self, wav_bytes: bytes, meta: dict):
        if not self.enable or not self.rec_voz:
            return
        self._poner(("voz", wav_bytes, dict(meta)))

    def snapshot(self, estado: dict):
        """Snapshot completo (memoria/codebook/metabolismo/homeostasis/contadores). Bloqueante-ligero
        (lo encola; el hilo lo escribe). Actualiza el índice."""
        if not self.enable:
            return
        self._poner(("snapshot", dict(estado)))

    def stats(self) -> dict:
        return {"enable": self.enable, "dir": self.dir, "escritos": dict(self._escritos),
                "errores": self._errores, "cola": self._q.qsize(), "fisiologia": self._fis_path}

    # --------------------------------------------------------- interno
    def _poner(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._errores += 1   # cola llena → descartar (no bloquear el organismo)

    def _index(self, entry: dict):
        try:
            with open(os.path.join(self.base_dir, "index.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            self._errores += 1

    def _abrir_fisiologia(self, cols):
        """Abre (rotando) el CSV de fisiología por bucket de tiempo (VST_HISTORY_ROTATE_SECONDS)."""
        bucket = int(_ahora() // max(1.0, self.rotate_s))
        if self._fis_fh is not None and bucket == self._fis_bucket:
            return
        # cerrar el anterior + indexarlo
        if self._fis_fh is not None:
            try:
                self._fis_fh.flush(); self._fis_fh.close()
            except Exception:
                pass
            self._index({"ts": round(_ahora(), 3), "tipo_registro": "fisiologia",
                         "organismo": self.organismo_id, "archivo": self._fis_path,
                         "duracion_s": self.rotate_s, "columnas_principales": (cols or [])[:12]})
        self._fis_bucket = bucket
        nombre = f"fisiologia_{_stamp(fmt='%Y-%m-%d_%H-%M')}.csv"
        self._fis_path = os.path.join(self.dir, "fisiologia", nombre)
        nuevo = not os.path.exists(self._fis_path)
        self._fis_fh = open(self._fis_path, "a", encoding="utf-8", newline="")
        self._fis_cols = list(cols)
        if nuevo:
            self._fis_fh.write(",".join(self._fis_cols) + "\n")
            self._index({"ts": round(_ahora(), 3), "tipo_registro": "fisiologia", "organismo": self.organismo_id,
                         "archivo": self._fis_path, "estado": "abierto", "columnas_principales": (cols or [])[:12]})

    def _escribir_fila(self, fila, modo):
        fila = dict(fila)
        fila.setdefault("ts_real", round(_ahora(), 3))
        fila.setdefault("modo_vida", modo)
        # columnas = unión estable: las de la fila (preserva TODAS las columnas actuales) + ts_real/modo_vida al frente
        if self._fis_cols is None or not all(c in fila for c in self._fis_cols):
            base = ["ts_real", "modo_vida"] + [k for k in fila.keys() if k not in ("ts_real", "modo_vida")]
            self._fis_cols = base
        self._abrir_fisiologia(self._fis_cols)
        row = ",".join(str(fila.get(c, "")) for c in self._fis_cols)
        self._fis_fh.write(row + "\n")
        self._escritos["fila"] += 1

    def _append_jsonl(self, sub, nombre, obj):
        path = os.path.join(self.dir, sub, nombre)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonsafe(obj), ensure_ascii=False) + "\n")
        return path

    def _writer_loop(self):
        ult_flush = _ahora()
        while not self._stop:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                item = None
            try:
                if item is not None:
                    tag = item[0]
                    if tag == "fila":
                        self._escribir_fila(item[1], item[2])
                    elif tag == "evento":
                        self._append_jsonl("eventos", f"eventos_{_stamp(fmt='%Y-%m-%d')}.jsonl", item[1])
                        self._escritos["evento"] += 1
                    elif tag == "com":
                        sub = "comunicacion"
                        self._append_jsonl(sub, f"comunicacion_{_stamp(fmt='%Y-%m-%d')}.jsonl", item[1])
                        self._escritos["com"] += 1
                    elif tag == "snapshot":
                        nombre = f"snapshot_{_stamp(fmt='%Y-%m-%d_%H-%M')}.json"
                        path = os.path.join(self.dir, "snapshots", nombre)
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(_jsonsafe(item[1]), f, ensure_ascii=False, indent=2)
                        self._escritos["snapshot"] += 1
                        self._index({"ts": round(_ahora(), 3), "tipo_registro": "snapshot",
                                     "organismo": self.organismo_id, "archivo": path})
                    elif tag == "voz":
                        st = _stamp()
                        wpath = os.path.join(self.dir, "voz", f"voz_{self.organismo_id}_{st}.wav")
                        with open(wpath, "wb") as f:
                            f.write(item[1])
                        with open(os.path.join(self.dir, "voz", f"voz_{self.organismo_id}_{st}.json"), "w", encoding="utf-8") as f:
                            json.dump(item[2], f, ensure_ascii=False, indent=2)
                        self._escritos["voz"] += 1
            except Exception as e:
                self._errores += 1
                sys.stderr.write(f"[historia] error de escritura ({item[0] if item else '?'}): {e}\n")
            # flush periódico (no por cada línea)
            if _ahora() - ult_flush > 2.0:
                try:
                    if self._fis_fh is not None:
                        self._fis_fh.flush()
                except Exception:
                    pass
                ult_flush = _ahora()


# --------------------------- SELFTEST ---------------------------
def _selftest():
    import tempfile, glob
    d = os.path.join(tempfile.gettempdir(), "vst_historia_test")
    os.environ["VST_HISTORY_DIR"] = d
    os.environ["VST_HISTORY_ENABLE"] = "true"
    print("== SELFTEST Historiador →", d)
    h = Historiador("A")
    for i in range(5):
        h.registrar_fila({"t": i * 0.1, "OI": 0.1 * i, "voz_emitida": "chat"}, modo="basal")
    h.evento("docker_start", "arranque")
    h.comunicacion({"emisor": "A", "receptor": "B", "prototipo_codebook": "chat"})
    h.snapshot({"memoria": {"x": 1}, "edad": 123})
    h.voz(b"RIFF....fake", {"emisor": "A", "duracion": 1.0})
    time.sleep(2.0)
    print("  stats:", h.stats())
    archs = glob.glob(os.path.join(d, "**", "*"), recursive=True)
    print(f"  archivos creados: {len([a for a in archs if os.path.isfile(a)])}")
    for a in sorted(archs):
        if os.path.isfile(a):
            print("    ·", a.replace(d, ""))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
