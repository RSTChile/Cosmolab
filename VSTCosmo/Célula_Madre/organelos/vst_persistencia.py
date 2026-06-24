#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_PERSISTENCIA — EL ESPACIO EN DISCO DEL ORGANISMO  (Incremento 1)
================================================================================
QUIÉN SOY
  Soy el coordinador que hace que la HISTORIA del organismo sobreviva al apagón.
  "Sin memoria no hay futuro": hoy cada arranque hace MEMORIA.reset() y el organismo
  nace de cero. Yo recojo el snapshot() de cada organelo, lo escribo a disco de forma
  ATÓMICA, y al despertar lo reparto con restore() → el organismo continúa desde donde
  quedó, recordando lo que aprendió.

  QUÉ PERSISTE (la historia, no el instante): codebook (repertorio aprendido), memoria
  (valencias, episodios, vínculo con el otro, vida vivida), metabolismo (reservas E,
  preferencias). QUÉ NO: el campo Φ — se re-forma al despertar (el cuerpo vuelve a su
  ritmo; lo que persiste es lo aprendido, no el estado instantáneo).

  Cada organelo es responsable de su propio snapshot()/restore() (encapsulación). Yo sólo
  los coordino y manejo el disco. No sé NADA de la fisiología.

DÓNDE GUARDA
  Directorio configurable por ANIMA_ESTADO_DIR (en Docker = el VOLUMEN montado, p.ej. /data).
  Por defecto Célula_Madre/estado_persistente/. Cada organismo: anima_<ID>/estado.json.

CÓMO CORRER
    venv/bin/python Célula_Madre/organelos/vst_persistencia.py --selftest
================================================================================
"""
from __future__ import annotations
import os, sys, json, tempfile, time

VERSION = 1
DEFAULT_DIR = os.environ.get(
    "ANIMA_ESTADO_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "estado_persistente"),
)


def _dir_organismo(organismo_id: str, base_dir: str) -> str:
    return os.path.join(base_dir, f"anima_{organismo_id}")


# --- Codificación reversible: los organelos indexan por clave-TUPLA (p.ej. (lb,ab) de la situación),
# y JSON no admite dicts con claves no-string. Preservamos tuplas (como CLAVE y como VALOR) con una
# marca, para que la historia round-trippee EXACTA. Transparente para los organelos. ---
def _encode(o):
    if isinstance(o, tuple):
        return {"__t__": [_encode(x) for x in o]}
    if isinstance(o, dict):
        if all(isinstance(k, str) for k in o) and "__t__" not in o and "__kv__" not in o:
            return {k: _encode(v) for k, v in o.items()}
        return {"__kv__": [[_encode(k), _encode(v)] for k, v in o.items()]}   # claves no-string
    if isinstance(o, list):
        return [_encode(x) for x in o]
    return o


def _decode(o):
    if isinstance(o, dict):
        if len(o) == 1 and "__t__" in o:
            return tuple(_decode(x) for x in o["__t__"])
        if len(o) == 1 and "__kv__" in o:
            return {_decode(k): _decode(v) for k, v in o["__kv__"]}
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(x) for x in o]
    return o


def guardar(organismo_id: str, organelos: dict, base_dir: str = DEFAULT_DIR,
            extra: dict | None = None) -> str:
    """Recoge el snapshot() de cada organelo que lo tenga y lo escribe a disco ATÓMICAMENTE
    (tmp + os.replace → nunca un archivo a medias, aunque se apague el Mac en medio).
    organelos: dict nombre->objeto. Devuelve la ruta escrita."""
    estado = {nombre: o.snapshot() for nombre, o in organelos.items() if hasattr(o, "snapshot")}
    blob = {"organismo": organismo_id, "version": VERSION, "guardado_ts": time.time(),
            "organelos": list(estado.keys()), "estado": estado}
    if extra:
        blob["extra"] = extra
    dest = _dir_organismo(organismo_id, base_dir)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "estado.json")
    fd, tmp = tempfile.mkstemp(prefix=".estado_", suffix=".tmp", dir=dest)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_encode(blob), f, ensure_ascii=False)   # _encode → tolera claves-tupla de los organelos
            f.flush(); os.fsync(f.fileno())            # asegura el bytes-en-disco antes de reemplazar
        os.replace(tmp, path)                           # atómico: o el viejo entero, o el nuevo entero
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def cargar(organismo_id: str, base_dir: str = DEFAULT_DIR) -> dict | None:
    """Lee el estado persistido del organismo. Devuelve el blob (dict) o None si no hay nada guardado
    (= organismo que nace por primera vez). Tolerante a archivo corrupto: devuelve None, no rompe."""
    path = os.path.join(_dir_organismo(organismo_id, base_dir), "estado.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            blob = _decode(json.load(f))                # _decode → restaura las claves-tupla exactas
        if isinstance(blob, dict) and "estado" in blob:
            return blob
    except Exception:
        pass
    return None


def restaurar(organelos: dict, blob: dict | None) -> list:
    """Reparte el estado guardado a cada organelo (restore()). Devuelve los nombres restaurados.
    Si blob es None (organismo nuevo) o falta un organelo, simplemente no restaura ese → nace de cero."""
    if not blob:
        return []
    estado = blob.get("estado", {})
    restaurados = []
    for nombre, o in organelos.items():
        if nombre in estado and hasattr(o, "restore"):
            try:
                o.restore(estado[nombre]); restaurados.append(nombre)
            except Exception as e:
                sys.stderr.write(f"[persistencia] no pude restaurar {nombre}: {e}\n")
    return restaurados


# ============================== SELFTEST (round-trip) ==============================
def _selftest():
    # rutas a los órganos (este módulo vive en organelos/; el soma en campo/)
    AQUI = os.path.dirname(os.path.abspath(__file__))
    CM = os.path.dirname(AQUI)
    for d in ("organelos", "campo", "genoma"):
        p = os.path.join(CM, d)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import importlib.util as u
    from VST_Memoria import OrganeloMemoria
    from VST_Metabolismo import OrganeloMetabolismo
    spec = u.spec_from_file_location("cmf", os.path.join(CM, "campo", "Célula_Madre_Funcional_001.py"))
    cmf = u.module_from_spec(spec); spec.loader.exec_module(cmf)

    print("== SELFTEST persistencia (round-trip: aprender → guardar → apagar → despertar) ==")
    test_dir = os.path.join(tempfile.gettempdir(), "anima_estado_test")

    # 1) un organismo que YA VIVIÓ algo (le ponemos historia a mano)
    mem = OrganeloMemoria(); met = OrganeloMetabolismo()
    soma = cmf.OrganeloSoma(np.zeros(4096), binaural=False)
    mem.valencia = {(-1, 2): {"corto": 0.3, "largo": 0.81}}   # CLAVE-TUPLA real (situación)
    mem.n_visitas = {(-1, 2): 5, (1, 0): 2}
    mem.episodios = [{"t": 12.0, "clave": (-1, 2), "tipo": 1, "intensidad": 0.7}]
    mem.confianza_otro = 0.42; mem.t = 137.0
    met.E = 0.66; met.preferencia = {(1, 1): 0.2}             # CLAVE-TUPLA real (alimento)
    soma._codebook = np.arange(soma._K * soma._n_bands, dtype=np.float64).reshape(soma._codebook.shape) / 10.0
    cb_orig = soma._codebook.copy()

    # 2) guardar (apagón) y construir un organismo NUEVO (de cero)
    ruta = guardar("TEST", {"memoria": mem, "metabolismo": met, "soma": soma}, base_dir=test_dir)
    print(f"  guardado en: {ruta}")
    mem2 = OrganeloMemoria(); met2 = OrganeloMetabolismo()
    soma2 = cmf.OrganeloSoma(np.zeros(4096), binaural=False)
    print(f"  organismo NUEVO antes de despertar: valencia={mem2.valencia} E={met2.E} codebook_sum={soma2._codebook.sum():.1f}")

    # 3) despertar: restaurar desde disco
    blob = cargar("TEST", base_dir=test_dir)
    rest = restaurar({"memoria": mem2, "metabolismo": met2, "soma": soma2}, blob)
    print(f"  restaurados: {rest}")

    # 4) verificar identidad de la historia
    ok = []
    ok.append(("valencia (clave-tupla)", mem2.valencia == mem.valencia))
    ok.append(("n_visitas (clave-tupla)", mem2.n_visitas == mem.n_visitas))
    ok.append(("episodios", mem2.episodios == mem.episodios))
    ok.append(("preferencia (clave-tupla)", met2.preferencia == met.preferencia))
    ok.append(("confianza_otro", abs(mem2.confianza_otro - 0.42) < 1e-9))
    ok.append(("vida_t", abs(mem2.t - 137.0) < 1e-9))
    ok.append(("E_metabolica", abs(met2.E - 0.66) < 1e-9))
    ok.append(("preferencia", met2.preferencia == met.preferencia))
    ok.append(("codebook", bool(np.allclose(soma2._codebook, cb_orig))))
    print("  ---- verificación ----")
    for nombre, passed in ok:
        print(f"    {'✓' if passed else '✗ FALLA'}  {nombre}")
    total = sum(1 for _, p in ok if p)
    print(f"  RESUMEN: {total}/{len(ok)} {'PASS — la historia sobrevive al apagón' if total == len(ok) else 'FALLA'}")
    return total == len(ok)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    else:
        print(__doc__)
