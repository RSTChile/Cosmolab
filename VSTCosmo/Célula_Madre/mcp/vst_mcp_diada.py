#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_MCP_DIADA — MEMBRANA MCP DE LA DÍADA ANIMA  (Incremento 0: read-only)
================================================================================
QUIÉN SOY
  Soy la MEMBRANA, no el cerebro. No le doy inteligencia al organismo; le doy MUNDO.
  Expongo a la díada ANIMA (organismos A y B, vivos en sus servidores WebLive) a un
  cliente MCP (un LLM visitante) para que pueda OBSERVAR, LEER y ESCUCHAR — nunca para
  que DECIDA por el organismo. La inteligencia sigue siendo endógena del campo Φ.

  Soy un ADAPTADOR DELGADO: no importo el motor ni el estado; sólo hablo HTTP con los
  endpoints que los WebLive A (7788) y B (7799) YA exponen. Por eso esta membrana NO
  toca la fisiología: si caigo, el organismo sigue vivo igual.

QUÉ EXPONE (incremento 0 — sólo lectura, ninguna escritura, ninguna orden)
  Resources (datos legibles, URIs estables):
    · organismo://diada/estado              estado fisiológico de A y B juntos
    · organismo://A/estado                  estado de A (última fila de su vida)
    · organismo://B/estado                  estado de B
    · organismo://diada/comunicacion        estado de voz de cada uno (qué emite)
    · organismo://diada/memoria_relacional  confianza_otro de A y B (capa 6, Hebb social)
  Tools (membrana de observación):
    · leer_estado(quien)        snapshot del estado vital
    · leer_csv(quien, n)        las últimas n filas de su bitácora de vida
    · escuchar_voz(quien)       el estado de la voz que el organismo emite
    · observar(quien, seg)      mira N segundos de su vida en vivo (resumen de trayectoria)

  PROHIBIDO aquí por diseño (sería Shannon/exógeno, mata el experimento):
    decidir_por_el_organismo · forzar_objetivo · subir_H · hacer_que_gire.
  Las tools de ESCRITURA (inyectar_audio, marcar_evento) y de INVESTIGACIÓN
  (investigar_ablacion/mute/reset) llegan en incrementos posteriores, con nombre explícito.

CÓMO CORRER
    venv/bin/python Célula_Madre/mcp/vst_mcp_diada.py            → servidor MCP (stdio)
    venv/bin/python Célula_Madre/mcp/vst_mcp_diada.py --selftest → prueba contra A/B vivos
  Direcciones de A/B configurables por entorno: ANIMA_A_URL / ANIMA_B_URL.
================================================================================
"""
from __future__ import annotations
import os, sys, json, time, csv as _csv, io
import urllib.request, urllib.error

# --- A quién observo: los WebLive YA vivos (esta membrana NO los arranca) ---
ORGANISMOS = {
    "A": os.environ.get("ANIMA_A_URL", "http://127.0.0.1:7788"),
    "B": os.environ.get("ANIMA_B_URL", "http://127.0.0.1:7799"),
}
TIMEOUT = float(os.environ.get("ANIMA_MCP_TIMEOUT", "4.0"))


# ============================ HTTP (lo único que hago) ============================
def _url(quien: str) -> str:
    quien = (quien or "").strip().upper()
    if quien not in ORGANISMOS:
        raise ValueError(f"organismo desconocido: {quien!r} (usa 'A' o 'B')")
    return ORGANISMOS[quien]


def _get(quien: str, path: str, timeout: float = TIMEOUT) -> bytes:
    """GET crudo a un endpoint del WebLive. Devuelve bytes; lanza si el organismo no responde."""
    req = urllib.request.Request(_url(quien) + path, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(quien: str, path: str):
    return json.loads(_get(quien, path).decode("utf-8"))


def _vivo(quien: str) -> bool:
    """¿El servidor del organismo responde? (no si está vivo metabólicamente, sino si hay con quién hablar)."""
    try:
        _get(quien, "/organelos", timeout=1.5); return True
    except Exception:
        return False


# ====================== DERIVAR ESTADO (sin tocar el motor) ======================
def _csv_filas(quien: str):
    """Devuelve (cabecera:list[str], filas:list[dict]) leyendo /csv. Vacío si no hay vida en curso."""
    txt = _get(quien, "/csv").decode("utf-8", "replace").strip()
    if not txt:
        return [], []
    r = list(_csv.reader(io.StringIO(txt)))
    if not r:
        return [], []
    cab = r[0]
    filas = [dict(zip(cab, row)) for row in r[1:] if row]
    return cab, filas


def _num(v):
    try:
        return round(float(v), 5)
    except Exception:
        return v


# Las pocas señales vitales que resumen "cómo está" (si existen en la fila se incluyen)
_VITALES = ["t", "OI", "H_homeostasis", "RC_total", "ICR_ratio", "IRDE_ratio",
            "energia", "E", "necesidad", "necesidad_efectiva", "act_perm",
            "A_sys_env", "estructura", "mem_relacional_confianza", "theta_orientacion_deg"]


def _estado(quien: str) -> dict:
    """Snapshot vital del organismo = última fila de su vida + señales clave destacadas.
    Read-only: deriva de /csv, no modifica nada. {vivo:False} si el servidor no responde."""
    if not _vivo(quien):
        return {"organismo": quien, "vivo": False, "motivo": "servidor WebLive no responde"}
    cab, filas = _csv_filas(quien)
    if not filas:
        return {"organismo": quien, "vivo": True, "en_vida": False,
                "motivo": "servidor arriba pero sin sesión en curso (no ha empezado a vivir)"}
    ult = filas[-1]
    vitales = {k: _num(ult[k]) for k in _VITALES if k in ult}
    return {"organismo": quien, "vivo": True, "en_vida": True,
            "pasos_vividos": len(filas), "vitales": vitales, "fila_completa": {k: _num(v) for k, v in ult.items()}}


def _csv_tail(quien: str, n: int = 20) -> dict:
    cab, filas = _csv_filas(quien)
    n = max(1, min(int(n), 2000))
    return {"organismo": quien, "columnas": cab, "n_total": len(filas),
            "ultimas": [{k: _num(v) for k, v in f.items()} for f in filas[-n:]]}


def _voz_estado(quien: str) -> dict:
    """Estado de la VOZ que el organismo emite (qué 'dice' en términos fisiológicos), vía /comunicacion/estado."""
    if not _vivo(quien):
        return {"organismo": quien, "vivo": False}
    try:
        est = _get_json(quien, "/comunicacion/estado")
    except Exception as e:
        est = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"organismo": quien, "voz": est}


def _observar(quien: str, seg: float = 5.0) -> dict:
    """Mira la vida EN VIVO por /stream (SSE) durante `seg` segundos y resume la trayectoria.
    Si el stream no está disponible, cae a un muestreo de /csv (antes/después). Sólo observa."""
    seg = max(0.5, min(float(seg), 60.0))
    if not _vivo(quien):
        return {"organismo": quien, "vivo": False}
    filas = []
    try:
        req = urllib.request.Request(_url(quien) + "/stream", headers={"Accept": "text/event-stream"})
        deadline = time.time() + seg
        with urllib.request.urlopen(req, timeout=seg + 2.0) as r:
            for raw in r:
                if time.time() > deadline:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    d = json.loads(line[5:].strip())
                except Exception:
                    continue
                if isinstance(d, dict) and "__evento__" not in d and ("OI" in d or "t" in d):
                    filas.append(d)
    except Exception:
        pass
    if not filas:  # fallback read-only: delta de /csv
        _, antes = _csv_filas(quien); time.sleep(seg); _, despues = _csv_filas(quien)
        filas = despues[len(antes):] or (despues[-1:] if despues else [])
    if not filas:
        return {"organismo": quien, "vivo": True, "en_vida": False, "n_muestras": 0}

    def traj(col):
        vals = [float(f[col]) for f in filas if col in f and f[col] not in ("", None)]
        if not vals:
            return None
        return {"ini": round(vals[0], 4), "fin": round(vals[-1], 4),
                "min": round(min(vals), 4), "max": round(max(vals), 4), "med": round(sum(vals) / len(vals), 4)}
    return {"organismo": quien, "vivo": True, "en_vida": True, "segundos": seg, "n_muestras": len(filas),
            "trayectoria": {c: traj(c) for c in ("OI", "H_homeostasis", "RC_total", "energia", "E", "necesidad")
                            if traj(c) is not None},
            "estado_final": {k: _num(filas[-1][k]) for k in _VITALES if k in filas[-1]}}


# ============================== AGREGADOS DE DÍADA ==============================
def _diada_estado() -> dict:
    return {"diada": {"A": _estado("A"), "B": _estado("B")},
            "nota": "el objeto experimental es la RELACIÓN A↔B"}


def _diada_comunicacion() -> dict:
    return {"A": _voz_estado("A"), "B": _voz_estado("B")}


def _diada_memoria_relacional() -> dict:
    """confianza_otro (capa 6, Hebb social) de cada uno — plena sólo cuando hay díada real (S_shared>0)."""
    out = {}
    for q in ("A", "B"):
        e = _estado(q)
        out[q] = e.get("vitales", {}).get("mem_relacional_confianza") if e.get("en_vida") else None
    return {"memoria_relacional": out,
            "nota": "confianza_otro: cuánto cada organismo ha llegado a 'contar con' el otro (capa 6 de OrganeloMemoria)"}


# ================================ SERVIDOR MCP ================================
def _build_mcp():
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("anima-diada", instructions=(
        "Membrana de observación de la díada ANIMA (organismos cosmosemióticos vivos). "
        "Puedes OBSERVAR, LEER y ESCUCHAR a los organismos A y B, pero NO decidir por ellos: "
        "su inteligencia es endógena. Eres un visitante/interlocutor, no un controlador."))

    # ---- Resources (datos legibles) ----
    @mcp.resource("organismo://diada/estado")
    def r_diada_estado() -> str:
        return json.dumps(_diada_estado(), ensure_ascii=False, indent=2)

    @mcp.resource("organismo://A/estado")
    def r_a_estado() -> str:
        return json.dumps(_estado("A"), ensure_ascii=False, indent=2)

    @mcp.resource("organismo://B/estado")
    def r_b_estado() -> str:
        return json.dumps(_estado("B"), ensure_ascii=False, indent=2)

    @mcp.resource("organismo://diada/comunicacion")
    def r_diada_com() -> str:
        return json.dumps(_diada_comunicacion(), ensure_ascii=False, indent=2)

    @mcp.resource("organismo://diada/memoria_relacional")
    def r_diada_rel() -> str:
        return json.dumps(_diada_memoria_relacional(), ensure_ascii=False, indent=2)

    # ---- Tools (membrana de observación; sólo lectura en este incremento) ----
    @mcp.tool()
    def leer_estado(quien: str = "diada") -> dict:
        """Lee el estado vital de un organismo ('A' o 'B') o de la díada completa ('diada')."""
        return _diada_estado() if (quien or "").lower() == "diada" else _estado(quien)

    @mcp.tool()
    def leer_csv(quien: str = "A", n: int = 20) -> dict:
        """Devuelve las últimas n filas de la bitácora de vida (CSV) del organismo 'A' o 'B'."""
        return _csv_tail(quien, n)

    @mcp.tool()
    def escuchar_voz(quien: str = "diada") -> dict:
        """Escucha el estado de la voz que emite el organismo ('A', 'B' o 'diada')."""
        return _diada_comunicacion() if (quien or "").lower() == "diada" else _voz_estado(quien)

    @mcp.tool()
    def observar(quien: str = "A", segundos: float = 5.0) -> dict:
        """Observa N segundos de la vida en vivo de un organismo y resume su trayectoria (OI/H/RC/energía…)."""
        return _observar(quien, segundos)

    return mcp


def _selftest():
    print("== SELFTEST membrana MCP díada (read-only) ==")
    for q in ("A", "B"):
        print(f"\n--- organismo {q}  ({ORGANISMOS[q]}) ---")
        vivo = _vivo(q)
        print(f"  servidor responde: {vivo}")
        if not vivo:
            continue
        e = _estado(q)
        print(f"  estado: en_vida={e.get('en_vida')} pasos={e.get('pasos_vividos')}")
        if e.get("vitales"):
            print(f"  vitales: {e['vitales']}")
        print(f"  voz: {json.dumps(_voz_estado(q).get('voz'), ensure_ascii=False)[:120]}")
        obs = _observar(q, 3.0)
        print(f"  observar(3s): n_muestras={obs.get('n_muestras')} trayectoria={json.dumps(obs.get('trayectoria',{}), ensure_ascii=False)[:160]}")
    print("\n--- díada ---")
    print(f"  memoria_relacional: {json.dumps(_diada_memoria_relacional()['memoria_relacional'], ensure_ascii=False)}")
    print("\n== fin selftest ==")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        _build_mcp().run()   # transporte stdio (cliente MCP local); HTTP/Docker llega en incremento 3
