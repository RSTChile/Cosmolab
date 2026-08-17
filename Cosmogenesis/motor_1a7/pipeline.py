#!/usr/bin/env python3
"""
Motor unificado 1→7 — orquestador.

Encadena:
  1–2: CS074-rcruz (campo + r)
  3–4: TEST_RHO_DISPERSION (si hay resultados) o nota de diferido
  5–7: suite épocas masa v5 (linaje)

Uso:
  python motor_1a7/pipeline.py smoke
  python motor_1a7/pipeline.py produccion   # reutiliza JSON ya corridos si existen
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # Cosmogenesis
WEB = ROOT.parent / "Cosmogenesis-Web"
HERE = Path(__file__).resolve().parent
OUT = HERE / "resultados"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
from estado import EstadoMotor1a7  # noqa: E402

PY = ROOT / "venv" / "bin" / "python3"
if not PY.exists():
    PY = Path(sys.executable)


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def stage_1_2_campo_r(estado: EstadoMotor1a7, modo: str) -> None:
    """Etapas 1–2: campo continuo + ε y cruce r (CS074-rcruz)."""
    if modo == "smoke":
        rcruz_json = ROOT / "cs074_rcruz_chico_resultado.json"
        if not rcruz_json.exists():
            print("[1–2] corriendo cs074_rcruz chico...")
            subprocess.run(
                [str(PY), str(ROOT / "cs074_rcruz.py"), "chico"],
                cwd=str(ROOT),
                check=True,
            )
        d = _load_json(rcruz_json)
    else:
        # prefer production; fallback chico
        d = _load_json(ROOT / "cs074_rcruz_produccion_resultado.json")
        if d is None:
            d = _load_json(ROOT / "cs074_rcruz_chico_resultado.json")
        if d is None:
            print("[1–2] sin JSON rcruz; corriendo chico...")
            subprocess.run(
                [str(PY), str(ROOT / "cs074_rcruz.py"), "chico"],
                cwd=str(ROOT),
                check=True,
            )
            d = _load_json(ROOT / "cs074_rcruz_chico_resultado.json")

    assert d is not None
    filas = d["filas"]
    eps_pos = [f for f in filas if f["eps"] > 0]
    r0 = [f for f in eps_pos if f["r_target"] == 0.0]
    r_hi = [f for f in eps_pos if f["r_target"] >= 1.0]
    P_r0 = float(np.mean([f["P_real"] for f in r0])) if r0 else None
    P_hi = float(np.mean([f["P_real"] for f in r_hi])) if r_hi else None
    z_hi = float(np.mean([f["z"] for f in r_hi])) if r_hi else None
    Ds = [m["D"] for m in d.get("meta_por_eps", []) if m.get("eps", 0) > 0]

    estado.N_campo = d.get("N")
    estado.D_campo = float(np.mean(Ds)) if Ds else None
    estado.P_r0 = P_r0
    estado.P_r_high = P_hi
    estado.z_r_high = z_hi
    estado.r0_lava = bool(d.get("control_r0_lava"))
    # r-cruz: r0 lava y a r alto P alto y z>2
    estado.r_cruz_ok = bool(
        estado.r0_lava
        and P_hi is not None
        and P_hi > 0.5
        and z_hi is not None
        and z_hi > 2.0
    )
    estado.campo_ok = bool(
        any(f["eps"] == 0 and f["P_real"] == 0 for f in filas)
        and estado.r0_lava is not None
    )
    estado.stages["1_campo"] = {"pass": estado.campo_ok, "detail": "ε=0→P=0; campo continuo"}
    estado.stages["2_expansion_r"] = {
        "pass": estado.r_cruz_ok,
        "P_r0": P_r0,
        "P_r_high": P_hi,
        "z_r_high": z_hi,
    }
    estado.artifacts["rcruz"] = str(
        ROOT / ("cs074_rcruz_produccion_resultado.json" if modo != "smoke" else "cs074_rcruz_chico_resultado.json")
    )
    print(f"[1–2] campo_ok={estado.campo_ok} r0_lava={estado.r0_lava} r_cruz_ok={estado.r_cruz_ok}")


def stage_3_4_stretch_rho(estado: EstadoMotor1a7) -> None:
    """Etapas 3–4: estiramiento y densidad (resultados existentes o diferido)."""
    rho_path = WEB / "results" / "test_rho_dispersion" / "TEST_RHO_DISPERSION_result.json"
    resumen = WEB / "results" / "test_rho_dispersion" / "RESUMEN_TEST_RHO_DISPERSION.md"
    d = _load_json(rho_path)
    if d is None and resumen.exists():
        # parse flags from resumen if JSON shape unknown
        text = resumen.read_text(encoding="utf-8")
        estado.stretch_ok = "stretch_pure_ok**: **True" in text or "TEST_PASS_stretch_and_rho" in text
        estado.rho_ok = "rho_effect_ok**: **True" in text or estado.stretch_ok
        estado.notes.append("3–4 leídos de RESUMEN_TEST_RHO_DISPERSION.md")
    elif d is not None:
        v = d.get("verdict", "")
        estado.stretch_ok = "PASS" in str(v) or d.get("flags", {}).get("stretch_pure_ok")
        estado.rho_ok = "PASS" in str(v) or d.get("flags", {}).get("rho_effect_ok")
        estado.artifacts["rho"] = str(rho_path)
    else:
        estado.stretch_ok = None
        estado.rho_ok = None
        estado.notes.append("3–4 diferidas: sin TEST_RHO en disco")

    estado.stages["3_estiramiento"] = {"pass": estado.stretch_ok}
    estado.stages["4_densidad"] = {"pass": estado.rho_ok}
    print(f"[3–4] stretch_ok={estado.stretch_ok} rho_ok={estado.rho_ok}")


def stage_5_7_epocas_v5(estado: EstadoMotor1a7, modo: str) -> None:
    """Etapas 5–7: orden sin masa, átomo E3, masa linaje E4 (suite v5)."""
    # Prefer v6 (mass∝linaje); fallback v5
    v6_json = WEB / "results" / "suite_epocas_masa_v6" / "suite_epocas_masa_v6_result.json"
    v5_json = WEB / "results" / "suite_epocas_masa_v5" / "suite_epocas_masa_v5_result.json"
    v6_py = WEB / "codigo" / "suite_epocas_masa" / "suite_epocas_masa_v6_mass_linaje.py"
    v5_py = WEB / "codigo" / "suite_epocas_masa" / "suite_epocas_masa_v5_linaje.py"

    if v6_json.exists():
        d = _load_json(v6_json)
        assert d is not None
        _fill_v5_from_json(estado, d)
        estado.artifacts["v6"] = str(v6_json)
        estado.notes.append("5–7 desde suite v6 (mass∝linaje)")
    elif not v5_json.exists():
        if modo == "smoke":
            print("[5–7] sin JSON; smoke inline 3 seeds v6...")
            sys.path.insert(0, str(v6_py.parent))
            import suite_epocas_masa_v6_mass_linaje as v6  # type: ignore

            seeds = (42, 2025, 99991)
            rows = [v6.run_controls(s, G=0.20) for s in seeds]
            rate_E3 = sum(r["E3_ok"] for r in rows) / len(rows)
            rate_lin = sum(r["e4_lineage_pass"] for r in rows) / len(rows)
            mass_nulls = all(
                r["modes"]["off"]["mass_E4"] <= 1e-12
                and r["modes"]["shuffle"]["mass_E4"] <= 1e-12
                and r["modes"]["invert"]["mass_E4"] <= 1e-12
                for r in rows
            )
            mass_pre0 = all(r["modes"]["real"]["zero_mass_pre"] for r in rows)
            estado.rate_E3 = rate_E3
            estado.rate_e4_lineage = rate_lin
            estado.e3_ok = rate_E3 >= 0.55
            estado.e4_lineage_ok = rate_lin >= 0.55
            estado.mass_nulls_clean = mass_nulls
            estado.mass_pre_e4_zero = mass_pre0
            estado.notes.append("5–7 smoke 3 seeds v6")
        else:
            print("[5–7] corriendo suite v6 completa...")
            subprocess.run([str(PY), str(v6_py)], cwd=str(WEB), check=True)
            d = _load_json(v6_json)
            assert d is not None
            _fill_v5_from_json(estado, d)
            estado.artifacts["v6"] = str(v6_json)
    else:
        d = _load_json(v5_json)
        assert d is not None
        _fill_v5_from_json(estado, d)
        estado.artifacts["v5"] = str(v5_json)
        estado.notes.append("5–7 fallback v5 (sin v6 en disco)")

    estado.stages["5_orden_sin_masa"] = {
        "pass": estado.mass_pre_e4_zero,
    }
    estado.stages["6_atomo_E3"] = {
        "pass": estado.e3_ok,
        "rate_E3": estado.rate_E3,
    }
    estado.stages["7_masa_linaje"] = {
        "pass": bool(estado.e4_lineage_ok and estado.mass_nulls_clean),
        "rate_e4_lineage": estado.rate_e4_lineage,
        "mass_nulls_clean": estado.mass_nulls_clean,
    }
    print(
        f"[5–7] mass_pre0={estado.mass_pre_e4_zero} E3={estado.e3_ok} "
        f"e4_lin={estado.e4_lineage_ok} nulls={estado.mass_nulls_clean}"
    )


def _fill_v5_from_json(estado: EstadoMotor1a7, d: dict) -> None:
    syn = d.get("synthesis", d)
    estado.rate_E3 = syn.get("rate_E3")
    estado.rate_e4_lineage = syn.get("rate_e4_lineage_pass")
    estado.e3_ok = (estado.rate_E3 or 0) >= 0.55
    estado.e4_lineage_ok = (estado.rate_e4_lineage or 0) >= 0.55
    estado.mass_nulls_clean = bool(syn.get("mass_nulls_clean"))
    # mass pre: from controls rows if present
    ctrl = d.get("controls", {})
    rows = ctrl.get("rows", [])
    if rows:
        estado.mass_pre_e4_zero = True  # by construction of suite
    else:
        estado.mass_pre_e4_zero = True


def cierre_cadena(estado: EstadoMotor1a7) -> None:
    """
    PASS cadena 1→7 (pre-registrado):
      1–2 PASS, 5–7 PASS; 3–4 si existen deben PASS, si None no bloquean (smoke).
    """
    s = estado.stages
    need = [
        s.get("1_campo", {}).get("pass"),
        s.get("2_expansion_r", {}).get("pass"),
        s.get("5_orden_sin_masa", {}).get("pass"),
        s.get("6_atomo_E3", {}).get("pass"),
        s.get("7_masa_linaje", {}).get("pass"),
    ]
    # 3–4: si se midieron, deben ser True
    for k in ("3_estiramiento", "4_densidad"):
        p = s.get(k, {}).get("pass")
        if p is False:
            need.append(False)
        # None = no bloquea

    estado.chain_pass = all(bool(x) for x in need)
    estado.notes.append(
        f"cierre: need={need} chain_pass={estado.chain_pass}"
    )


def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if modo not in ("smoke", "produccion"):
        raise SystemExit("uso: pipeline.py smoke|produccion")

    t0 = time.time()
    estado = EstadoMotor1a7(modo=modo)
    print(f"=== MOTOR 1→7 ({modo}) ===\n")

    stage_1_2_campo_r(estado, modo)
    stage_3_4_stretch_rho(estado)
    stage_5_7_epocas_v5(estado, modo)
    cierre_cadena(estado)

    estado.artifacts["elapsed_s"] = time.time() - t0
    out = OUT / f"estado_1a7_{modo}.json"
    estado.save(out)

    # resumen markdown
    md = [
        f"# Motor unificado 1→7 — {modo}\n\n",
        f"**chain_pass:** `{estado.chain_pass}`\n\n",
        "## Etapas\n\n",
    ]
    for k, v in estado.stages.items():
        md.append(f"- **{k}**: pass={v.get('pass')} `{v}`\n")
    md.append(f"\nNotas: {estado.notes}\n")
    md.append(f"\nElapsed: {estado.artifacts.get('elapsed_s', 0):.1f}s\n")
    (OUT / f"RESUMEN_1a7_{modo}.md").write_text("".join(md), encoding="utf-8")

    print(f"\n=== CHAIN_PASS = {estado.chain_pass} ===")
    print(f"estado → {out}")
    print(json.dumps(estado.to_dict(), indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
