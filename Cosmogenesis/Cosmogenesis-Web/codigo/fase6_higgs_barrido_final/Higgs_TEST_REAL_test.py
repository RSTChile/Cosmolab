"""
Higgs_TEST_REAL — test de verdad del mecanismo tipo Higgs

Pregunta pre-registrada:
  ¿Un campo de medio Phi con VEV no nulo y UNA sola ley de acoplamiento
  produce masas de dominio distintas de la geometría pura (~1/3),
  de modo que el NULL (m sin Phi) colapse a geometría?

Éxito ≠ 1/1836.
Éxito = VEV vivo + separación REAL vs NULL + (opcional) Rm estructurado.

Constantes FIJAS (sello). No retocar tras ver resultados.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

# --- sello ---
L = 30
PASOS = 400
H_TOPO = 0.01
SEED = 2025
Y0 = 0.3
R0 = 2.0
U = 0.5
TC = 0.55  # SSB si Tnorm < TC
D_PHI = 0.05
DT_PHI = 0.08
SIGMA0 = 0.08  # ruido ~ sqrt(Tnorm)
VEV_THR = 0.15
SEP_THR = 0.08
NULL_GEO_TOL = 0.08
HIER_THR = 0.10


def clusters_of(phi, ar, ad, Phi):
    media = phi.mean()
    visto = np.zeros_like(phi, dtype=bool)
    out = []
    n = phi.shape[0]
    for y in range(n):
        for x in range(n):
            if visto[y, x]:
                continue
            q = deque([(y, x)])
            visto[y, x] = True
            nodes = [(y, x)]
            lado = phi[y, x] >= media
            sum_rho = float(phi[y, x])
            sum_abs = float(abs(Phi[y, x]))
            perim = 0
            while q:
                cy, cx = q.popleft()
                if (not ar[cy, cx]) or (phi[cy, (cx + 1) % n] >= media) != lado:
                    perim += 1
                if (not ar[cy, (cx - 1) % n]) or (phi[cy, (cx - 1) % n] >= media) != lado:
                    perim += 1
                if (not ad[cy, cx]) or (phi[(cy + 1) % n, cx] >= media) != lado:
                    perim += 1
                if (not ad[(cy - 1) % n, cx]) or (phi[(cy - 1) % n, cx] >= media) != lado:
                    perim += 1
                for ny, nx, cond in (
                    (cy, (cx + 1) % n, ar[cy, cx]),
                    (cy, (cx - 1) % n, ar[cy, (cx - 1) % n]),
                    ((cy + 1) % n, cx, ad[cy, cx]),
                    ((cy - 1) % n, cx, ad[(cy - 1) % n, cx]),
                ):
                    if cond and not visto[ny, nx] and (phi[ny, nx] >= media) == lado:
                        visto[ny, nx] = True
                        q.append((ny, nx))
                        nodes.append((ny, nx))
                        sum_rho += float(phi[ny, nx])
                        sum_abs += float(abs(Phi[ny, nx]))
            k = len(nodes)
            v = sum_abs / k if k else 0.0
            out.append(
                {"k": k, "perim": perim, "sum_rho": sum_rho, "v_phi": v, "nodes": nodes}
            )
    return out


def mass_ratio(clusters, y0, use_phi_field: bool):
    """Fórmula ÚNICA: m = y0 * factor * sum_rho.
    REAL: factor = < |Phi| >_dom
    NULL geométrico: factor = 1  (solo tamaño)
    """
    m1, m3 = [], []
    for c in clusters:
        factor = c["v_phi"] if use_phi_field else 1.0
        m = y0 * factor * c["sum_rho"]
        if c["k"] == 1:
            m1.append(m)
        if c["k"] == 3 and c["perim"] == 8:
            m3.append(m)
    if not m1 or not m3:
        return None, 0, 0, 0.0, 0.0
    mk1, mk3 = float(np.mean(m1)), float(np.mean(m3))
    v1 = float(np.mean([c["v_phi"] for c in clusters if c["k"] == 1]))
    v3 = float(
        np.mean([c["v_phi"] for c in clusters if c["k"] == 3 and c["perim"] == 8])
    )
    return mk1 / (mk3 + 1e-30), len(m1), len(m3), v1, v3


def run():
    rng = np.random.default_rng(SEED)
    phi = np.ones((L, L)) + 0.3 * rng.normal(size=(L, L))
    # semilla fuera de 0 para caer en pozos ±v
    Phi = 0.5 * np.sign(rng.normal(size=(L, L))) + 0.15 * rng.normal(size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    hist = []
    for step in range(PASOS):
        tg = step / PASOS
        a = float(np.exp(6 * tg))
        Tnorm = float(np.exp(-6 * tg))
        r = R0 * (Tnorm - TC)

        lap = (
            np.roll(Phi, -1, 1)
            + np.roll(Phi, 1, 1)
            + np.roll(Phi, -1, 0)
            + np.roll(Phi, 1, 0)
            - 4 * Phi
        )
        dV = 2 * r * Phi + 4 * U * Phi**3
        noise = SIGMA0 * np.sqrt(max(Tnorm, 1e-6)) * rng.normal(size=(L, L))
        Phi = Phi + DT_PHI * (-dV + D_PHI * lap) + noise

        left = np.roll(ar, 1, 1)
        up = np.roll(ad, 1, 0)
        cnt = ar.astype(int) + left.astype(int) + ad.astype(int) + up.astype(int)
        s = (
            np.where(ar, np.roll(phi, -1, 1), 0)
            + np.where(left, np.roll(phi, 1, 1), 0)
            + np.where(ad, np.roll(phi, -1, 0), 0)
            + np.where(up, np.roll(phi, 1, 0), 0)
        )
        mean = np.divide(s, cnt, out=np.zeros_like(phi), where=cnt > 0)
        phi_new = phi.copy()
        phi_new[cnt > 0] = phi[cnt > 0] + 0.3 * (mean[cnt > 0] - phi[cnt > 0])
        phi = phi_new

        H_fis = H_TOPO * np.sqrt(Tnorm + 1e-12)
        tot = int(ar.sum() + ad.sum())
        nc = int(round(H_fis * tot))
        nr = 0
        if nc > 0 and tot > 0 and ar.any():
            idx = np.argwhere(ar)
            nr = int(round(nc * float(ar.sum()) / tot))
            if nr > 0 and len(idx):
                sel = rng.choice(len(idx), size=min(nr, len(idx)), replace=False)
                for i in sel:
                    ar[tuple(idx[i])] = False
            rem = nc - nr
            if rem > 0 and ad.any():
                idx = np.argwhere(ad)
                sel = rng.choice(len(idx), size=min(rem, len(idx)), replace=False)
                for i in sel:
                    ad[tuple(idx[i])] = False

        if step % 50 == 0 or step == PASOS - 1:
            cl = clusters_of(phi, ar, ad, Phi)
            Rm, n1, n3, v1, v3 = mass_ratio(cl, Y0, use_phi_field=True)
            Rn, _, _, _, _ = mass_ratio(cl, Y0, use_phi_field=False)
            rec = {
                "step": step,
                "a": a,
                "Tnorm": Tnorm,
                "r": float(r),
                "Phi_abs_mean": float(np.mean(np.abs(Phi))),
                "ratio_REAL": Rm,
                "ratio_NULL": Rn,
                "v_k1": v1,
                "v_k3": v3,
                "k1": n1,
                "k3": n3,
            }
            hist.append(rec)
            print(
                f"step {step:3d} a={a:6.2f} r={r:+.3f} <|Phi|>={rec['Phi_abs_mean']:.4f} "
                f"v1={v1:.4f} v3={v3:.4f} Rm={Rm if Rm is not None else float('nan'):.4f} "
                f"NULL={Rn if Rn is not None else float('nan'):.4f} k1={n1} k3={n3}"
            )

    late = [
        h
        for h in hist
        if h["Tnorm"] < TC
        and h["k1"] >= 5
        and h["k3"] >= 3
        and h["ratio_REAL"] is not None
        and h["ratio_NULL"] is not None
    ]

    Phi_ssb = float(np.mean([h["Phi_abs_mean"] for h in late])) if late else 0.0
    Rm_ssb = float(np.mean([h["ratio_REAL"] for h in late])) if late else None
    Null_ssb = float(np.mean([h["ratio_NULL"] for h in late])) if late else None
    sep = abs(Rm_ssb - Null_ssb) if Rm_ssb is not None and Null_ssb is not None else None

    vev_ok = Phi_ssb > VEV_THR
    null_ok = Null_ssb is not None and abs(Null_ssb - 1.0 / 3.0) < NULL_GEO_TOL
    signal_ok = sep is not None and sep > SEP_THR and vev_ok
    hierarchy = Rm_ssb is not None and Rm_ssb < HIER_THR

    if not late:
        verdict = "TEST_FAIL_no_window"
    elif not vev_ok:
        verdict = "TEST_FAIL_no_VEV"
    elif vev_ok and not signal_ok:
        verdict = "TEST_FAIL_VEV_but_no_mass_signal"
    elif signal_ok and not hierarchy:
        verdict = "TEST_PARTIAL_medium_coupling"
    elif signal_ok and hierarchy:
        verdict = "TEST_PASS_higgs_like"
    else:
        verdict = "TEST_INCONCLUSIVE"

    notes = {
        "TEST_FAIL_no_VEV": (
            "No hay fondo Phi vivo en la fase SSB: NO se testeo el Higgs de verdad. "
            "Fallo de dinamica de VEV (instrumento), no refutacion de la Teoria."
        ),
        "TEST_FAIL_VEV_but_no_mass_signal": (
            "Hay VEV pero Rm no se separa del NULL geometrico: el acople y0*|Phi|*sum_rho "
            "no discrimina dominios (Phi casi uniforme o residual de tamano)."
        ),
        "TEST_PARTIAL_medium_coupling": (
            "VEV + separacion Rm vs NULL: germen de mecanismo tipo medio; jerarquia no fuerte."
        ),
        "TEST_PASS_higgs_like": (
            "VEV + senal vs NULL + Rm estructurado. Mecanismo tipo Higgs operativo en el juguete "
            "(sin exigir 1/1836)."
        ),
        "TEST_FAIL_no_window": "Sin ventana con k1 y k3 suficientes.",
    }

    return {
        "verdict": verdict,
        "note": notes.get(verdict, ""),
        "Phi_abs_mean_SSB": Phi_ssb,
        "Rm_mean_SSB": Rm_ssb,
        "NULL_mean_SSB": Null_ssb,
        "separation_Rm_NULL": sep,
        "flags": {
            "vev_ok": vev_ok,
            "null_ok": bool(null_ok),
            "signal_ok": bool(signal_ok),
            "hierarchy_Rm_lt_0.1": bool(hierarchy),
        },
        "constants": {
            "L": L,
            "PASOS": PASOS,
            "R0": R0,
            "U": U,
            "TC": TC,
            "D_PHI": D_PHI,
            "SIGMA0": SIGMA0,
            "Y0": Y0,
            "SEED": SEED,
            "VEV_THR": VEV_THR,
            "SEP_THR": SEP_THR,
        },
        "history": hist,
    }


def main():
    print("=== Higgs_TEST_REAL ===")
    print("Pregunta: VEV + formula unica => masa != geometria?  (exito != 1/1836)")
    print(
        f"sello R0={R0} U={U} TC={TC} SIGMA0={SIGMA0} D_PHI={D_PHI} Y0={Y0} L={L} seed={SEED}"
    )
    out = run()
    print("\n=== VEREDICTO ===")
    print(out["verdict"])
    print(out["note"])
    print(f"<|Phi|>_SSB={out['Phi_abs_mean_SSB']:.4f}  flags={out['flags']}")
    print(f"Rm_SSB={out['Rm_mean_SSB']}  NULL_SSB={out['NULL_mean_SSB']}  |Rm-NULL|={out['separation_Rm_NULL']}")
    out_path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "fase6_higgs_barrido_final"
        / "Higgs_TEST_REAL_result.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
