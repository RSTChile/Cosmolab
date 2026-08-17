"""
MF-1 — Masa inercial térmica sobre F0 cerrado
Meta=Mac F0: PASS; mk1/mk3=1/3 geométrico (sum_rho ~ k * 1).

Contrato MF-1 (preregistro + Alexis):
  m_fis = sum_i ρ_ℰ,i * exp(-var_in,i / T_local,i) / (P_norm * a^β)
  k3 coherente: var_in ~ 0 → peso ~ 1 retiene ℰ
  k1 fluctuante: var_in alta → disipa

Fix vs F0: var de cluster k=1 es trivialmente 0; se usa rugosidad LOCAL
  (varianza de phi en vecindario + dispersión de theta), no var interna del
  conjunto de 1 celda.

Constantes FIJAS (no retocar para 1/1836).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

# --- sello (misma familia F0 final unico + extras MF-1 fijos) ---
T0 = 1e15
rho0 = 1e30
H_topo = 0.01
L = 30
pasos = 500
EPS = 1e-5  # S>0 asimetría ∇T basal (fija)
BETA = 0.0  # a^β; 0 = no potencia extra (fija)
W = 1.0 / 3.0
C2 = (3e8) ** 2
SEED = 2025


def local_var_phi(phi: np.ndarray) -> np.ndarray:
    """Rugosidad de phi en malla completa (4 vecinos), independiente del grafo cortado.

    Si se usara solo ar/ad, los k1 aislados tendrían var=0 trivial y no podrían
    'disipar' — eso reintroduce el ratio geométrico 1/3.
    """
    acc = phi.astype(float).copy()
    acc2 = phi.astype(float) ** 2
    for shift, ax in ((-1, 1), (1, 1), (-1, 0), (1, 0)):
        nb = np.roll(phi, shift, axis=ax)
        acc += nb
        acc2 += nb * nb
    cnt = 5.0
    mean = acc / cnt
    return np.maximum(acc2 / cnt - mean * mean, 0.0)


def local_var_theta(theta: np.ndarray) -> np.ndarray:
    """Dispersión angular local en malla completa (sin^2)."""
    s = np.zeros_like(theta, dtype=float)
    for shift, ax in ((-1, 1), (1, 1), (-1, 0), (1, 0)):
        d = np.sin(np.roll(theta, shift, axis=ax) - theta)
        s += d * d
    return s / 4.0


def run() -> dict:
    rng = np.random.default_rng(SEED)
    xs = np.linspace(0, 1, L)
    xx, yy = np.meshgrid(xs, xs)
    pert = np.zeros((L, L))
    for mx in range(1, 4):
        for my in range(1, 4):
            pert += np.sin(2 * np.pi * (mx * xx + my * yy) + rng.uniform(0, 2 * np.pi)) / (
                mx + my
            )
    pert -= pert.mean()
    pert /= pert.std() if pert.std() > 0 else 1.0

    phi = np.ones((L, L)) + 1e-9 * pert
    theta = rng.uniform(0, 2 * np.pi, size=(L, L))
    ar = np.ones((L, L), dtype=bool)
    ad = np.ones((L, L), dtype=bool)

    max_err = 0.0
    samples: list[dict] = []

    for step in range(pasos):
        tg = step / pasos
        a = float(np.exp(6 * tg))
        T_field = T0 / a
        rho_field = rho0 / (a**3)
        err = abs(T_field * a / T0 - 1)
        max_err = max(max_err, err)

        # T_local = T0/a * (1 + ε * pert) — asimetría basal S>0
        T_local = T_field * (1.0 + EPS * pert)
        T_local = np.maximum(T_local, 1e-30)

        # difusión phi
        left = np.roll(ar, 1, axis=1)
        up = np.roll(ad, 1, axis=0)
        cnt = ar.astype(int) + left.astype(int) + ad.astype(int) + up.astype(int)
        s = (
            np.where(ar, np.roll(phi, -1, axis=1), 0)
            + np.where(left, np.roll(phi, 1, axis=1), 0)
            + np.where(ad, np.roll(phi, -1, axis=0), 0)
            + np.where(up, np.roll(phi, 1, axis=0), 0)
        )
        mean = np.divide(s, cnt, out=np.zeros_like(phi), where=cnt > 0)
        phi_new = phi.copy()
        phi_new[cnt > 0] = phi[cnt > 0] + 0.5 * (mean[cnt > 0] - phi[cnt > 0])
        phi = phi_new

        # fase
        K = np.exp(-1.0 / (a + 1e-10))
        dth = (
            np.where(ar, np.sin(np.roll(theta, -1, axis=1) - theta), 0)
            + np.where(left, np.sin(np.roll(theta, 1, axis=1) - theta), 0)
            + np.where(ad, np.sin(np.roll(theta, -1, axis=0) - theta), 0)
            + np.where(up, np.sin(np.roll(theta, 1, axis=0) - theta), 0)
        )
        theta = np.mod(theta + 0.1 * K * dth, 2 * np.pi)

        # corte
        H_fis = H_topo * np.sqrt(T_field / T0)
        tot = int(np.sum(ar) + np.sum(ad))
        nc = int(round(H_fis * tot))
        if nc > 0 and tot > 0 and np.sum(ar) > 0:
            nr = int(round(nc * np.sum(ar) / tot))
            idx = np.argwhere(ar)
            if len(idx) > 0 and nr > 0:
                sel = rng.choice(len(idx), size=min(nr, len(idx)), replace=False)
                for i in sel:
                    ar[tuple(idx[i])] = False
            rem = nc - nr
            if rem > 0 and np.sum(ad) > 0:
                idx = np.argwhere(ad)
                sel = rng.choice(len(idx), size=min(rem, len(idx)), replace=False)
                for i in sel:
                    ad[tuple(idx[i])] = False

        if step % 50 == 0:
            vphi = local_var_phi(phi)
            vth = local_var_theta(theta)
            # contraste local de T (asimetría S>0): |T_local/T_field - 1|
            t_fluct = np.abs(T_local / T_field - 1.0)
            # var_in: fase + campo + fluctuación térmica local
            var_in = vphi + vth + t_fluct

            # ρ_ℰ: energía relativa por celda
            rho_E = rho_field * (T_local / T0) * phi
            P_norm = W * rho_field * C2
            if P_norm <= 0:
                P_norm = 1e-30
            a_factor = a**BETA

            media = phi.mean()
            visto = np.zeros((L, L), dtype=bool)
            k_cnt = {1: 0, 2: 0, 3: 0, 6: 0}
            perim8 = 0
            m1_list = []
            m3_list = []
            var1 = []
            var3 = []

            for y in range(L):
                for x in range(L):
                    if visto[y, x]:
                        continue
                    q = deque([(y, x)])
                    visto[y, x] = True
                    nodes = [(y, x)]
                    lado = phi[y, x] >= media
                    perim = 0
                    while q:
                        cy, cx = q.popleft()
                        if (not ar[cy, cx]) or (phi[cy, (cx + 1) % L] >= media) != lado:
                            perim += 1
                        if (not ar[cy, (cx - 1) % L]) or (phi[cy, (cx - 1) % L] >= media) != lado:
                            perim += 1
                        if (not ad[cy, cx]) or (phi[(cy + 1) % L, cx] >= media) != lado:
                            perim += 1
                        if (not ad[(cy - 1) % L, cx]) or (phi[(cy - 1) % L, cx] >= media) != lado:
                            perim += 1
                        if (
                            ar[cy, cx]
                            and not visto[cy, (cx + 1) % L]
                            and (phi[cy, (cx + 1) % L] >= media) == lado
                        ):
                            visto[cy, (cx + 1) % L] = True
                            q.append((cy, (cx + 1) % L))
                            nodes.append((cy, (cx + 1) % L))
                        if (
                            ar[cy, (cx - 1) % L]
                            and not visto[cy, (cx - 1) % L]
                            and (phi[cy, (cx - 1) % L] >= media) == lado
                        ):
                            visto[cy, (cx - 1) % L] = True
                            q.append((cy, (cx - 1) % L))
                            nodes.append((cy, (cx - 1) % L))
                        if (
                            ad[cy, cx]
                            and not visto[(cy + 1) % L, cx]
                            and (phi[(cy + 1) % L, cx] >= media) == lado
                        ):
                            visto[(cy + 1) % L, cx] = True
                            q.append(((cy + 1) % L, cx))
                            nodes.append(((cy + 1) % L, cx))
                        if (
                            ad[(cy - 1) % L, cx]
                            and not visto[(cy - 1) % L, cx]
                            and (phi[(cy - 1) % L, cx] >= media) == lado
                        ):
                            visto[(cy - 1) % L, cx] = True
                            q.append(((cy - 1) % L, cx))
                            nodes.append(((cy - 1) % L, cx))

                    k = len(nodes)
                    if k in k_cnt:
                        k_cnt[k] += 1
                    if k == 3 and perim == 8:
                        perim8 += 1

                    # m_fis del dominio
                    m = 0.0
                    v_acc = 0.0
                    for cy, cx in nodes:
                        vi = var_in[cy, cx]
                        # T adimensional O(1): T_local/T_field ≈ 1+ε·pert
                        T_dimless = max(float(T_local[cy, cx] / T_field), 1e-30)
                        weight = np.exp(-vi / T_dimless)
                        m += float(rho_E[cy, cx]) * weight
                        v_acc += vi
                    m /= P_norm * a_factor
                    v_mean = v_acc / k

                    if k == 1:
                        m1_list.append(m)
                        var1.append(v_mean)
                    if k == 3 and perim == 8:
                        m3_list.append(m)
                        var3.append(v_mean)

            mk1 = float(np.mean(m1_list)) if m1_list else 0.0
            mk3 = float(np.mean(m3_list)) if m3_list else 0.0
            # también suma total (no solo media por cluster) — jerarquía de carga de masa
            sum1 = float(np.sum(m1_list)) if m1_list else 0.0
            sum3 = float(np.sum(m3_list)) if m3_list else 0.0
            ratio_mean = mk1 / (mk3 + 1e-30) if mk3 > 0 else 0.0
            ratio_sum = sum1 / (sum3 + 1e-30) if sum3 > 0 else 0.0
            # ratio geométrico de referencia (conteo)
            ratio_geo = (1.0) / (3.0) if mk3 > 0 else 0.0

            rec = {
                "step": step,
                "a": a,
                "T": float(T_field),
                "k1": k_cnt[1],
                "k3": k_cnt[3],
                "perim8": perim8,
                "mk1_mean": mk1,
                "mk3_mean": mk3,
                "ratio_mean": ratio_mean,
                "ratio_sum": ratio_sum,
                "ratio_geo_ref": ratio_geo,
                "var1_mean": float(np.mean(var1)) if var1 else 0.0,
                "var3_mean": float(np.mean(var3)) if var3 else 0.0,
                "err_Ta": float(err),
            }
            samples.append(rec)
            print(
                f"step {step:3d} a={a:7.2f} T={T_field:.2e} | "
                f"k1={k_cnt[1]:3d} k3={k_cnt[3]:3d} p8={perim8:2d} | "
                f"var1={rec['var1_mean']:.3e} var3={rec['var3_mean']:.3e} | "
                f"Rm_mean={ratio_mean:.4f} Rm_sum={ratio_sum:.4f} geo=0.3333 | err={err:.1e}"
            )

    # veredicto MF-1 (preregistro: R_m deja O(1) hacia jerarquía estructurada)
    # usar último sample con k3>0
    last = None
    for s in reversed(samples):
        if s["k3"] > 0 and s["k1"] > 0:
            last = s
            break
    if last is None:
        mf1 = "MF1-FAIL-no-clusters"
        Rm = None
    else:
        Rm = last["ratio_mean"]
        # geo ~ 0.33; éxito si se separa claramente de 1/3 hacia abajo (≪ O(1))
        if Rm < 0.05:
            mf1 = "MF1-STRONG"  # jerarquía estructurada (meta ~0.01)
        elif Rm < 0.20:
            mf1 = "MF1-PARTIAL"  # mejor que T/geo, no aún 0.01
        elif abs(Rm - 1.0 / 3.0) < 0.05:
            mf1 = "MF1-FAIL-geometric"  # sigue ~1/3
        else:
            mf1 = "MF1-FAIL-O1"

    smoke = "PASS" if max_err < 0.05 else "FAIL"
    out = {
        "smoke": smoke,
        "max_err_Ta": max_err,
        "mf1_verdict": mf1,
        "Rm_last": Rm,
        "formula": "m=sum(rho_E*exp(-var_local/T_dimless))/(P_norm*a^beta)",
        "var_definition": "local var_phi + local var_theta (not trivial k=1 cluster var)",
        "constants": {
            "T0": T0,
            "rho0": rho0,
            "H_topo": H_topo,
            "L": L,
            "pasos": pasos,
            "EPS": EPS,
            "BETA": BETA,
            "W": W,
            "SEED": SEED,
        },
        "samples": samples,
    }
    return out


def main() -> None:
    print("MF-1 masa termica sobre F0 cerrado")
    print("m ~ sum rho_E exp(-var_in/T_dimless) / (P a^beta) | var=local(phi,theta)")
    out = run()
    print(f"\nSMOKE: {out['smoke']} max|Ta/T0-1|={out['max_err_Ta']:.2e}")
    print(f"MF-1: {out['mf1_verdict']} Rm_last={out['Rm_last']}")
    root = Path(__file__).resolve().parents[1]  # Cosmogenesis-Web
    path = root / "results" / "mf1_masa_termica_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
