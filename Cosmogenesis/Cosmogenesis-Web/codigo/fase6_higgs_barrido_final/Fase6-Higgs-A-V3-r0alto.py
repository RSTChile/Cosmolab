"""
FASE6-HIGGS-A segundo barrido: r0 altos (5-20) para buscar VEV vivo
Misma base L=30 H_topo=0.01; formula unica m=y0*|Phi|_avg*sum_rho
"""
import itertools
from collections import deque

import numpy as np

L = 30
pasos = 350
H_topo = 0.01
rng = np.random.default_rng(2025)

r0_list = [5.0, 8.0, 12.0, 20.0]
u_list = [0.1, 0.3, 0.7]
y0 = 0.3
Tc = 0.5


def run_once(r0, u):
    phi = np.ones((L, L)) + 0.3 * rng.normal(size=(L, L))
    Phi = 0.5 * rng.normal(size=(L, L))
    ar = np.ones((L, L), bool)
    ad = np.ones((L, L), bool)
    nr = 0
    for step in range(pasos):
        tg = step / pasos
        a = np.exp(6 * tg)
        Tnorm = np.exp(-6 * tg)
        r = r0 * (Tc - Tnorm)
        lap = (
            np.roll(Phi, -1, axis=1)
            + np.roll(Phi, 1, axis=1)
            + np.roll(Phi, -1, axis=0)
            + np.roll(Phi, 1, axis=0)
            - 4 * Phi
        )
        dV = 2 * r * Phi + 4 * u * (Phi**3)
        Phi += 0.08 * (-dV + 0.3 * lap) + 0.005 * rng.normal(size=(L, L))
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
        phi_new[cnt > 0] = phi[cnt > 0] + 0.3 * (mean[cnt > 0] - phi[cnt > 0])
        phi = phi_new
        H_fis = H_topo * np.sqrt(Tnorm + 1e-12)
        tot = int(np.sum(ar) + np.sum(ad))
        nc = int(round(H_fis * tot))
        if nc > 0 and tot > 0:
            idx = np.argwhere(ar)
            if len(idx) > 0:
                nr = int(round(nc * np.sum(ar) / tot))
                if nr > 0:
                    sel = rng.choice(len(idx), size=min(nr, len(idx)), replace=False)
                    for i in sel:
                        ar[tuple(idx[i])] = False
            rem = nc - nr
            if rem > 0:
                idx = np.argwhere(ad)
                if len(idx) > 0:
                    sel = rng.choice(len(idx), size=min(rem, len(idx)), replace=False)
                    for i in sel:
                        ad[tuple(idx[i])] = False
    media = phi.mean()
    visto = np.zeros((L, L), bool)
    clusters = []
    for y in range(L):
        for x in range(L):
            if visto[y, x]:
                continue
            q = deque([(y, x)])
            visto[y, x] = True
            nodes = [(y, x)]
            lado = phi[y, x] >= media
            sum_rho = phi[y, x]
            sum_Phi = abs(Phi[y, x])
            perim = 0
            while q:
                cy, cx = q.popleft()
                if not ar[cy, cx] or (phi[cy, (cx + 1) % L] >= media) != lado:
                    perim += 1
                if not ar[cy, (cx - 1) % L] or (phi[cy, (cx - 1) % L] >= media) != lado:
                    perim += 1
                if not ad[cy, cx] or (phi[(cy + 1) % L, cx] >= media) != lado:
                    perim += 1
                if not ad[(cy - 1) % L, cx] or (phi[(cy - 1) % L, cx] >= media) != lado:
                    perim += 1
                for ny, nx, cond in [
                    (cy, (cx + 1) % L, ar[cy, cx]),
                    (cy, (cx - 1) % L, ar[cy, (cx - 1) % L]),
                    ((cy + 1) % L, cx, ad[cy, cx]),
                    ((cy - 1) % L, cx, ad[(cy - 1) % L, cx]),
                ]:
                    if cond and not visto[ny, nx] and (phi[ny, nx] >= media) == lado:
                        visto[ny, nx] = True
                        q.append((ny, nx))
                        nodes.append((ny, nx))
                        sum_rho += phi[ny, nx]
                        sum_Phi += abs(Phi[ny, nx])
            k = len(nodes)
            v_k = sum_Phi / k if k > 0 else 0
            m = y0 * v_k * sum_rho
            clusters.append((k, perim, v_k, m))
    Phi_mean = np.mean(np.abs(Phi))
    k1_m = [c[3] for c in clusters if c[0] == 1]
    k3_m = [c[3] for c in clusters if c[0] == 3 and c[1] == 8]
    ratio = np.mean(k1_m) / (np.mean(k3_m) + 1e-30) if k1_m and k3_m else 0
    vk1 = np.mean([c[2] for c in clusters if c[0] == 1]) if any(c[0] == 1 for c in clusters) else 0
    vk3 = (
        np.mean([c[2] for c in clusters if c[0] == 3 and c[1] == 8])
        if any(c[0] == 3 and c[1] == 8 for c in clusters)
        else 0
    )
    return Phi_mean, ratio, vk1, vk3, len(k1_m), len(k3_m)


print("FASE6-HIGGS-A V3 segundo barrido r0 altos (5-20)")
for r0, u in itertools.product(r0_list, u_list):
    Pm, ratio, vk1, vk3, k1n, k3n = run_once(r0, u)
    flag = "VIVO" if Pm > 0.15 and k1n > 10 and k3n > 5 else "APAGADO"
    print(
        f"r0={r0:.1f} u={u:.1f} <|Phi|>={Pm:.3f} v_k1={vk1:.3f} v_k3={vk3:.3f} "
        f"ratio={ratio:.5f} k1={k1n} k3={k3n} {flag}"
    )
