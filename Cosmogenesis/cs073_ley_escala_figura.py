# Genera la figura log-log pedida: z(N) y (masa/M_J)(N), con la ley ajustada y la extrapolación anotada.
# No se grafica N_fisico en el mismo eje que los datos (1e62 vs 500 -- 60 ordenes de magnitud harian
# ilegible el rango real medido); se anota como texto con su banda de incertidumbre, honesto sobre la
# distancia real entre lo medido y lo extrapolado.
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cs073_ley_escala import ajustar_ley_potencia, extrapolar, N_FISICO_POP_III

with open("cs073_ley_escala_resultados.json") as f:
    filas = json.load(f)

ok = [f for f in filas if f.get("ok")]
Ns = np.array([f["N"] for f in ok], float)
razones = np.array([f["razon_jeans_real"] for f in ok], float)
zs_all = [f["z_ligados"] for f in ok]
Ns_z = np.array([f["N"] for f, z in zip(ok, zs_all) if z is not None], float)
zs = np.array([z for z in zs_all if z is not None], float)

ajuste_r = ajustar_ley_potencia(Ns, razones)
ajuste_z = ajustar_ley_potencia(Ns_z, zs)
extra_r = extrapolar(ajuste_r, N_FISICO_POP_III)
extra_z = extrapolar(ajuste_z, N_FISICO_POP_III) if ajuste_z.get("ok") else None

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- panel 1: razon masa/M_J vs N ---
ax = axes[0]
mask = razones > 0
ax.loglog(Ns[mask], razones[mask], "o", color="#2563eb", label="datos (REAL, mejor cluster)", zorder=3)
if ajuste_r.get("ok"):
    Nfit = np.linspace(Ns[mask].min(), Ns[mask].max(), 50)
    yfit = 10 ** (ajuste_r["alpha"] * np.log10(Nfit) + ajuste_r["log10_A"])
    ax.loglog(Nfit, yfit, "-", color="#2563eb", alpha=0.6,
               label=f"ajuste: N^{ajuste_r['alpha']:.2f}±{ajuste_r['error_alpha']:.2f} (R²={ajuste_r['R2']:.3f})")
ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1, label="umbral de Jeans (razón=1)")
ax.set_xlabel("N (átomos reales por corrida)")
ax.set_ylabel("masa_cluster / M_J_local (máx., REAL)")
ax.set_title("Razón masa/Jeans vs escala")
ax.legend(fontsize=8, loc="upper left")
txt = (f"extrapolado a N={N_FISICO_POP_III:.0e}:\nlog10(razón) = {extra_r['log10_y_central']:.1f} "
       f"[{extra_r['log10_y_lo']:.1f}, {extra_r['log10_y_hi']:.1f}]\n"
       f"(banda ±{extra_r['delta_log10']:.1f} dec.; cruza_umbral_1={extra_r['cruza_umbral_1']})")
ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
        bbox=dict(boxstyle="round", fc="#fef3c7", ec="#d97706", alpha=0.9))

# --- panel 2: z-score (clusters ligados REAL vs NULL) vs N ---
ax = axes[1]
ax.set_xscale("log")
ax.plot(Ns_z, zs, "o", color="#16a34a", label="datos (z de n_clusters_ligados)", zorder=3)
if ajuste_z.get("ok"):
    Nfit = np.linspace(Ns_z.min(), Ns_z.max(), 50)
    yfit = 10 ** (ajuste_z["alpha"] * np.log10(Nfit) + ajuste_z["log10_A"])
    ax.plot(Nfit, yfit, "-", color="#16a34a", alpha=0.6,
             label=f"ajuste: N^{ajuste_z['alpha']:.2f}±{ajuste_z['error_alpha']:.2f} (R²={ajuste_z['R2']:.3f})")
ax.axhline(2.0, color="#dc2626", linestyle="--", linewidth=1, label="z=2 (convención 2σ)")
ax.set_xlabel("N (átomos reales por corrida)")
ax.set_ylabel("z (clusters ligados REAL vs NULL)")
ax.set_title("Discriminante REAL-NULL vs escala\n(AJUSTE POBRE -- R² bajo, no monótono, ver nota)")
ax.legend(fontsize=8, loc="upper right")

fig.suptitle("CS073 — puente causal: ley de escala del discriminante (19-jul-2026)", fontsize=10)
fig.tight_layout()
fig.savefig("cs073_ley_escala_figura.png", dpi=150)
print("guardado: cs073_ley_escala_figura.png")
print(json.dumps(dict(ajuste_razon=ajuste_r, ajuste_z=ajuste_z, extrapolacion_razon=extra_r, extrapolacion_z=extra_z),
                  indent=2, default=str))
