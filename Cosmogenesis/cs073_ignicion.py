"""
cs073_ignicion.py — IGNICIÓN DE LA PRIMERA ESTRELLA (DISENO_CS073_ignicion_PARA_CC.md).

Orquestador NUEVO -- NO modifica cs073_cierre_holistico.py ni cs073_ley_escala.py (los experimentos ya
reportados siguen corriendo exactamente igual que antes, con gravedad de softening FIJO). Reusa las
piezas ya validadas (Expansion, MateriaOscuraHalo, la malla causal como semilla dinámica, _extraer_
bariones, _fof) e incorpora SÓLO lo que este diseño autoriza: gravedad de resolución ADAPTATIVA
(GravedadGeneral.aceleraciones_adaptativas, p_gravedad_general.py) + paso de tiempo adaptativo
(paso_tiempo_adaptativo) + la MISMA rho_i por partícula compartida entre gravedad y el ρ_local de
Jeans/H2 (EnfriamientoH2.actualizar(rho_externo=...)).

Bucle exterior (cosmológico): idéntico en estructura al del puente -- n_pasos_estructura pasos, cada uno
evalúa el MISMO reloj T(t) del resto del motor y aplica UN estiramiento de expansión. Bucle INTERIOR
(numérico): en vez de n_subpasos fijos, se acumulan sub-pasos de tamaño ADAPTATIVO (paso_tiempo_
adaptativo, función de rho_max) hasta cubrir el mismo presupuesto dt=0.05 por paso cosmológico -- se
preserva el reloj ya validado; sólo la resolución INTERNA se refina donde la densidad lo pide.

Núcleo/Jeans: por cada cluster ligado (FoF, igual criterio que el resto del arco), rho_local = MÁXIMO
rho_i entre sus miembros (el punto más denso resuelto del cluster = su núcleo, no el promedio) -- una
elección de diseño explícita, no está en el texto palabra por palabra, documentada aquí.

Observable (pre-registrado): ¿algún núcleo cruza M_J local (masa_cluster/M_J_nucleo >= 1) por colapso
medido? REAL vs NULL (malla causal con aristas barajadas) -- z-score sobre >=5 semillas x >=8 NULL.
"""
import numpy as np

from cs072_modulos.piezas.p_expansion import Expansion
from cs072_modulos.piezas.p_gravedad_general import GravedadGeneral, paso_tiempo_adaptativo
from cs072_modulos.piezas.p_materia_oscura_halo import MateriaOscuraHalo
from cs072_modulos.piezas.p_enfriamiento_H2 import EnfriamientoH2
from cs072_modulos.piezas.p_semilla_causal import malla_causal_atomos, layout_resortes, barajar_aristas

from cs073_cierre_holistico import _extraer_bariones, _fof, _T_reloj, _z, T0

SOFT_MAX_SUBPASOS = 500   # válvula de seguridad numérica (evita bucle infinito si rho_max diverge), no física


def _dinamica_ignicion(masa_bar, dens_bar, amp_rugosidad,
                        n_pasos_estructura=60, dt=0.05, seed_dens_null=None,
                        cdm_on=True, cooling_on=True, expansion_on=True,
                        D_causal=3, k_causal=4, iters_layout=100, k_adaptativo=6,
                        min_miembros_fof=5):
    """Un bucle temporal (Regla 1 heredada), semilla causal (como el puente), pero gravedad y paso de
    tiempo ADAPTATIVOS (DISENO_CS073_ignicion_PARA_CC.md). seed_dens_null=None -> REAL; entero -> NULL
    (aristas de la malla causal barajadas, densidad SIEMPRE real -- mismo criterio que el puente)."""
    n_bar = len(masa_bar)
    if n_bar < 8:
        return dict(ok=False, nota=f"sólo {n_bar} átomos reales (<8)")

    n_cdm = n_bar if cdm_on else 0
    n_total = n_bar + n_cdm
    lado = float(n_total) ** (1.0 / 3.0)

    adj_bar, _m = malla_causal_atomos(dens_bar, D=D_causal, k=k_causal, seed_ejes=2000)
    if seed_dens_null is not None:
        adj_bar = barajar_aristas(adj_bar, n_bar, seed=seed_dens_null)
    pos_bar = layout_resortes(adj_bar, n_bar, lado=lado, iters=iters_layout, seed=12345)

    cdm = MateriaOscuraHalo(n_cdm, amp_rugosidad, lado_escenario=lado, activa=cdm_on,
                             seed_pos=54321, seed_dens=7000)
    if n_cdm:
        adj_cdm, _m2 = malla_causal_atomos(cdm.densidad, D=D_causal, k=k_causal, seed_ejes=6000)
        if seed_dens_null is not None:
            adj_cdm = barajar_aristas(adj_cdm, n_cdm, seed=seed_dens_null + 1)
        cdm.pos = layout_resortes(adj_cdm, n_cdm, lado=lado, iters=iters_layout, seed=54321)

    grav = GravedadGeneral(activa=True)   # G_ADIM=1, ver p_gravedad_general.py -- sin softening fijo
    expansion = Expansion(T0=T0, activa=expansion_on)
    h2 = EnfriamientoH2(n_bar, T_inicial=T0, activa_cooling=cooling_on, seed=9000)

    pos = np.vstack([pos_bar, cdm.pos]) if n_cdm else pos_bar.copy()
    masa_eff = np.concatenate([masa_bar * dens_bar, cdm.masa * cdm.densidad]) if n_cdm else masa_bar * dens_bar
    vel = np.zeros_like(pos)

    ignicion_en_paso = None
    max_subpasos_usados = 0

    for step in range(n_pasos_estructura):
        T_actual = _T_reloj(step)
        t_acum = 0.0
        n_sub = 0
        rho_i_ultimo = None
        while t_acum < dt and n_sub < SOFT_MAX_SUBPASOS:
            acc, eps_i, rho_i, rho_max = grav.aceleraciones_adaptativas(pos, k=k_adaptativo, masa=masa_eff)
            rho_i_ultimo = rho_i
            dt_i = paso_tiempo_adaptativo(rho_max)
            dt_i = min(dt_i, dt - t_acum)
            vel = vel + acc * dt_i
            if n_bar:
                vel[:n_bar] = vel[:n_bar] + h2.kick_termico(escala=0.02) * np.sqrt(max(dt_i, 1e-12))
            pos = pos + vel * dt_i
            if not np.all(np.isfinite(pos)):
                return dict(ok=False, nota=f"NaN/inf en las posiciones al paso {step} (subpaso {n_sub})")
            t_acum += dt_i
            n_sub += 1
        max_subpasos_usados = max(max_subpasos_usados, n_sub)
        if n_bar and rho_i_ultimo is not None:
            h2.actualizar(pos[:n_bar], rho_externo=rho_i_ultimo[:n_bar])

        factor = expansion.paso_de_estiramiento(T_actual)
        if factor != 1.0:
            pos = pos * factor

        # chequeo de ignición este paso (con la rho_i YA calculada, sin costo extra)
        if n_bar and rho_i_ultimo is not None and ignicion_en_paso is None:
            _, _, rho_i_bar_final, _ = grav.aceleraciones_adaptativas(pos, k=k_adaptativo, masa=masa_eff)
            rho_i_bar = rho_i_bar_final[:n_bar]
            lado_actual = lado * (expansion._a_prev if expansion_on else 1.0)
            linking_length = 0.2 * lado_actual / (float(n_total) ** (1 / 3))
            clusters = _fof(pos[:n_bar], linking_length, min_miembros=min_miembros_fof)
            for miembros in clusters:
                masa_cluster = float(masa_eff[:n_bar][miembros].sum())
                rho_nucleo = float(rho_i_bar[miembros].max())
                T_nucleo = float(h2.T[miembros].mean())
                M_J = T_nucleo ** 1.5 / np.sqrt(max(rho_nucleo, 1e-12))
                if masa_cluster >= M_J:
                    ignicion_en_paso = step
                    break

    # estado final: mejor razón masa/M_J alcanzada (con núcleo = rho_i MÁXIMA del cluster)
    acc, eps_i, rho_i_final, rho_max_final = grav.aceleraciones_adaptativas(pos, k=k_adaptativo, masa=masa_eff)
    rho_i_bar = rho_i_final[:n_bar] if n_bar else np.zeros(0)
    lado_actual = lado * (expansion._a_prev if expansion_on else 1.0)
    linking_length = 0.2 * lado_actual / (float(n_total) ** (1 / 3))
    clusters = _fof(pos[:n_bar], linking_length, min_miembros=min_miembros_fof) if n_bar else []
    detalle = []
    for miembros in clusters:
        masa_cluster = float(masa_eff[:n_bar][miembros].sum())
        rho_nucleo = float(rho_i_bar[miembros].max())
        T_nucleo = float(h2.T[miembros].mean())
        M_J = T_nucleo ** 1.5 / np.sqrt(max(rho_nucleo, 1e-12))
        detalle.append(dict(n_miembros=len(miembros), masa=masa_cluster, T_nucleo=T_nucleo,
                             rho_nucleo=rho_nucleo, M_J=M_J, razon=masa_cluster / M_J if M_J > 0 else None,
                             supera_jeans=bool(masa_cluster >= M_J)))

    return dict(ok=True, n_bariones=n_bar, n_cdm=n_cdm,
                ignicion=ignicion_en_paso is not None, ignicion_en_paso=ignicion_en_paso,
                n_clusters_ligados=len(clusters),
                n_nucleos_ignicion=sum(1 for d in detalle if d["supera_jeans"]),
                razon_max=max((d["razon"] for d in detalle if d["razon"] is not None), default=0.0),
                max_subpasos_por_paso_cosmologico=max_subpasos_usados,
                detalle=detalle)


def correr_ignicion_real_vs_null(nq, naq, ne, npos, pasos_basal=150, amp_rugosidad=1.5,
                                  n_semillas_real=5, n_null=8, seed_real_base=1000, seed_null_base=5000,
                                  **kw):
    """>=5 semillas REAL x >=8 NULL cada una (DISENO_CS073_ignicion_PARA_CC.md). 'semilla REAL' aquí =
    distinta semilla de layout_resortes/malla causal para la parte REAL (no re-extrae el motor basal,
    que es determinista) -- reutiliza el mismo pool de átomos, sólo cambia la realización del despliegue."""
    masa_bar, dens_bar, obs_basal = _extraer_bariones(nq, naq, ne, npos, pasos_basal, amp_rugosidad)

    reales = []
    for s in range(n_semillas_real):
        r = _dinamica_ignicion(masa_bar, dens_bar, amp_rugosidad, seed_dens_null=None, **kw)
        reales.append(r)
        print(f"  REAL semilla {s}: ignicion={r.get('ignicion')} razon_max={r.get('razon_max')}", flush=True)

    nulls = []
    for i in range(n_null):
        r = _dinamica_ignicion(masa_bar, dens_bar, amp_rugosidad, seed_dens_null=seed_null_base + i * 2, **kw)
        nulls.append(r)
        print(f"  NULL {i}: ignicion={r.get('ignicion')} razon_max={r.get('razon_max')}", flush=True)

    reales_ok = [r for r in reales if r.get("ok")]
    nulls_ok = [r for r in nulls if r.get("ok")]

    ignicion_real = [1 if r["ignicion"] else 0 for r in reales_ok]
    ignicion_null = [1 if r["ignicion"] else 0 for r in nulls_ok]
    razon_real = [r["razon_max"] for r in reales_ok]
    razon_null = [r["razon_max"] for r in nulls_ok]

    z_ignicion = _z(float(np.mean(ignicion_real)), ignicion_null) if ignicion_null else None
    z_razon = _z(float(np.mean(razon_real)), razon_null) if razon_null else None

    return dict(ok=True,
                n_semillas_real_ok=len(reales_ok), n_null_ok=len(nulls_ok),
                tasa_ignicion_real=float(np.mean(ignicion_real)) if ignicion_real else None,
                tasa_ignicion_null=float(np.mean(ignicion_null)) if ignicion_null else None,
                razon_max_media_real=float(np.mean(razon_real)) if razon_real else None,
                razon_max_media_null=float(np.mean(razon_null)) if razon_null else None,
                razon_max_std_null=float(np.std(razon_null, ddof=1)) if len(razon_null) > 1 else 0.0,
                z_ignicion=z_ignicion, z_razon_max=z_razon,
                obs_basal=dict(hidrogeno=obs_basal.get("hidrogeno"), helio=obs_basal.get("helio")),
                detalle_reales=reales_ok, detalle_nulls=nulls_ok)
