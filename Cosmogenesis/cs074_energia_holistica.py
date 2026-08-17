#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs074_energia_holistica.py — Experimento holístico: la energía como capa transversal
=========================================================================================

Quién soy / qué hago (código autodescriptivo):
  Orquestador NUEVO (no edita cs072_modulos/ ni cs073_cierre_holistico.py -- ambos se
  importan/leen, no se tocan). Implementa PROTOCOLO_cs074_energia_holistica_PREREGISTRO.md
  (leer ese documento primero; congelado ANTES de este motor).

  Corre la dinámica de formación de estructura de CS073 (gravedad general + expansión +
  enfriamiento H2 + materia oscura -- las mismas piezas, mismas convenciones G_ADIM=1,
  SOFTENING=0.3) con una capa de energía añadida: un presupuesto E_total cerrado, exergía
  medida por diferencias, y un costo de ligadura que se cobra cuando la gravedad forma
  estructura nueva -- el mecanismo por el que el presupuesto tiene un efecto CAUSAL real
  sobre qué cuenta como "materia" al final (protocolo §3 Regla 4, §6 admisibilidad).

  Las 23 piezas del Modelo Estándar (quark->átomo, cs072_modulos/nucleo.py) NO se tocan:
  se llama corre() UNA vez para obtener bariones/masa/densidad, igual que hace CS073. Todo
  lo demás (bucle de dinámica, ledger de energía, criterio de ligadura) es propio de este
  archivo -- self-contained, no importa la lógica privada de cs073_cierre_holistico.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cs072_modulos.nucleo import corre  # noqa: E402
from cs072_modulos.piezas.p_expansion import Expansion  # noqa: E402
from cs072_modulos.piezas.p_gravedad_general import (  # noqa: E402
    GravedadGeneral, posiciones_escenario, energia_total, G_ADIM,
)
from cs072_modulos.piezas.p_materia_oscura_halo import MateriaOscuraHalo  # noqa: E402
from cs072_modulos.piezas.p_enfriamiento_H2 import EnfriamientoH2  # noqa: E402

OUT = HERE / "resultados_cs074_energia_holistica"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes declaradas en el protocolo (§2, §3) -- no se tocan tras ver resultados (T1/T3)
# ---------------------------------------------------------------------------
T0 = 3.0
TASA_EXPANSION_DEFAULT = 0.02
SOFTENING = 0.3  # misma convención compartida gravedad/H2 que usa CS073
TOL_DERIVA_CONTROL = 0.05  # 5%, protocolo §3 Regla 1


def _T_reloj(step, tasa_expansion):
    """MISMO reloj de enfriamiento que nucleo.Estado.enfria / CS073 -- ninguna ley nueva."""
    return T0 / np.sqrt(1.0 + (tasa_expansion * 50.0) * step)


def _extraer_bariones(nq, naq, ne, npos, pasos_basal, amp_rugosidad, tasa_expansion):
    """Corre el motor basal YA VALIDADO una vez; devuelve masa y densidad #23 reales de los
    átomos (H). No se re-deriva nada del Modelo Estándar -- se reutiliza el resultado
    (mismo patrón que cs073_cierre_holistico._extraer_bariones)."""
    obs, e = corre(nq, naq, ne, npos, tasa_expansion=tasa_expansion, pasos=pasos_basal,
                    amp_rugosidad=amp_rugosidad, devolver_estado=True)
    atomos_H = [n for (n, _) in e.Bem]
    masa = np.array([e.masa_trio.get(a, 1.0) for a in atomos_H], float)
    densidad = np.array([e.densidad[a] for a in atomos_H], float)
    return masa, densidad, obs


def _fof(pos, linking_length, min_miembros=2):
    """Friends-of-friends: cKDTree + componentes conexas -- el mismo algoritmo estándar
    que usa cs073_cierre_holistico.py, copiado aquí (no importado) para que este archivo
    sea autocontenido, mismo criterio que cada motor de ENFOQUE 5."""
    n = len(pos)
    if n < min_miembros:
        return []
    tree = cKDTree(pos)
    pares = list(tree.query_pairs(r=linking_length))
    if not pares:
        return []
    filas = [p[0] for p in pares] + [p[1] for p in pares]
    cols = [p[1] for p in pares] + [p[0] for p in pares]
    adj = csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    clusters = []
    for c in range(n_comp):
        miembros = np.where(labels == c)[0]
        if len(miembros) >= min_miembros:
            clusters.append(miembros)
    return clusters


def _pe_interno(pos, vel, masa, idx, G=G_ADIM, softening=SOFTENING):
    """PE interno de un subconjunto idx, aislado de KE por álgebra: energia_total()=KE+PE
    (reusada tal cual, protocolo §3), así que PE = energia_total - KE_total_no_relativa."""
    p, v, m = pos[idx], vel[idx], masa[idx]
    eps_i = np.full(len(idx), softening)
    et = energia_total(p, v, m, eps_i, G)
    ke_total = 0.5 * float(np.sum(m * np.sum(v ** 2, axis=1)))
    return et - ke_total


def _ke_interno_relativo(vel, masa, idx):
    """KE interno relativo al centro de masa del cluster (excluye movimiento de conjunto
    -- criterio virial correcto de ligadura, protocolo §3 Regla 4)."""
    v, m = vel[idx], masa[idx]
    v_cm = np.average(v, axis=0, weights=m)
    return 0.5 * float(np.sum(m * np.sum((v - v_cm) ** 2, axis=1)))


def correr_holistico_energia(
    nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5,
    E_reserva=1.0, reserva_como_multiplo_mecanica=True,
    n_pasos_estructura=60, dt=0.05, n_subpasos=10,
    cdm_on=True, cooling_on=True, expansion_on=True, gravedad_on=True, energia_on=True,
    tasa_expansion=TASA_EXPANSION_DEFAULT, seed_layout=12345, seed_dens_cdm=7000,
    min_miembros_ligadura=2, tol_deriva_control=TOL_DERIVA_CONTROL, guardar_curva=True,
    incluir_presion_termica=True, tasa_enfriamiento=0.3, seed_dens_null=None,
):
    """UN bucle temporal, TODOS los módulos juntos (protocolo §2/§5 -- Regla holística: nunca
    piezas por separado). Devuelve el balance completo, crudo, sin adjudicar (protocolo §8).

    energia_on=False -> E_reserva efectiva = infinito (el gate de Regla 4 nunca bloquea);
    es el brazo "SIN energía" de la prueba de admisibilidad (protocolo §6).

    tasa_enfriamiento=0.3 -- dial CONTINUO de intensidad del canal H2 (Experimento B,
    DISENO_tres_experimentos_holistico_PARA_CC.md), pasado tal cual a EnfriamientoH2 (su
    propio default es 0.3 -- este parámetro es aditivo, no cambia ningún resultado previo
    de cs074 que no lo especifique). 0.0 = sin relajación hacia el piso (equivalente en
    efecto a cooling_on=False aunque cooling_on siga controlando el interruptor discreto).

    incluir_presion_termica=False -- SOLO para el smoke test de conservación (protocolo §3):
    EnfriamientoH2.kick_termico() es presión térmica/EM que, por diseño de esa pieza,
    "SIEMPRE actúa" (su propio docstring), independiente de cooling_on -- así que
    cooling_on=False NO alcanza para aislar "gravedad pura". energia_total() (p_gravedad_
    general.py) ya documenta que su chequeo de conservación "sólo tiene sentido... sin
    kicks térmicos" -- este flag apaga ESE canal únicamente para el control, nunca para
    una corrida física real (ahí incluir_presion_termica=True, default)."""
    t0_wall = time.time()
    masa_bar, dens_bar, obs_basal = _extraer_bariones(nq, naq, ne, npos, pasos_basal,
                                                        amp_rugosidad, tasa_expansion)
    if seed_dens_null is not None:
        # NULL (Experimento B, protocolo §control): misma receta que cs073_cierre_holistico
        # (barajar la densidad #23 entre bariones, preservando masa/posición/física intactas
        # -- destruye SOLO la coherencia densidad<->identidad, no la cantidad).
        dens_bar = dens_bar[np.random.default_rng(seed_dens_null).permutation(len(dens_bar))]
    n_bar = len(masa_bar)
    if n_bar < 8:
        return dict(ok=False, nota=f"sólo {n_bar} átomos reales (<8): sin masa suficiente")

    n_cdm = n_bar if cdm_on else 0
    lado = float(n_bar + n_cdm) ** (1.0 / 3.0)
    pos_bar, _ = posiciones_escenario(n_bar, lado=lado, seed=seed_layout)
    cdm = MateriaOscuraHalo(n_cdm, amp_rugosidad, lado_escenario=lado, activa=cdm_on,
                             seed_pos=54321, seed_dens=seed_dens_cdm)

    pos = np.vstack([pos_bar, cdm.pos]) if n_cdm else pos_bar.copy()
    masa = np.concatenate([masa_bar * dens_bar, cdm.masa * cdm.densidad]) if n_cdm else masa_bar * dens_bar
    vel = np.zeros_like(pos)
    n_tot = len(pos)

    grav = GravedadGeneral(activa=gravedad_on, softening=SOFTENING)
    expansion = Expansion(T0=T0, activa=expansion_on)
    h2 = EnfriamientoH2(n_bar, T_inicial=T0, activa_cooling=cooling_on, seed=9000,
                         softening=SOFTENING, tasa_enfriamiento=tasa_enfriamiento)

    # --- ledger de energía (protocolo §3) ---
    et0 = energia_total(pos, vel, masa, np.full(n_tot, SOFTENING), G_ADIM)  # vel=0 -> et0 = PE(0)
    mecanica_ref = abs(et0) if et0 != 0 else float(np.sum(masa))
    if not energia_on:
        E_reserva_abs = float("inf")
    elif reserva_como_multiplo_mecanica:
        E_reserva_abs = float(E_reserva) * mecanica_ref
    else:
        E_reserva_abs = float(E_reserva)
    E_total0 = et0 + E_reserva_abs  # cantidad conservada REAL (con signo -- PE(0) es <=0)
    # Denominador de reporte: SUMA de magnitudes (nunca abs(et0+E_reserva_abs), que puede
    # cancelarse a ~0 si E_reserva_abs se elige justo para compensar et0<0 -- degenerado
    # para fracciones/tolerancias, no para la ecuación de cierre en sí, que sí usa E_total0
    # con signo). Siempre positivo y lejos de cero salvo el caso trivial masa=0.
    denom_frac = mecanica_ref + E_reserva_abs

    reserva_restante = E_reserva_abs
    ligada_acum = 0.0
    acreditados = set()
    masa_bariones_total = float(np.sum(masa[:n_bar])) if n_bar else 0.0
    masa_acreditada = 0.0  # observable PRIMARIO (protocolo, refinado en verificación):
    # fracción de MASA bariónica acreditada, denominador FIJO (masa_bariones_total no
    # depende de E_reserva) -- a diferencia de frac_ligada_estructura (energía/denom_frac),
    # que NO sirve para comparar entre reservas distintas porque denom_frac crece con
    # E_reserva (a E_reserva=inf, frac_ligada_estructura->0 trivialmente sin importar
    # cuánta estructura se acreditó de verdad). Hallado en la verificación del Paso 2/3
    # (ver PROTOCOLO, adenda de implementación) -- se reporta ambas, mass-fraction es la
    # que se usa para admisibilidad y para comparar contra 4,9%/31,5%.

    curva = []
    dt_sub = dt / n_subpasos
    fallo_conservacion = False

    for step in range(n_pasos_estructura):
        T_actual = _T_reloj(step, tasa_expansion)
        for _sub in range(n_subpasos):
            acc = grav.aceleraciones(pos, masa)
            vel = vel + acc * dt_sub
            if n_bar and incluir_presion_termica:
                vel[:n_bar] = vel[:n_bar] + h2.kick_termico(escala=0.02) * np.sqrt(dt_sub)
            pos = pos + vel * dt_sub
            if not np.all(np.isfinite(pos)):
                return dict(ok=False, nota=f"NaN/inf en las posiciones al paso {step}")
        if n_bar:
            rho_local = h2._densidad_local_dinamica(pos[:n_bar])
            h2.actualizar(pos[:n_bar], rho_externo=rho_local)
        else:
            rho_local = np.zeros(0)
        factor = expansion.paso_de_estiramiento(T_actual)
        if factor != 1.0:
            pos = pos * factor

        # --- Regla 4: FoF + cobro de ligadura (el gate causal, protocolo §3) ---
        a_actual = expansion._a_prev if expansion_on else 1.0
        linking_length = 0.2 * a_actual
        clusters = _fof(pos, linking_length, min_miembros=min_miembros_ligadura)
        for miembros in clusters:
            ke_int = _ke_interno_relativo(vel, masa, miembros)
            pe_int = _pe_interno(pos, vel, masa, miembros)
            if (ke_int + pe_int) >= 0.0:
                continue  # no ligado (criterio virial)
            nuevos = np.array([i for i in miembros if i not in acreditados])
            if len(nuevos) == 0:
                continue
            costo = abs(pe_int) * (len(nuevos) / len(miembros))
            if costo <= reserva_restante:
                reserva_restante -= costo
                ligada_acum += costo
                acreditados.update(int(i) for i in nuevos)
                masa_acreditada += float(np.sum(masa[nuevos[nuevos < n_bar]])) if n_bar else 0.0
            # si no alcanza: NO se acreditan (no cuentan como materia); no se marcan como
            # "intentados" tampoco -- seguirán reintentando en pasos siguientes, nunca se
            # cobra dos veces porque sólo se descuenta cuando costo<=reserva_restante.

        # --- Regla 1: auditoría de conservación (nunca asumida, medida cada paso) ---
        KE = 0.5 * float(np.sum(masa * np.sum(vel ** 2, axis=1)))
        et_actual = energia_total(pos, vel, masa, np.full(n_tot, SOFTENING), G_ADIM)
        PE = et_actual - KE
        if np.isfinite(E_total0):
            residual = E_total0 - (KE + PE) - reserva_restante - ligada_acum
            residual_rel = abs(residual) / denom_frac if denom_frac else abs(residual)
        else:
            # E_reserva=inf (brazo "sin energía" de admisibilidad): el chequeo de
            # conservación no es matemáticamente evaluable en ese límite -- no se audita
            # aquí, se audita en el brazo finito gemelo (protocolo §6).
            residual, residual_rel = 0.0, 0.0

        # --- Regla 2: exergía (depende de las diferencias) ---
        X = float(np.mean((rho_local / max(float(rho_local.mean()), 1e-12) - 1.0) ** 2)) if len(rho_local) else 0.0

        fila = dict(t=step, KE=KE, PE=PE, X=X, reserva_restante=reserva_restante,
                    ligada_acum=ligada_acum, residual=residual, residual_rel=residual_rel)
        if guardar_curva:
            curva.append(fila)
        elif step == n_pasos_estructura - 1:
            curva.append(fila)  # siempre se guarda al menos el último paso

        if (not expansion_on) and (not cooling_on) and residual_rel > tol_deriva_control:
            fallo_conservacion = True

    elapsed = time.time() - t0_wall
    final = curva[-1]
    frac_ligada = final["ligada_acum"] / denom_frac if denom_frac else 0.0
    frac_reserva = (final["reserva_restante"] / denom_frac
                     if np.isfinite(final["reserva_restante"]) and denom_frac else 0.0)
    frac_mecanica = (final["KE"] + final["PE"]) / denom_frac if denom_frac else 0.0

    a_final = expansion._a_prev if expansion_on else 1.0
    clusters_final = _fof(pos[:n_bar], 0.2 * a_final, min_miembros=2) if n_bar else []
    frac_masa_ligada = masa_acreditada / masa_bariones_total if masa_bariones_total else 0.0

    # --- fragmentación (añadido para Experimentos A/B, aditivo -- no cambia ningún campo
    # ya reportado de cs074 original): tamaño de cada grumo final en masa, y qué fracción
    # de la masa bariónica vive en el grumo más grande (1.0 = todo en un solo grumo,
    # ->0 = repartida en muchos grumos chicos).
    masas_clusters_finales = ([float(np.sum(masa[c])) for c in clusters_final]
                               if clusters_final else [])
    frac_masa_en_mayor_cluster = (max(masas_clusters_finales) / masa_bariones_total
                                   if masas_clusters_finales and masa_bariones_total else 0.0)

    return dict(
        ok=True, n_bariones=n_bar, n_cdm=n_cdm,
        E_total0=E_total0, denom_frac=denom_frac, E_reserva_abs=E_reserva_abs,
        mecanica_ref=mecanica_ref,
        frac_masa_ligada=frac_masa_ligada,  # OBSERVABLE PRIMARIO (candidata a "materia")
        masa_acreditada=masa_acreditada, masa_bariones_total=masa_bariones_total,
        frac_ligada_estructura=frac_ligada, frac_reserva_no_gastada=frac_reserva,
        frac_mecanica_residual=frac_mecanica,
        n_particulas_acreditadas=len(acreditados),
        n_clusters_finales=len(clusters_final),
        masas_clusters_finales=masas_clusters_finales,
        frac_masa_en_mayor_cluster=frac_masa_en_mayor_cluster,
        X_final=final["X"], X_inicial=curva[0]["X"] if curva else None,
        fallo_conservacion_control=fallo_conservacion,
        max_residual_rel=max((c["residual_rel"] for c in curva), default=None),
        curva=curva if guardar_curva else None,
        params=dict(nq=nq, naq=naq, ne=ne, npos=npos, amp_rugosidad=amp_rugosidad,
                    E_reserva=E_reserva, cdm_on=cdm_on, cooling_on=cooling_on,
                    expansion_on=expansion_on, gravedad_on=gravedad_on, energia_on=energia_on,
                    tasa_expansion=tasa_expansion, seed_layout=seed_layout,
                    n_pasos_estructura=n_pasos_estructura,
                    tasa_enfriamiento=tasa_enfriamiento, seed_dens_null=seed_dens_null),
        elapsed_s=elapsed,
    )


def correr_admisibilidad(E_reserva=1.0, **kw):
    """Protocolo §6: mismo punto (misma semilla/palancas), E_reserva finita vs infinita
    (energia_on=False). Se compara frac_masa_ligada (denominador FIJO = masa bariónica
    total, NO depende de E_reserva) -- deben diferir si el presupuesto actúa de verdad
    (no decorativo). NO se usa frac_ligada_estructura aquí: su denominador crece con
    E_reserva, así que a E_reserva=inf da 0.0 trivialmente sin importar cuánta estructura
    se acreditó de verdad (hallado en la verificación, ver nota en correr_holistico_energia)."""
    finita = correr_holistico_energia(E_reserva=E_reserva, energia_on=True,
                                       guardar_curva=False, **kw)
    infinita = correr_holistico_energia(energia_on=False, guardar_curva=False, **kw)
    difieren = None
    if finita.get("ok") and infinita.get("ok"):
        difieren = abs(finita["frac_masa_ligada"] - infinita["frac_masa_ligada"]) > 1e-9
    return dict(E_reserva=E_reserva, finita=finita, infinita=infinita, difieren=difieren)


def correr_barrido(amp_rugosidad_list, E_reserva_list, semillas=range(3), **kw_fijos):
    """Protocolo §5: ε × E_reserva × semillas, todo junto por punto (nunca piezas por
    separado). kw_fijos puede fijar cdm_on/cooling_on/expansion_on/gravedad_on/etc."""
    filas = []
    for amp in amp_rugosidad_list:
        for er in E_reserva_list:
            for s in semillas:
                r = correr_holistico_energia(amp_rugosidad=amp, E_reserva=er,
                                              seed_layout=12345 + s, guardar_curva=False,
                                              **kw_fijos)
                r["seed"] = s
                filas.append(r)
    return filas


AMP_RUGOSIDAD_BARRIDO = [0.5, 1.0, 1.5, 2.5, 4.0]                    # protocolo §5
E_RESERVA_BARRIDO = list(np.logspace(-3, 3, 7))                     # protocolo §5
SEMILLAS_BARRIDO = list(range(8))                                    # protocolo §5
E_RESERVA_CONTROL_PIEZAS = 1.0                                       # punto medio, controles §5
AMP_RUGOSIDAD_CONTROL_PIEZAS = 1.5


def correr_protocolo_completo(nq=300, naq=210, ne=100, npos=70, pasos_basal=150,
                               log_fn=print):
    """El barrido COMPLETO pre-registrado (protocolo §5) + admisibilidad en cada punto base
    (§6) + los 4 controles de apagado de una pieza a la vez (§5/§7 T-admisibilidad). Se
    entrega crudo, sin adjudicar (protocolo §8).

    Optimización (no cambia el diseño, sólo el orden de cómputo): el brazo 'infinita' de
    la admisibilidad NO depende de E_reserva (energia_on=False la ignora) -- así que se
    calcula UNA vez por (ε, semilla) en vez de una vez por cada punto (ε, E_reserva,
    semilla), evitando recomputar la misma física 7 veces de más."""
    t0 = time.time()

    log_fn(f"[protocolo] barrido: {len(AMP_RUGOSIDAD_BARRIDO)} eps x "
           f"{len(E_RESERVA_BARRIDO)} E_reserva x {len(SEMILLAS_BARRIDO)} semillas = "
           f"{len(AMP_RUGOSIDAD_BARRIDO)*len(E_RESERVA_BARRIDO)*len(SEMILLAS_BARRIDO)} corridas finitas "
           f"+ {len(AMP_RUGOSIDAD_BARRIDO)*len(SEMILLAS_BARRIDO)} infinitas + "
           f"4x{len(SEMILLAS_BARRIDO)} controles de pieza")

    # --- brazo infinito, una vez por (eps, semilla) ---
    infinitas = {}
    n_inf = 0
    for amp in AMP_RUGOSIDAD_BARRIDO:
        for s in SEMILLAS_BARRIDO:
            r = correr_holistico_energia(nq=nq, naq=naq, ne=ne, npos=npos,
                                          pasos_basal=pasos_basal, amp_rugosidad=amp,
                                          energia_on=False, seed_layout=12345 + s,
                                          guardar_curva=False)
            infinitas[(amp, s)] = r
            n_inf += 1
    log_fn(f"[protocolo] brazo infinito: {n_inf} corridas, "
           f"{sum(1 for r in infinitas.values() if r.get('ok'))} ok, t={time.time()-t0:.0f}s")

    # --- barrido principal (finito) + admisibilidad contra su gemelo infinito ---
    filas = []
    n_done = 0
    n_total = len(AMP_RUGOSIDAD_BARRIDO) * len(E_RESERVA_BARRIDO) * len(SEMILLAS_BARRIDO)
    for amp in AMP_RUGOSIDAD_BARRIDO:
        for er in E_RESERVA_BARRIDO:
            for s in SEMILLAS_BARRIDO:
                r = correr_holistico_energia(nq=nq, naq=naq, ne=ne, npos=npos,
                                              pasos_basal=pasos_basal, amp_rugosidad=amp,
                                              E_reserva=er, energia_on=True,
                                              seed_layout=12345 + s, guardar_curva=False)
                r["seed"] = s
                inf = infinitas.get((amp, s), {})
                if r.get("ok") and inf.get("ok"):
                    r["frac_masa_ligada_infinita"] = inf["frac_masa_ligada"]
                    r["admisibilidad_difieren"] = (
                        abs(r["frac_masa_ligada"] - inf["frac_masa_ligada"]) > 1e-9)
                filas.append(r)
                n_done += 1
        log_fn(f"[protocolo] eps={amp} listo ({n_done}/{n_total}) t={time.time()-t0:.0f}s")

    # --- controles de admisibilidad de piezas (protocolo §5/§7): apagar una a la vez ---
    controles = {}
    for palanca in ("cdm_on", "cooling_on", "expansion_on", "gravedad_on"):
        filas_ctrl = []
        for s in SEMILLAS_BARRIDO:
            kw = {"cdm_on": True, "cooling_on": True, "expansion_on": True, "gravedad_on": True}
            kw[palanca] = False
            r = correr_holistico_energia(nq=nq, naq=naq, ne=ne, npos=npos,
                                          pasos_basal=pasos_basal,
                                          amp_rugosidad=AMP_RUGOSIDAD_CONTROL_PIEZAS,
                                          E_reserva=E_RESERVA_CONTROL_PIEZAS,
                                          seed_layout=12345 + s, guardar_curva=False, **kw)
            r["seed"] = s
            filas_ctrl.append(r)
        controles[palanca] = filas_ctrl
        log_fn(f"[protocolo] control {palanca}=False: "
               f"{sum(1 for r in filas_ctrl if r.get('ok'))}/{len(filas_ctrl)} ok, "
               f"t={time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    log_fn(f"[protocolo] TOTAL elapsed={elapsed:.0f}s")
    return dict(filas=filas, infinitas={f"{k[0]}_{k[1]}": v for k, v in infinitas.items()},
                controles_piezas=controles, elapsed_s=elapsed,
                grid=dict(amp_rugosidad=AMP_RUGOSIDAD_BARRIDO, E_reserva=E_RESERVA_BARRIDO,
                          semillas=SEMILLAS_BARRIDO))


def main():
    """Verificación en 3 pasos (plan aprobado): (1) smoke test de conservación en gravedad
    pura, (2) chequeo aislado de Regla 4 (reserva chica vs grande), (3) barrido reducido +
    admisibilidad, para validar el motor entero antes de escalar al barrido completo."""
    log = []

    def p(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log.append(line)

    p("=== PASO 1: smoke test, gravedad pura (sin expansión, sin enfriamiento) ===")
    smoke = correr_holistico_energia(
        nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5,
        E_reserva=1.0, expansion_on=False, cooling_on=False, cdm_on=True,
        incluir_presion_termica=False, n_pasos_estructura=60,
    )
    p(f"ok={smoke.get('ok')} n_bariones={smoke.get('n_bariones')} "
      f"max_residual_rel={smoke.get('max_residual_rel')} "
      f"fallo_conservacion_control={smoke.get('fallo_conservacion_control')} "
      f"frac_masa_ligada={smoke.get('frac_masa_ligada')} elapsed={smoke.get('elapsed_s'):.1f}s"
      if smoke.get("ok") else f"FALLÓ: {smoke.get('nota')}")

    p("=== PASO 2: Regla 4 aislada -- reserva chica vs grande (mismo escenario) ===")
    chica = correr_holistico_energia(
        nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5,
        E_reserva=1e-3, guardar_curva=False,
    )
    grande = correr_holistico_energia(
        nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5,
        E_reserva=1e3, guardar_curva=False,
    )
    p(f"reserva chica: frac_masa_ligada={chica.get('frac_masa_ligada')} "
      f"n_acreditadas={chica.get('n_particulas_acreditadas')}")
    p(f"reserva grande: frac_masa_ligada={grande.get('frac_masa_ligada')} "
      f"n_acreditadas={grande.get('n_particulas_acreditadas')}")
    regla4_ok = (chica.get("ok") and grande.get("ok") and
                 chica["frac_masa_ligada"] <= grande["frac_masa_ligada"])
    p(f"Regla 4 direccionalmente correcta (chica <= grande, por masa): {regla4_ok}")

    p("=== PASO 3: barrido reducido + admisibilidad ===")
    barrido = correr_barrido(
        amp_rugosidad_list=[0.5, 1.5], E_reserva_list=[1e-2, 1.0, 1e2], semillas=range(2),
        nq=300, naq=210, ne=100, npos=70, pasos_basal=150,
    )
    # E_reserva=1e-3 (no 1.0): el Paso 2 mostró que a multiplo=1.0 la reserva ya está
    # "saturada" para este escenario (mismo resultado que infinita) -- la prueba de
    # admisibilidad es informativa en la zona ESCASA, donde el gate realmente puede
    # bloquear estructura (protocolo §6, adenda de implementación).
    admis = correr_admisibilidad(
        E_reserva=1e-3, nq=300, naq=210, ne=100, npos=70, pasos_basal=150, amp_rugosidad=1.5,
    )
    p(f"barrido reducido: {len(barrido)} corridas, "
      f"{sum(1 for r in barrido if r.get('ok'))} ok")
    p(f"admisibilidad: difieren={admis.get('difieren')} "
      f"finita.frac_masa_ligada={admis['finita'].get('frac_masa_ligada')} "
      f"infinita.frac_masa_ligada={admis['infinita'].get('frac_masa_ligada')}")

    resultado = dict(
        experimento="cs074_energia_holistica -- smoke+verificacion",
        smoke_test=smoke, regla4_chica=chica, regla4_grande=grande, regla4_ok=regla4_ok,
        barrido_reducido=barrido, admisibilidad=admis, log=log,
    )
    out_json = OUT / "cs074_verificacion_result.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    p(f"[archivo] {out_json}")


def analizar_balance(filas):
    """Observable holístico (protocolo §4): balance de fracciones sobre las corridas OK
    del barrido principal, y comparación con 4,9%/31,5% SOLO como salida (T-target, nunca
    se ajusta nada para acercarse)."""
    ok = [r for r in filas if r.get("ok")]
    if not ok:
        return dict(n_ok=0)
    fm = np.array([r["frac_masa_ligada"] for r in ok])
    fallos_cons = sum(1 for r in ok if r.get("fallo_conservacion_control"))
    difieren = [r["admisibilidad_difieren"] for r in ok if "admisibilidad_difieren" in r]
    cerca_49 = [(r["params"]["amp_rugosidad"], r["params"]["E_reserva"], r["frac_masa_ligada"])
                for r in ok if abs(r["frac_masa_ligada"] - 0.049) < 0.01]
    cerca_315 = [(r["params"]["amp_rugosidad"], r["params"]["E_reserva"], r["frac_masa_ligada"])
                 for r in ok if abs(r["frac_masa_ligada"] - 0.315) < 0.01]
    return dict(
        n_ok=len(ok), n_total=len(filas),
        frac_masa_ligada_media=float(fm.mean()), frac_masa_ligada_std=float(fm.std()),
        frac_masa_ligada_min=float(fm.min()), frac_masa_ligada_max=float(fm.max()),
        fallos_conservacion_control=fallos_cons,
        fraccion_admisibilidad_difieren=(sum(difieren) / len(difieren)) if difieren else None,
        n_celdas_cerca_4_9pct=len(cerca_49), celdas_cerca_4_9pct=cerca_49[:20],
        n_celdas_cerca_31_5pct=len(cerca_315), celdas_cerca_31_5pct=cerca_315[:20],
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        log_lines = []

        def _log(msg):
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
            print(line, file=sys.stderr, flush=True)
            log_lines.append(line)

        resultado = correr_protocolo_completo(log_fn=_log)
        resultado["balance"] = analizar_balance(resultado["filas"])
        resultado["log"] = log_lines
        out_json = OUT / "cs074_barrido_completo_result.json"
        out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str),
                             encoding="utf-8")
        _log(f"[archivo] {out_json}")
        _log(f"[balance] {json.dumps(resultado['balance'], indent=2, ensure_ascii=False, default=str)}")
    else:
        main()
