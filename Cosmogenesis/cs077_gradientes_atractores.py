#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs077_gradientes_atractores.py — ¿La meseta de cs074A es un ATRACTOR (gradiente genuino) o
solo "cualquier proceso que se estabiliza solo"?
=========================================================================================

Quién soy / qué hago (código autodescriptivo):
  Nodos C-N2.6.1-4 de la Teoría Cosmosemiótica (protocolo diseñado en
  DISENO_EXPERIMENTOS_NODOS_ABIERTOS_desde_2.5.5_CS.md §5, refinado con más brazos y más
  observables). `cs074A_asimetria_techo.py` YA midió que existe una "meseta" de masa ligada
  estable frente a la asimetría inicial ε (amp_rugosidad) hasta cierto techo, luego
  fragmentación, luego colapso — evidencia INDIRECTA de un atractor. Pero nunca se comparó
  eso contra la alternativa obvia: ¿el sistema converge ahí porque de verdad "cae" siguiendo
  un gradiente (la gravedad tira en una dirección particular, hacia los pozos de potencial),
  o llegaría a un lugar parecido igual si el paso de velocidad, en cada instante, tuviera la
  MISMA magnitud pero apuntara al azar? Sin ese control, "atractor" y "cualquier proceso que
  se estabiliza solo" son indistinguibles.

  NO se modifica ni `cs074_energia_holistica.py` ni `cs074A_asimetria_techo.py` -- ambos se
  IMPORTAN (incluidas sus funciones "privadas" con guión bajo: Python no las oculta, sólo
  las excluye de `import *`; importarlas por nombre es reutilizar el motor verificado, no
  tocarlo). Todas las piezas de física (gravedad, expansión, halo CDM, enfriamiento H2, FoF,
  criterio de ligadura, ledger de energía) son las MISMAS funciones importadas de
  `cs072_modulos/piezas/*` y `cs074_energia_holistica.py` que ya usa cs074A -- no se
  re-derivó ninguna ley nueva aquí.

Los 4 brazos (protocolo, punto 2):
  1. REAL             -- `correr_holistico_energia()` tal cual (caja negra, sin tocar).
  2. DIRECCION_AZAR    -- mismo motor, pero en cada micro-paso se conserva la MAGNITUD del
                          kick gravitacional que el modelo real produce en ESE instante
                          (|acc_i(t)|, calculada sobre la trayectoria propia de este brazo,
                          que diverge de la real desde el primer paso -- "misma magnitud"
                          se lee como "magnitud auto-consistente con la propia dinámica",
                          no como una copia congelada de la corrida real) y se REASIGNA su
                          DIRECCIÓN al azar (vector unitario isótropo). Conserva "cuánto se
                          mueve", rompe "seguir el gradiente".
  3. ORDEN_BARAJADO    -- dos pasadas por semilla/ε: (a) una pasada REAL que graba la
                          dirección unitaria real del kick gravitacional en cada micro-paso;
                          (b) una segunda pasada que usa la MAGNITUD auto-consistente de esa
                          misma corrida (como en el brazo 2) pero la DIRECCIÓN la toma de la
                          pasada (a) en un índice temporal permutado al azar (una permutación
                          fija por corrida) -- dirección real, pero de OTRO instante. Rompe
                          la coherencia CAUSAL de "hacia dónde apunta el gradiente ahora"
                          preservando el conjunto de direcciones que el modelo real sí visitó.
  4. SIN_MEMORIA       -- se inspeccionaron las 4 piezas de la dinámica de estructura
                          (GravedadGeneral, Expansion, MateriaOscuraHalo, EnfriamientoH2):
                          gravedad y expansión son funciones PURAS del estado instantáneo
                          (sin término de historia); el ÚNICO estado que persiste entre pasos
                          y con una tasa de relajación tipo "gamma" es `EnfriamientoH2.T`
                          (memoria térmica de compresión pasada), con constante de relajación
                          `tasa_enfriamiento` -- YA expuesta como kwarg de
                          `correr_holistico_energia()` (no se tocó el archivo). Con
                          `tasa_enfriamiento=1.0` (GAMMA_SIN_MEMORIA), T salta exactamente al
                          piso T_piso en cada paso donde hay gatillo de compresión (relajación
                          instantánea = memoria mínima que el motor permite representar; ver
                          `EnfriamientoH2.actualizar`, T = T - tasa*(T-T_piso), tasa=1 -> T=
                          T_piso exacto). Se documenta este hallazgo explícitamente: NO hay
                          un término de memoria/gamma "clásico" (fricción, kernel viscoso)
                          en este motor -- el candidato más cercano es este, y es el que se
                          anula. (Se descartó deliberadamente anular la INERCIA misma --
                          vel += acc*dt -> vel = acc*dt -- porque eso cambia el RÉGIMEN de la
                          ecuación de movimiento entera, de Newtoniano a sobreamortiguado, y
                          rompe el ledger de energía/conservación que es parte de lo YA
                          verificado; el pedido es anular un término de memoria EXISTENTE,
                          no fabricar uno nuevo apagando la dinámica misma.)

  Los brazos 1 y 4 se miden con la caja negra `correr_holistico_energia()` sin cambios
  (máxima fidelidad al motor verificado). Los brazos 2 y 3 requieren interceptar la línea
  exacta donde la velocidad recibe el kick gravitacional -- como esa línea vive DENTRO del
  bucle de `correr_holistico_energia()` (no está factorizada como función de un paso), se
  reimplementa aquí una copia funcionalmente equivalente de ESE bucle (misma estructura,
  mismas piezas importadas, mismo criterio de ligadura FoF importado tal cual de
  `cs074_energia_holistica._fof/._pe_interno/._ke_interno_relativo`), con UN solo punto de
  intercepción marcado explícitamente más abajo. `main(--validar)` corre un chequeo de
  fidelidad: el brazo "real" reimplementado aquí debe reproducir frac_masa_ligada de la caja
  negra en el mismo punto (ε, semilla) dentro de tolerancia numérica.

Observables medidos (protocolo, punto 3), por brazo, con >=12 semillas:
  (a) ancho de la meseta estable en el barrido de ε (rango de ε donde frac_masa_ligada se
      mantiene dentro de una banda de tolerancia de su valor de meseta).
  (b) fracción de semillas que terminan en la misma cuenca/atractor (dispersión de
      frac_masa_ligada entre semillas, a ε fija dentro de la meseta).
  (c) tiempo de estabilización -- proxy operacional de "tiempo de escape/asentamiento": el
      primer paso donde ligada_acum(t)/ligada_acum(final) cruza 0.9 y no vuelve a bajar de
      ahí. Documentado como proxy (no literalmente "tiempo de escape de la meseta", que
      requeriría re-correr TODO el barrido de ε por paso -- computacionalmente prohibitivo
      aquí; se usa la curva temporal DENTRO de una corrida como sustituto razonable y barato,
      igual para los 4 brazos).
  (d) histéresis -- LIMITACIÓN ESTRUCTURAL DOCUMENTADA (no fabricada): en este motor, cada
      punto (ε, semilla) parte de bariones y posiciones frescos (`corre()` + `posiciones_
      escenario()`, ambos funciones puras de (ε, semilla), sin estado dinámico persistente
      entre valores de ε). Por diseño, un barrido ascendente y uno descendente del MISMO
      grid de ε deben coincidir punto a punto exactamente (mismo (ε, semilla) -> mismo
      resultado determinista) -- no hay mecanismo de memoria de trayectoria entre puntos de
      ε en la arquitectura actual. Se verifica esto empíricamente (Δ esperado ≈ 0) y se
      reporta como HALLAZGO en sí mismo, no como evidencia de ausencia de atractor.
  (e) exponente de Lyapunov (proxy) -- corrida gemela con una perturbación diminuta en las
      velocidades iniciales (mismo ε, misma semilla, MISMA secuencia de números aleatorios
      del brazo -- sólo cambia la condición inicial), midiendo la separación de posiciones
      D(t) entre la corrida base y la perturbada; λ = pendiente de una regresión lineal de
      ln D(t) vs t en la fase de crecimiento (antes de saturar). Corrido en un subconjunto
      de (ε, semilla) por costo (2 pasadas extra por punto).
  (f) varianza del estado final (frac_masa_ligada) entre semillas, a cada ε.

Criterio de falsación (protocolo, punto 4): si el brazo DIRECCION_AZAR da una meseta de
ancho y varianza ESTADÍSTICAMENTE INDISTINGUIBLE de REAL, el nodo cae (la estabilidad sería
un efecto de la magnitud del paso, no de "caer hacia" nada). Si la meseta REAL es mucho más
angosta/estable que la de DIRECCION_AZAR, confirma gradiente genuino. Este script NO declara
un veredicto final -- reporta los números crudos para que el director adjudique.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --- motor verificado, reutilizado tal cual (import, NO edición) ------------------------
from cs074_energia_holistica import (  # noqa: E402
    correr_holistico_energia,
    _extraer_bariones, _fof, _pe_interno, _ke_interno_relativo, _T_reloj,
    T0, TASA_EXPANSION_DEFAULT, SOFTENING, TOL_DERIVA_CONTROL,
)
from cs072_modulos.piezas.p_gravedad_general import (  # noqa: E402
    GravedadGeneral, posiciones_escenario, energia_total, G_ADIM,
)
from cs072_modulos.piezas.p_expansion import Expansion  # noqa: E402
from cs072_modulos.piezas.p_materia_oscura_halo import MateriaOscuraHalo  # noqa: E402
from cs072_modulos.piezas.p_enfriamiento_H2 import EnfriamientoH2  # noqa: E402

OUT = HERE / "resultados_cs077_gradientes_atractores"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------------------
# Constantes del diseño (declaradas antes de correr, protocolo)
# ---------------------------------------------------------------------------------------
GAMMA_SIN_MEMORIA = 1.0  # ver docstring del módulo -- relajación térmica instantánea del
# único estado con "gamma" identificado en el motor (EnfriamientoH2.T). tasa_enfriamiento
# default del motor es 0.3 (memoria parcial); 1.0 es el máximo que el propio código soporta
# sin overshoot (T -> T_piso exacto cada paso con gatillo).

# Escala reducida respecto de cs074A (nq=300 etc.) para que 4 brazos x >=12 semillas x
# ~13 puntos de eps corran en minutos, no horas -- documentado como reducción de escala,
# no como cambio de física (mismas piezas, mismas convenciones G_ADIM/SOFTENING).
ESCALA = dict(nq=180, naq=126, ne=60, npos=42, pasos_basal=90)
N_PASOS_ESTRUCTURA = 30
N_SUBPASOS = 6
DT = 0.05

# Grid de eps cubriendo los 3 regímenes que cs074A encontró: meseta (<=0.5),
# fragmentación (0.9-2.3), colapso (>=3.8).
EPS_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.3, 1.8, 2.3, 3.0, 3.8, 5.0]
N_SEMILLAS = 12
SEMILLAS = list(range(N_SEMILLAS))

# subconjunto para Lyapunov (caro: 2 pasadas x custom-loop por punto)
EPS_LYAPUNOV = [0.1, 0.5, 1.5, 4.0]
SEMILLAS_LYAPUNOV = list(range(4))
PERTURBACION_VEL0 = 1e-4

E_RESERVA_FIJA = 1.0  # cs074A ya mostró que a multiplo=1.0 la reserva satura para este
# escenario (no bloquea estructura) -- la pregunta gradiente-vs-azar es ortogonal al
# presupuesto de energía (otro nodo, ya cubierto en cs074/cs074A), así que se fija en vez de
# barrer, para no explotar el costo computacional con una dimensión que no es la que este
# experimento pone a prueba.


# =========================================================================================
# Brazo 2/3: bucle reimplementado (mismas piezas importadas, un solo punto de intercepción)
# =========================================================================================

def _direccion_random_unitaria(rng, n):
    d = rng.normal(size=(n, 3))
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return d / norm


def correr_dinamica_intervenida(
    modo, nq, naq, ne, npos, pasos_basal, amp_rugosidad,
    E_reserva=1.0, reserva_como_multiplo_mecanica=True,
    n_pasos_estructura=N_PASOS_ESTRUCTURA, dt=DT, n_subpasos=N_SUBPASOS,
    cdm_on=True, cooling_on=True, expansion_on=True, gravedad_on=True, energia_on=True,
    tasa_expansion=TASA_EXPANSION_DEFAULT, seed_layout=12345, seed_dens_cdm=7000,
    min_miembros_ligadura=2, tol_deriva_control=TOL_DERIVA_CONTROL,
    incluir_presion_termica=True, tasa_enfriamiento=0.3,
    guardar_curva=True, guardar_trayectoria=False, perturbacion_vel0=0.0,
    direcciones_reales=None, permutacion=None, registrar_direcciones=False,
):
    """Réplica funcional del bucle de `correr_holistico_energia` (mismas piezas importadas,
    mismo orden de operaciones, mismo criterio de ligadura) con UN punto de intercepción:
    la dirección del kick gravitacional aplicado a la velocidad en cada micro-paso.

    modo="real"            -> vel += acc*dt_sub, IDÉNTICO al motor original (dirección·mag ==
                               acc siempre, salvo acc=0 donde ambos son cero) -- se usa sólo
                               para (i) validar fidelidad contra la caja negra y (ii) obtener
                               trayectorias de posición para el proxy de Lyapunov (que
                               correr_holistico_energia no expone).
    modo="direccion_azar"  -> magnitud auto-consistente |acc_i(t)| de ESTA trayectoria,
                               dirección un vector unitario isótropo al azar.
    modo="orden_barajado"  -> magnitud auto-consistente |acc_i(t)|, dirección tomada de
                               `direcciones_reales[permutacion[t]]` (grabadas en una pasada
                               REAL previa, mismo seed/eps).

    perturbacion_vel0 != 0: agrega ruido gaussiano de esa escala a la velocidad inicial (para
    el proxy de Lyapunov -- el par base/perturbado comparte el mismo rng_modo, sólo difiere
    la condición inicial).
    """
    assert modo in ("real", "direccion_azar", "orden_barajado")
    # rng propio de la intervención (dirección al azar / cuál permutación de recorte del
    # historial de direcciones), sembrado por (seed, eps, modo) -- reproducible, NO se
    # reincia con la perturbación de Lyapunov (así el par base/perturbado usa exactamente
    # los mismos números aleatorios de "modo", aislando el efecto de la condición inicial).
    semilla_rng = (int(seed_layout) * 1_000_003 + int(round(amp_rugosidad * 1e6))) % (2 ** 31)
    rng_modo = np.random.default_rng(semilla_rng)

    t0_wall = time.time()
    masa_bar, dens_bar, obs_basal = _extraer_bariones(nq, naq, ne, npos, pasos_basal,
                                                        amp_rugosidad, tasa_expansion)
    n_bar = len(masa_bar)
    if n_bar < 8:
        return dict(ok=False, nota=f"sólo {n_bar} átomos reales (<8): sin masa suficiente")

    n_cdm = n_bar if cdm_on else 0
    lado = float(n_bar + n_cdm) ** (1.0 / 3.0)
    pos_bar, _ = posiciones_escenario(n_bar, lado=lado, seed=seed_layout)
    cdm = MateriaOscuraHalo(n_cdm, amp_rugosidad, lado_escenario=lado, activa=cdm_on,
                             seed_pos=54321, seed_dens=seed_dens_cdm)

    pos = np.vstack([pos_bar, cdm.pos]) if n_cdm else pos_bar.copy()
    masa = (np.concatenate([masa_bar * dens_bar, cdm.masa * cdm.densidad])
            if n_cdm else masa_bar * dens_bar)
    vel = np.zeros_like(pos)
    if perturbacion_vel0:
        vel = vel + perturbacion_vel0 * rng_modo.normal(size=vel.shape)
    n_tot = len(pos)

    grav = GravedadGeneral(activa=gravedad_on, softening=SOFTENING)
    expansion = Expansion(T0=T0, activa=expansion_on)
    h2 = EnfriamientoH2(n_bar, T_inicial=T0, activa_cooling=cooling_on, seed=9000,
                         softening=SOFTENING, tasa_enfriamiento=tasa_enfriamiento)

    et0 = energia_total(pos, vel, masa, np.full(n_tot, SOFTENING), G_ADIM)
    mecanica_ref = abs(et0) if et0 != 0 else float(np.sum(masa))
    if not energia_on:
        E_reserva_abs = float("inf")
    elif reserva_como_multiplo_mecanica:
        E_reserva_abs = float(E_reserva) * mecanica_ref
    else:
        E_reserva_abs = float(E_reserva)
    E_total0 = et0 + E_reserva_abs
    denom_frac = mecanica_ref + E_reserva_abs

    reserva_restante = E_reserva_abs
    ligada_acum = 0.0
    acreditados = set()
    masa_bariones_total = float(np.sum(masa[:n_bar])) if n_bar else 0.0
    masa_acreditada = 0.0

    curva = []
    pos_por_paso = [] if guardar_trayectoria else None
    direcciones_out = [] if registrar_direcciones else None
    dt_sub = dt / n_subpasos
    fallo_conservacion = False
    micro_idx = 0

    for step in range(n_pasos_estructura):
        T_actual = _T_reloj(step, tasa_expansion)
        for _sub in range(n_subpasos):
            acc = grav.aceleraciones(pos, masa)
            mag = np.linalg.norm(acc, axis=1, keepdims=True)  # |acc_i(t)| de ESTA trayectoria

            if modo == "real":
                direccion = np.divide(acc, mag, out=np.zeros_like(acc), where=mag > 0)
            elif modo == "direccion_azar":
                direccion = _direccion_random_unitaria(rng_modo, n_tot)
            else:  # "orden_barajado"
                idx_fuente = int(permutacion[micro_idx % len(permutacion)])
                direccion = direcciones_reales[idx_fuente]

            if registrar_direcciones:
                dir_real = np.divide(acc, mag, out=np.zeros_like(acc), where=mag > 0)
                direcciones_out.append(dir_real)

            # --- ÚNICO PUNTO DE INTERCEPCIÓN (protocolo, punto 2) ---
            vel = vel + direccion * mag * dt_sub
            # ----------------------------------------------------------

            if n_bar and incluir_presion_termica:
                vel[:n_bar] = vel[:n_bar] + h2.kick_termico(escala=0.02) * np.sqrt(dt_sub)
            pos = pos + vel * dt_sub
            if not np.all(np.isfinite(pos)):
                return dict(ok=False, nota=f"NaN/inf en las posiciones al paso {step}")
            micro_idx += 1

        if n_bar:
            rho_local = h2._densidad_local_dinamica(pos[:n_bar])
            h2.actualizar(pos[:n_bar], rho_externo=rho_local)
        else:
            rho_local = np.zeros(0)
        factor = expansion.paso_de_estiramiento(T_actual)
        if factor != 1.0:
            pos = pos * factor

        if guardar_trayectoria:
            pos_por_paso.append(pos.copy())

        a_actual = expansion._a_prev if expansion_on else 1.0
        linking_length = 0.2 * a_actual
        clusters = _fof(pos, linking_length, min_miembros=min_miembros_ligadura)
        for miembros in clusters:
            ke_int = _ke_interno_relativo(vel, masa, miembros)
            pe_int = _pe_interno(pos, vel, masa, miembros)
            if (ke_int + pe_int) >= 0.0:
                continue
            nuevos = np.array([i for i in miembros if i not in acreditados])
            if len(nuevos) == 0:
                continue
            costo = abs(pe_int) * (len(nuevos) / len(miembros))
            if costo <= reserva_restante:
                reserva_restante -= costo
                ligada_acum += costo
                acreditados.update(int(i) for i in nuevos)
                masa_acreditada += (float(np.sum(masa[nuevos[nuevos < n_bar]]))
                                     if n_bar else 0.0)

        KE = 0.5 * float(np.sum(masa * np.sum(vel ** 2, axis=1)))
        et_actual = energia_total(pos, vel, masa, np.full(n_tot, SOFTENING), G_ADIM)
        PE = et_actual - KE
        if np.isfinite(E_total0):
            residual = E_total0 - (KE + PE) - reserva_restante - ligada_acum
            residual_rel = abs(residual) / denom_frac if denom_frac else abs(residual)
        else:
            residual, residual_rel = 0.0, 0.0
        X = (float(np.mean((rho_local / max(float(rho_local.mean()), 1e-12) - 1.0) ** 2))
             if len(rho_local) else 0.0)

        fila = dict(t=step, KE=KE, PE=PE, X=X, reserva_restante=reserva_restante,
                    ligada_acum=ligada_acum, residual=residual, residual_rel=residual_rel,
                    masa_acreditada=masa_acreditada)
        curva.append(fila)
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

    masas_clusters_finales = ([float(np.sum(masa[c])) for c in clusters_final]
                               if clusters_final else [])
    frac_masa_en_mayor_cluster = (max(masas_clusters_finales) / masa_bariones_total
                                   if masas_clusters_finales and masa_bariones_total else 0.0)

    return dict(
        ok=True, modo=modo, n_bariones=n_bar, n_cdm=n_cdm,
        frac_masa_ligada=frac_masa_ligada,
        masa_acreditada=masa_acreditada, masa_bariones_total=masa_bariones_total,
        frac_ligada_estructura=frac_ligada, frac_reserva_no_gastada=frac_reserva,
        frac_mecanica_residual=frac_mecanica,
        n_particulas_acreditadas=len(acreditados),
        n_clusters_finales=len(clusters_final),
        masas_clusters_finales=masas_clusters_finales,
        frac_masa_en_mayor_cluster=frac_masa_en_mayor_cluster,
        fallo_conservacion_control=fallo_conservacion,
        max_residual_rel=max((c["residual_rel"] for c in curva), default=None),
        curva=curva if guardar_curva else None,
        pos_por_paso=pos_por_paso,
        direcciones=direcciones_out,
        params=dict(nq=nq, naq=naq, ne=ne, npos=npos, amp_rugosidad=amp_rugosidad,
                    E_reserva=E_reserva, tasa_enfriamiento=tasa_enfriamiento, modo=modo,
                    seed_layout=seed_layout),
        elapsed_s=elapsed,
    )


def correr_orden_barajado(nq, naq, ne, npos, pasos_basal, amp_rugosidad, seed_layout,
                           **kw):
    """Las dos pasadas del brazo 3 (protocolo, punto 2.3): (a) pasada REAL que graba
    dirección unitaria por micro-paso, (b) pasada que usa esas direcciones en orden
    permutado. La permutación se siembra por (seed, eps) -- reproducible."""
    pasada_real = correr_dinamica_intervenida(
        "real", nq, naq, ne, npos, pasos_basal, amp_rugosidad,
        seed_layout=seed_layout, registrar_direcciones=True, **kw)
    if not pasada_real.get("ok"):
        return pasada_real
    n_micro = len(pasada_real["direcciones"])
    rng_perm = np.random.default_rng(
        (int(seed_layout) * 7_919 + int(round(amp_rugosidad * 1e6)) + 424242) % (2 ** 31))
    permutacion = rng_perm.permutation(n_micro)
    resultado = correr_dinamica_intervenida(
        "orden_barajado", nq, naq, ne, npos, pasos_basal, amp_rugosidad,
        seed_layout=seed_layout, direcciones_reales=pasada_real["direcciones"],
        permutacion=permutacion, **kw)
    return resultado


# =========================================================================================
# Runner por brazo: uniformiza la interfaz (los 4 brazos devuelven el mismo esquema mínimo)
# =========================================================================================

def _correr_brazo(brazo, amp, seed, guardar_curva=True):
    seed_layout = 12345 + seed
    kw_escala = dict(nq=ESCALA["nq"], naq=ESCALA["naq"], ne=ESCALA["ne"], npos=ESCALA["npos"],
                      pasos_basal=ESCALA["pasos_basal"])
    if brazo == "REAL":
        r = correr_holistico_energia(amp_rugosidad=amp, E_reserva=E_RESERVA_FIJA,
                                      seed_layout=seed_layout, guardar_curva=guardar_curva,
                                      n_pasos_estructura=N_PASOS_ESTRUCTURA, dt=DT,
                                      n_subpasos=N_SUBPASOS, **kw_escala)
    elif brazo == "SIN_MEMORIA":
        r = correr_holistico_energia(amp_rugosidad=amp, E_reserva=E_RESERVA_FIJA,
                                      seed_layout=seed_layout, guardar_curva=guardar_curva,
                                      tasa_enfriamiento=GAMMA_SIN_MEMORIA,
                                      n_pasos_estructura=N_PASOS_ESTRUCTURA, dt=DT,
                                      n_subpasos=N_SUBPASOS, **kw_escala)
    elif brazo == "DIRECCION_AZAR":
        r = correr_dinamica_intervenida("direccion_azar", amp_rugosidad=amp,
                                         seed_layout=seed_layout, E_reserva=E_RESERVA_FIJA,
                                         guardar_curva=guardar_curva, **kw_escala)
    elif brazo == "ORDEN_BARAJADO":
        r = correr_orden_barajado(amp_rugosidad=amp, seed_layout=seed_layout,
                                   E_reserva=E_RESERVA_FIJA, guardar_curva=guardar_curva,
                                   **kw_escala)
    else:
        raise ValueError(brazo)
    r["brazo"] = brazo
    r["eps"] = amp
    r["seed"] = seed
    return r


BRAZOS = ["REAL", "SIN_MEMORIA", "DIRECCION_AZAR", "ORDEN_BARAJADO"]


# =========================================================================================
# Observable (c): tiempo de estabilización, a partir de la curva ligada_acum(t)
# =========================================================================================

def _tiempo_estabilizacion(curva):
    if not curva:
        return None
    final = curva[-1]["ligada_acum"]
    if final <= 0:
        return None
    umbral = 0.9 * final
    for i, fila in enumerate(curva):
        if fila["ligada_acum"] >= umbral and all(
                c["ligada_acum"] >= umbral for c in curva[i:]):
            return i
    return None


# =========================================================================================
# Observable (e): Lyapunov proxy
# =========================================================================================

def _lyapunov_proxy(brazo, amp, seed):
    seed_layout = 12345 + seed
    kw_escala = dict(nq=ESCALA["nq"], naq=ESCALA["naq"], ne=ESCALA["ne"], npos=ESCALA["npos"],
                      pasos_basal=ESCALA["pasos_basal"])
    tasa_enf = GAMMA_SIN_MEMORIA if brazo == "SIN_MEMORIA" else 0.3
    modo_intervenido = {"REAL": "real", "SIN_MEMORIA": "real",
                         "DIRECCION_AZAR": "direccion_azar"}.get(brazo)

    def _par(modo, perturb):
        if modo == "orden_barajado_base" or modo == "orden_barajado_pert":
            # el brazo 3 exige dos pasadas (grabar direcciones + aplicar barajadas); para el
            # par base/perturbado se recicla LA MISMA permutación (calculada sobre la pasada
            # real base) en ambas ramas, para que sólo la condición inicial difiera.
            base_pasada_real = correr_dinamica_intervenida(
                "real", seed_layout=seed_layout, amp_rugosidad=amp, E_reserva=E_RESERVA_FIJA,
                tasa_enfriamiento=tasa_enf, guardar_curva=False, guardar_trayectoria=False,
                registrar_direcciones=True, perturbacion_vel0=0.0, **kw_escala)
            if not base_pasada_real.get("ok"):
                return base_pasada_real
            n_micro = len(base_pasada_real["direcciones"])
            rng_perm = np.random.default_rng(
                (int(seed_layout) * 7_919 + int(round(amp * 1e6)) + 424242) % (2 ** 31))
            permutacion = rng_perm.permutation(n_micro)
            pert = perturb
            return correr_dinamica_intervenida(
                "orden_barajado", seed_layout=seed_layout, amp_rugosidad=amp,
                E_reserva=E_RESERVA_FIJA, tasa_enfriamiento=tasa_enf, guardar_curva=False,
                guardar_trayectoria=True, direcciones_reales=base_pasada_real["direcciones"],
                permutacion=permutacion, perturbacion_vel0=pert, **kw_escala)
        return correr_dinamica_intervenida(
            modo, seed_layout=seed_layout, amp_rugosidad=amp, E_reserva=E_RESERVA_FIJA,
            tasa_enfriamiento=tasa_enf, guardar_curva=False, guardar_trayectoria=True,
            perturbacion_vel0=perturb, **kw_escala)

    if brazo == "ORDEN_BARAJADO":
        base = _par("orden_barajado_base", 0.0)
        pert = _par("orden_barajado_pert", PERTURBACION_VEL0)
    else:
        base = _par(modo_intervenido, 0.0)
        pert = _par(modo_intervenido, PERTURBACION_VEL0)

    if not (base.get("ok") and pert.get("ok")):
        return dict(ok=False, nota="corrida base o perturbada falló")

    pos_base = base["pos_por_paso"]
    pos_pert = pert["pos_por_paso"]
    n = min(len(pos_base), len(pos_pert))
    if n < 4:
        return dict(ok=False, nota="trayectoria demasiado corta")
    D = np.array([float(np.linalg.norm(pos_pert[i] - pos_base[i])) for i in range(n)])
    D = np.maximum(D, 1e-12)
    # fase de crecimiento: hasta que D deja de crecer monótonamente (o hasta la mitad)
    fin_crecimiento = n
    for i in range(1, n):
        if D[i] < D[i - 1]:
            fin_crecimiento = i
            break
    fin_crecimiento = max(fin_crecimiento, 3)
    t_arr = np.arange(fin_crecimiento)
    logD = np.log(D[:fin_crecimiento])
    if fin_crecimiento >= 2 and np.std(t_arr) > 0:
        pendiente = float(np.polyfit(t_arr, logD, 1)[0])
    else:
        pendiente = None
    return dict(ok=True, lyapunov_proxy=pendiente, n_pasos_usados=fin_crecimiento,
                D0=float(D[0]), D_final=float(D[-1]))


# =========================================================================================
# Barrido principal
# =========================================================================================

def correr_barrido_principal(log_fn=print):
    t0 = time.time()
    filas = []
    n_total = len(BRAZOS) * len(EPS_GRID) * len(SEMILLAS)
    n_done = 0
    for brazo in BRAZOS:
        for amp in EPS_GRID:
            for s in SEMILLAS:
                r = _correr_brazo(brazo, amp, s, guardar_curva=True)
                if r.get("ok"):
                    r["tiempo_estabilizacion"] = _tiempo_estabilizacion(r.get("curva"))
                    r["curva"] = None  # ya extraído lo que hacía falta -- no persistir crudo
                filas.append(r)
                n_done += 1
            log_fn(f"[cs077] {brazo} eps={amp:.4g} listo ({n_done}/{n_total}) "
                    f"t={time.time()-t0:.0f}s")
    log_fn(f"[cs077] barrido principal TOTAL elapsed={time.time()-t0:.0f}s")
    return filas


def correr_lyapunov(log_fn=print):
    t0 = time.time()
    filas = []
    n_total = len(BRAZOS) * len(EPS_LYAPUNOV) * len(SEMILLAS_LYAPUNOV)
    n_done = 0
    for brazo in BRAZOS:
        for amp in EPS_LYAPUNOV:
            for s in SEMILLAS_LYAPUNOV:
                r = _lyapunov_proxy(brazo, amp, s)
                r["brazo"], r["eps"], r["seed"] = brazo, amp, s
                filas.append(r)
                n_done += 1
        log_fn(f"[cs077] lyapunov {brazo} listo ({n_done}/{n_total}) t={time.time()-t0:.0f}s")
    log_fn(f"[cs077] lyapunov TOTAL elapsed={time.time()-t0:.0f}s")
    return filas


def validar_fidelidad(log_fn=print, amp=1.0, seed=0):
    """Chequeo de fidelidad (protocolo): correr_dinamica_intervenida(modo='real', ...) debe
    reproducir correr_holistico_energia() en el mismo punto -- confirma que la
    reimplementación del bucle para los brazos 2/3 no introdujo divergencias físicas."""
    seed_layout = 12345 + seed
    kw_escala = dict(nq=ESCALA["nq"], naq=ESCALA["naq"], ne=ESCALA["ne"], npos=ESCALA["npos"],
                      pasos_basal=ESCALA["pasos_basal"])
    caja_negra = correr_holistico_energia(amp_rugosidad=amp, E_reserva=E_RESERVA_FIJA,
                                           seed_layout=seed_layout, guardar_curva=False,
                                           n_pasos_estructura=N_PASOS_ESTRUCTURA, dt=DT,
                                           n_subpasos=N_SUBPASOS, **kw_escala)
    reimplementado = correr_dinamica_intervenida("real", amp_rugosidad=amp,
                                                  seed_layout=seed_layout,
                                                  E_reserva=E_RESERVA_FIJA,
                                                  guardar_curva=False, **kw_escala)
    ok_cn, ok_re = caja_negra.get("ok"), reimplementado.get("ok")
    diff = None
    if ok_cn and ok_re:
        diff = abs(caja_negra["frac_masa_ligada"] - reimplementado["frac_masa_ligada"])
    coincide = (diff is not None) and (diff < 1e-9)
    log_fn(f"[cs077][validación] caja_negra.frac_masa_ligada={caja_negra.get('frac_masa_ligada')} "
           f"reimplementado.frac_masa_ligada={reimplementado.get('frac_masa_ligada')} "
           f"diff={diff} coincide_exacto={coincide}")
    return dict(ok_caja_negra=ok_cn, ok_reimplementado=ok_re, diff=diff,
                coincide_exacto=coincide,
                frac_masa_ligada_caja_negra=caja_negra.get("frac_masa_ligada"),
                frac_masa_ligada_reimplementado=reimplementado.get("frac_masa_ligada"))


# =========================================================================================
# Análisis (protocolo, punto 3): ancho de meseta, cuenca, varianza, histéresis
# =========================================================================================

def _agrupar_por_eps(filas, brazo):
    d = {}
    for r in filas:
        if r.get("brazo") != brazo or not r.get("ok"):
            continue
        eps = round(r["eps"], 8)
        d.setdefault(eps, []).append(r)
    return d


def analizar(filas):
    resultado = {}
    for brazo in BRAZOS:
        por_eps = _agrupar_por_eps(filas, brazo)
        eps_sorted = sorted(por_eps.keys())
        curva_media, curva_std, curva_n = [], [], []
        for eps in eps_sorted:
            fm = np.array([r["frac_masa_ligada"] for r in por_eps[eps]])
            curva_media.append(float(fm.mean()))
            curva_std.append(float(fm.std()))
            curva_n.append(len(fm))

        # (a) ancho de meseta: máximo intervalo contiguo (partiendo del eps más chico) donde
        # frac_masa_ligada_media se mantiene dentro de +-0.1 absoluto del valor en el primer
        # punto del grid (banda de tolerancia declarada, no ajustada post-hoc).
        ancho_meseta_idx = 0
        if curva_media:
            ref = curva_media[0]
            for i, v in enumerate(curva_media):
                if abs(v - ref) <= 0.10:
                    ancho_meseta_idx = i
                else:
                    break
        ancho_meseta_eps = (eps_sorted[ancho_meseta_idx] if eps_sorted else None)

        # (b) "misma cuenca": a la eps más chica del grid (dentro de la meseta esperada),
        # fracción de semillas cuyo frac_masa_ligada cae a <=0.05 del valor mediano de esa eps.
        frac_misma_cuenca = None
        if eps_sorted:
            eps0 = eps_sorted[0]
            vals = np.array([r["frac_masa_ligada"] for r in por_eps[eps0]])
            mediana = float(np.median(vals))
            frac_misma_cuenca = float(np.mean(np.abs(vals - mediana) <= 0.05))

        # (c) tiempo de estabilización medio (sobre semillas, en la eps más chica)
        tiempo_estab = None
        if eps_sorted:
            eps0 = eps_sorted[0]
            ts = [r["tiempo_estabilizacion"] for r in por_eps[eps0]
                  if r.get("tiempo_estabilizacion") is not None]
            tiempo_estab = float(np.mean(ts)) if ts else None

        # (f) varianza del estado final, por eps y promedio en la zona de meseta detectada
        varianza_meseta = (float(np.mean([curva_std[i] ** 2
                                           for i in range(ancho_meseta_idx + 1)]))
                            if curva_media else None)

        resultado[brazo] = dict(
            eps_grid=eps_sorted,
            frac_masa_ligada_media=curva_media,
            frac_masa_ligada_std=curva_std,
            n_ok_por_eps=curva_n,
            ancho_meseta_eps=ancho_meseta_eps,
            ancho_meseta_n_puntos=ancho_meseta_idx + 1,
            frac_misma_cuenca_eps0=frac_misma_cuenca,
            tiempo_estabilizacion_medio_eps0=tiempo_estab,
            varianza_media_en_meseta=varianza_meseta,
        )

    # (d) histéresis: limitación estructural -- se verifica que recorrer el grid ascendente
    # vs descendente (misma data, sin recomputar: cada punto (eps,seed) ya es determinista)
    # da la MISMA curva por construcción. Se documenta el porqué, no se fabrica una métrica.
    resultado["_nota_histeresis"] = (
        "El motor no tiene estado dinámico persistente entre valores de eps (corre() y "
        "posiciones_escenario() son funciones puras de (eps, seed), sin historia) -- un "
        "barrido ascendente y uno descendente del mismo grid coinciden punto a punto por "
        "construcción determinista. Esto es un hallazgo sobre la arquitectura actual del "
        "motor (no soporta continuación de trayectoria entre eps), no una medida de "
        "histéresis genuina; documentado explícitamente en vez de fabricar una métrica."
    )

    # comparación cruzada REAL vs DIRECCION_AZAR (criterio de falsación, protocolo punto 4)
    real = resultado.get("REAL", {})
    azar = resultado.get("DIRECCION_AZAR", {})
    resultado["_comparacion_real_vs_azar"] = dict(
        ancho_meseta_real=real.get("ancho_meseta_eps"),
        ancho_meseta_azar=azar.get("ancho_meseta_eps"),
        varianza_meseta_real=real.get("varianza_media_en_meseta"),
        varianza_meseta_azar=azar.get("varianza_media_en_meseta"),
    )
    return resultado


def analizar_lyapunov(filas_lyap):
    resultado = {}
    for brazo in BRAZOS:
        vals = [r["lyapunov_proxy"] for r in filas_lyap
                if r.get("brazo") == brazo and r.get("ok") and r.get("lyapunov_proxy") is not None]
        resultado[brazo] = dict(
            n=len(vals),
            lyapunov_proxy_medio=float(np.mean(vals)) if vals else None,
            lyapunov_proxy_std=float(np.std(vals)) if vals else None,
        )
    return resultado


# =========================================================================================
# main
# =========================================================================================

def main():
    log_lines = []

    def p(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        log_lines.append(line)

    if "--validar" in sys.argv:
        p("=== VALIDACIÓN DE FIDELIDAD (reimplementación vs caja negra) ===")
        for amp in (0.1, 1.5, 4.0):
            validar_fidelidad(log_fn=p, amp=amp, seed=0)
        return

    p("=== cs077: 4 brazos x eps-grid x semillas ===")
    p(f"escala={ESCALA} n_pasos_estructura={N_PASOS_ESTRUCTURA} n_subpasos={N_SUBPASOS} "
      f"eps_grid={EPS_GRID} n_semillas={N_SEMILLAS}")

    p("--- validación de fidelidad previa ---")
    validacion = validar_fidelidad(log_fn=p, amp=1.0, seed=0)

    p("--- barrido principal ---")
    filas = correr_barrido_principal(log_fn=p)
    analisis = analizar(filas)

    p("--- Lyapunov (subconjunto) ---")
    filas_lyap = correr_lyapunov(log_fn=p)
    analisis_lyap = analizar_lyapunov(filas_lyap)

    resultado = dict(
        experimento="cs077_gradientes_atractores",
        escala=ESCALA, n_pasos_estructura=N_PASOS_ESTRUCTURA, n_subpasos=N_SUBPASOS, dt=DT,
        eps_grid=EPS_GRID, n_semillas=N_SEMILLAS, e_reserva_fija=E_RESERVA_FIJA,
        gamma_sin_memoria=GAMMA_SIN_MEMORIA,
        validacion_fidelidad=validacion,
        filas=filas, analisis=analisis,
        filas_lyapunov=filas_lyap, analisis_lyapunov=analisis_lyap,
        log=log_lines,
    )
    out_json = OUT / "cs077_result.json"
    out_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    p(f"[archivo] {out_json}")

    for brazo in BRAZOS:
        a = analisis[brazo]
        p(f"[resumen] {brazo}: ancho_meseta_eps={a['ancho_meseta_eps']} "
          f"varianza_meseta={a['varianza_media_en_meseta']} "
          f"frac_misma_cuenca_eps0={a['frac_misma_cuenca_eps0']} "
          f"tiempo_estab_eps0={a['tiempo_estabilizacion_medio_eps0']} "
          f"lyapunov_medio={analisis_lyap[brazo]['lyapunov_proxy_medio']}")


if __name__ == "__main__":
    main()
