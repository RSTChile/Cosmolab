#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_arquitectura_agentes.py — Arquitectura de agentes sobre campo común (estigmergia)
========================================================================================

Quién soy / qué hago:
  Implementa la ARQUITECTURA pedida en el protocolo CS075: UN AGENTE POR CADA ASPECTO del
  experimento, cada uno autónomo, que LEE el campo común y DEPOSITA su contribución en él.
  Ningún agente sabe que los otros existen. No hay bucle de fuerzas por turnos: el proceso
  común congela el estado del campo, pide a todos los agentes su depósito sobre ESE MISMO
  estado, y aplica la suma en un único paso.

  Este archivo NO contesta ninguna pregunta física. Es la máquina, y viene con su propia
  batería de pruebas de arquitectura (cs075_pruebas_arquitectura.py). Primero se prueba que
  la máquina funciona; después se le pregunta algo al universo.

Reglas de diseño que este archivo respeta (protocolo CS075 §2):
  A. CERO TURNOS — todos los agentes leen el MISMO Phi congelado; el orden en que se los
     consulta no puede cambiar el resultado (más allá del reordenamiento de sumas en punto
     flotante, ~1e-16). La prueba P2 lo verifica.
  B. EL CAMPO ES LA ÚNICA FUENTE DE VERDAD — ningún agente recibe la lista de los otros ni
     calcula distancias N×N. Cada agente declara su `radio` de vecindad y sólo puede leer
     dentro de él. La prueba P3 lo verifica perturbando celdas lejanas.
  C. FORZAMIENTO DE BORDE — la expansión y el enfriamiento no son mensajes: entran como
     `contexto` (el reloj de fondo: t, factor de escala a, tasa H, temperatura T), que el
     proceso común calcula y todos leen.
  D. INERCIA POR MEMORIA — el agente de plasticidad tiene memoria propia (W_local) que se
     construye con la historia del campo. No hay masa como atributo.

Separación pura/impura (necesaria para que la prueba de orden sea limpia):
  - `contribucion(Phi, contexto)` es PURA: no muta nada, sólo lee y devuelve su depósito.
  - `consolidar(Phi, contexto)` es donde el agente actualiza su memoria interna, y el
     proceso común la llama UNA vez por paso, después de aplicar todos los depósitos.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constantes. Las del campo plástico vienen de la arquitectura ANIMA real
# (VSTCosmo/Célula_Madre/campo/VST_Celula_Madre_001.py, l.463-469) — no son inventadas.
# ---------------------------------------------------------------------------
ETA_HEBB = 0.02      # CAMPO_ETA_HEBB   (l.463) tasa de aprendizaje hebbiano
TAU_W = 0.005        # CAMPO_TAU_W      (l.464) decaimiento de pesos
GAMMA_PLAST = 0.15   # CAMPO_GAMMA_PLAST(l.465) fuerza del término plástico
W_MAX = 1.0          # CAMPO_W_MAX      (l.466) límite de estabilidad
PHIVEL_CLIP = 5.0    # CAMPO_PHIVEL_CLIP(l.477)
DT_DEFAULT = 0.01    # DT               (l.108)


# ===========================================================================
# Interfaz de agente
# ===========================================================================
class AgenteCampo:
    """Un aspecto del experimento. Autónomo: lee el campo común, deposita en él.

    `radio` declara cuántas celdas de vecindad necesita leer. radio=0 es puramente local
    (sólo la propia celda); radio=1 lee los 6 vecinos inmediatos. La prueba P3 verifica que
    ningún agente lea más allá de lo que declara.
    """
    nombre = "agente"
    radio = 0

    def contribucion(self, Phi, contexto):
        """PURA. Devuelve el depósito de este agente sobre el campo congelado Phi."""
        raise NotImplementedError

    def consolidar(self, Phi, contexto):
        """Actualiza la memoria interna del agente. Sin memoria interna, no hace nada."""
        return None

    def estado_memoria(self):
        """Copia de la memoria interna (para snapshot/restore en las pruebas)."""
        return None

    def cargar_memoria(self, estado):
        return None


# ---------------------------------------------------------------------------
# Aspecto 1 — DIFUSIÓN (el laplaciano; la única lectura de vecindad)
# ---------------------------------------------------------------------------
class AgenteDifusion(AgenteCampo):
    """Laplaciano de 7 puntos en malla 3D periódica, en coordenadas comóviles.

    En coordenadas comóviles el laplaciano físico lleva 1/a²: al expandirse el espacio, la
    misma diferencia entre celdas vecinas corresponde a un gradiente físico menor. `a` llega
    por `contexto` (el reloj de fondo), NO por otro agente.
    """
    nombre = "difusion"
    radio = 1

    def __init__(self, coef=1.0):
        self.coef = coef

    def contribucion(self, Phi, contexto):
        lap = (np.roll(Phi, 1, 0) + np.roll(Phi, -1, 0)
               + np.roll(Phi, 1, 1) + np.roll(Phi, -1, 1)
               + np.roll(Phi, 1, 2) + np.roll(Phi, -1, 2) - 6.0 * Phi)
        return self.coef * lap / (contexto["a"] ** 2)


# ---------------------------------------------------------------------------
# Aspecto 2 — REACCIÓN no lineal (la que crea los dos atractores del campo)
# ---------------------------------------------------------------------------
class AgenteReaccion(AgenteCampo):
    """Phi(1 - Phi²): misma forma que la reacción de ANIMA (l.581). Puramente local."""
    nombre = "reaccion"
    radio = 0

    def __init__(self, coef=1.0):
        self.coef = coef

    def contribucion(self, Phi, contexto):
        return self.coef * Phi * (1.0 - Phi * Phi)


# ---------------------------------------------------------------------------
# Aspecto 3 — EXPANSIÓN (dilución por el estiramiento del espacio)
# ---------------------------------------------------------------------------
class AgenteExpansion(AgenteCampo):
    """Dilución -3·H·Phi. H llega por contexto (forzamiento de fondo, protocolo §2.C)."""
    nombre = "expansion"
    radio = 0

    def contribucion(self, Phi, contexto):
        return -3.0 * contexto["H"] * Phi


# ---------------------------------------------------------------------------
# Aspecto 4 — ENFRIAMIENTO (relajación hacia un piso térmico que baja con el reloj)
# ---------------------------------------------------------------------------
class AgenteEnfriamiento(AgenteCampo):
    """Arrastra el campo hacia el piso térmico T(t), que el reloj de fondo va bajando.
    Puramente local: cada celda sólo mira su propio valor."""
    nombre = "enfriamiento"
    radio = 0

    def __init__(self, tasa=0.3):
        self.tasa = tasa

    def contribucion(self, Phi, contexto):
        return -self.tasa * contexto["T"] * Phi


# ---------------------------------------------------------------------------
# Aspecto 5 — PLASTICIDAD (la memoria del campo; el aspecto CON estado propio)
# ---------------------------------------------------------------------------
class AgentePlasticidad(AgenteCampo):
    """Versión local y estigmérgica de los pesos plásticos W de ANIMA.

    ANIMA usa una matriz W de 32×32 sobre 32 nodos (l.630-633: W @ Phi). Esa matriz es
    global: cada nodo se acopla a todos. Acá NO se puede — el protocolo §2.B prohíbe el
    N×N. La reducción honesta: cada celda tiene UN peso escalar `W_local` que acopla su
    propio valor con la MEDIA DE SU VECINDAD inmediata. Es el mismo mecanismo hebbiano
    (correlación entre lo que la celda es y lo que su entorno es), reducido a lo local.

    Esto es una REDUCCIÓN de la arquitectura de ANIMA, no un port literal, y queda dicho.
    """
    nombre = "plasticidad"
    radio = 1

    def __init__(self, eta=ETA_HEBB, tau=TAU_W, gamma=GAMMA_PLAST, w_max=W_MAX, forma=None):
        self.eta, self.tau, self.gamma, self.w_max = eta, tau, gamma, w_max
        self.W_local = np.zeros(forma) if forma is not None else None

    def _media_vecindad(self, Phi):
        return (np.roll(Phi, 1, 0) + np.roll(Phi, -1, 0)
                + np.roll(Phi, 1, 1) + np.roll(Phi, -1, 1)
                + np.roll(Phi, 1, 2) + np.roll(Phi, -1, 2)) / 6.0

    def contribucion(self, Phi, contexto):
        if self.W_local is None:
            self.W_local = np.zeros(Phi.shape)
        vec = self._media_vecindad(Phi)
        return self.gamma * (self.W_local * vec - Phi)

    def consolidar(self, Phi, contexto):
        if self.W_local is None:
            self.W_local = np.zeros(Phi.shape)
        vec = self._media_vecindad(Phi)
        dt = contexto["dt"]
        self.W_local = np.clip(
            self.W_local + (self.eta * Phi * vec - self.tau * self.W_local) * dt,
            -self.w_max, self.w_max)

    def estado_memoria(self):
        return None if self.W_local is None else self.W_local.copy()

    def cargar_memoria(self, estado):
        self.W_local = None if estado is None else estado.copy()


# ===========================================================================
# El proceso común
# ===========================================================================
class ProcesoComun:
    """El campo común y el reloj de fondo. Congela el estado, pide a cada agente su
    depósito sobre ESE estado, suma, y aplica un único paso. Nadie actualiza el campo
    por su cuenta; nadie ve a los demás.
    """

    def __init__(self, agentes, N=16, dt=DT_DEFAULT, amplitud_inicial=0.1,
                 tasa_expansion=0.01, tasa_enfriamiento_reloj=0.05, seed=12345):
        self.agentes = list(agentes)
        self.N, self.dt = N, dt
        self.tasa_expansion = tasa_expansion
        self.tasa_enfriamiento_reloj = tasa_enfriamiento_reloj
        rng = np.random.default_rng(seed)
        # plasma primordial: perturbaciones de media cero (protocolo §3, inicialización)
        self.Phi = rng.normal(0.0, amplitud_inicial, (N, N, N))
        self.Phi -= self.Phi.mean()
        self.Phi_vel = np.zeros_like(self.Phi)
        self.t = 0.0
        self.historia = []

    def contexto(self):
        """El reloj de fondo. Protocolo §2.C: el entorno no manda mensajes, cambia las
        condiciones que todos leen."""
        a = np.exp(self.tasa_expansion * self.t)      # factor de escala
        H = self.tasa_expansion                        # a'/a
        T = np.exp(-self.tasa_enfriamiento_reloj * self.t)  # piso térmico que baja
        return dict(t=self.t, a=float(a), H=float(H), T=float(T), dt=self.dt)

    def paso(self, orden=None):
        """Un paso. `orden` permite consultar a los agentes en otro orden — el resultado
        debe ser el mismo (prueba P2). Por defecto, el orden de la lista."""
        ctx = self.contexto()
        Phi_congelado = self.Phi          # nadie lo muta: los agentes son puros
        idxs = range(len(self.agentes)) if orden is None else orden

        depositos = [self.agentes[i].contribucion(Phi_congelado, ctx) for i in idxs]
        total = np.zeros_like(self.Phi)
        for d in depositos:
            total = total + d

        self.Phi_vel = np.clip(self.Phi_vel + total * self.dt, -PHIVEL_CLIP, PHIVEL_CLIP)
        Phi_nuevo = np.clip(self.Phi + self.Phi_vel * self.dt, -1.0, 1.0)

        for i in idxs:
            self.agentes[i].consolidar(Phi_congelado, ctx)

        self.Phi = Phi_nuevo
        self.t += self.dt
        return ctx

    def correr(self, T_total, registrar_cada=50, orden=None):
        pasos = int(round(T_total / self.dt))
        for k in range(pasos):
            self.paso(orden=orden)
            if registrar_cada and (k % registrar_cada == 0 or k == pasos - 1):
                self.historia.append(self.observables())
        return self.observables()

    # -- observables (crudos, sin adjudicar) --
    def observables(self):
        Phi = self.Phi
        activo = np.abs(Phi) > 0.5
        return dict(
            t=float(self.t),
            a=float(np.exp(self.tasa_expansion * self.t)),
            frac_activa=float(activo.mean()),
            energia_campo=float(np.sum(0.5 * self.Phi_vel ** 2 + 0.25 * (1 - Phi ** 2) ** 2)),
            phi_abs_medio=float(np.abs(Phi).mean()),
            phi_max=float(np.abs(Phi).max()),
            n_grumos=int(self.contar_grumos()),
            hay_nan=bool(not np.all(np.isfinite(Phi))),
        )

    def contar_grumos(self, umbral=0.5):
        """Componentes conexas donde |Phi|>umbral. Vecindad inmediata, sin distancias N×N."""
        from scipy import ndimage
        _, n = ndimage.label(np.abs(self.Phi) > umbral)
        return n


def construir(N=16, dt=DT_DEFAULT, amplitud_inicial=0.1, tasa_expansion=0.01,
              tasa_enfriamiento_reloj=0.05, seed=12345, aspectos=None):
    """Arma el proceso con un agente por aspecto. `aspectos` permite dejar alguno afuera
    (para la prueba P1, que verifica que cada agente realmente aporta algo)."""
    todos = {
        "difusion": lambda: AgenteDifusion(),
        "reaccion": lambda: AgenteReaccion(),
        "expansion": lambda: AgenteExpansion(),
        "enfriamiento": lambda: AgenteEnfriamiento(),
        "plasticidad": lambda: AgentePlasticidad(forma=(N, N, N)),
    }
    nombres = list(todos) if aspectos is None else list(aspectos)
    agentes = [todos[n]() for n in nombres]
    return ProcesoComun(agentes, N=N, dt=dt, amplitud_inicial=amplitud_inicial,
                        tasa_expansion=tasa_expansion,
                        tasa_enfriamiento_reloj=tasa_enfriamiento_reloj, seed=seed)
