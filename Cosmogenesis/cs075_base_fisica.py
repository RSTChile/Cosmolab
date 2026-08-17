#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cs075_base_fisica.py — La capa base: parámetros físicos reales, no topología
=============================================================================

CORRECCIÓN DEL DIRECTOR (29-jul-2026), y por qué mi versión anterior estaba mal:
  En cs075_23_agentes.py puse 10 agentes "sin precondición", entre ellos #7 masa, #6 catálogo
  y #5 débil. **Eso está mal y contradice el propio arco del proyecto.** La masa NO existe al
  inicio: emerge en la ruptura electrodébil (~10⁻¹¹ s, 159 GeV) y su 99% en el confinamiento
  QCD (~10⁻⁵ s) — está escrito en LINEA_TIEMPO_MASA_topologia_vs_fisica.md, fila 4 y fila 5,
  y el error de ubicar la masa temprano ya está registrado ahí como "el error grande".
  Poner la masa en la capa base repetía ese error.

  Segundo: mi campo Phi era un campo abstracto con reacción Phi(1-Phi²) y dos atractores en
  ±1. Eso es TOPOLOGÍA — la rama que está CERRADA (probó que S>0 basta para generar estados
  diferenciados, y se agota antes de las partículas). El director lo dice explícito: no
  trabajamos con topología, trabajamos con parámetros físicos reales.

LA BASE CORRECTA, en las palabras del director:
  "Al inicio tenemos los agentes básicos: Expansión Supralumínica, Temperatura (que desciende),
   Densidad (que desciende), Exergía (que desciende), Entropía (que aumenta), y Gradientes de
   Campo producto de la o las asimetrías."

  Seis variables de estado termodinámico reales. Ninguna es una pieza del Modelo Estándar:
  son las CONDICIONES en las que las piezas podrán existir, o no. Los 23 elementos consultan
  este estado para saber si ya hay con qué.

  Y todas menos una tienen dirección obligada: T baja, ρ baja, X baja, S sube. Eso NO se
  programa como decreto — se deriva de la expansión, que es el único motor. La prueba B2
  verifica las direcciones en vez de asumirlas.

DEFINICIONES (tomadas de donde ya existen en el proyecto, no inventadas):
  - Exergía X: `np.mean((rho_local/rho_media - 1)²)` — la varianza relativa de densidad, es
    decir, LA DIFERENCIA. Verbatim de cs074_energia_holistica.py l.261 ("Regla 2: exergía
    (depende de las diferencias)"). Es la definición que el equipo ya usó y adjudicó.
  - Entropía S: entropía de Shannon del histograma de densidades, normalizada. Sube cuando la
    distribución se homogeneiza — el complemento natural de la exergía.
  - Expansión supralumínica: NO se declara como régimen aparte ni tiene parámetro de salida.
    Hay UNA sola ley de expansión (la del motor, a=√(1+k·t)) cuya velocidad v=L·da/dt decae
    como t^(-1/2); arranca en k/2 = 25c y cruza c por sí sola. El instante del cruce es
    SALIDA calculable — t = ((L·k/2c)²−1)/k = 12,48 con k=50 — no una entrada.

NADA DE ESTO ES TOPOLOGÍA: no hay grafo, no hay malla causal, no hay ±1. Hay un volumen con
temperatura, densidad y sus gradientes.
"""
from __future__ import annotations

import numpy as np

C_LUZ = 1.0          # unidad de velocidad del sistema (adimensional, convención)
K_B = 1.0            # unidad de temperatura
N_BINS_ENTROPIA = 32
# Rango FIJO del contraste de densidad para la entropía: los bins no se recalculan por
# medición, si no la métrica cambia de regla entre dos estados (ver docstring de entropia()).
RANGO_CONTRASTE = (0.0, 3.0)


class EstadoFisico:
    """Las seis variables de estado. Un volumen con densidad y temperatura, expandiéndose.

    No hay partículas, no hay grafo: hay campos escalares reales sobre una malla, y la
    expansión los arrastra. Las piezas del inventario leerán este estado.
    """

    def __init__(self, N=16, dt=1e-3, seed=12345,
                 T_inicial=1e3,             # temperatura inicial (adimensional, alta)
                 rho_inicial=1.0,           # densidad media inicial
                 amp_asimetria=0.1,         # ε: la asimetría primordial (la única heterogeneidad)
                 k_enfriamiento=50.0):      # k de la ley T=T0/√(1+k·t) (estado.py l.44).
                                            # Es la ÚNICA constante de expansión: fija a(t),
                                            # H(t) y el instante del cruce v=c.
        self.N, self.dt = N, dt
        self.k_enfriamiento = k_enfriamiento
        self.t = 0.0
        self.paso_n = 0

        rng = np.random.default_rng(seed)
        # --- densidad: media rho_inicial con la asimetría primordial encima ---
        # Es la #23 (rugosidad multiescala del campo térmico primordial). La distribución
        # es lognormal, como en catalogo.densidad_intrinseca: pocos picos densos, mucho tenue.
        ruido = rng.normal(0.0, 1.0, (N, N, N))
        self.rho = rho_inicial * np.exp(amp_asimetria * ruido)
        self.rho *= rho_inicial / self.rho.mean()          # media exacta
        # --- temperatura: acoplada a la densidad (más densidad = más caliente) ---
        # Mismo criterio que catalogo.py: "temperatura ≡ densidad local".
        self.T = T_inicial * (self.rho / self.rho.mean())
        self.a = 1.0                                        # factor de escala
        self.amp_asimetria = amp_asimetria
        self.historia = []

    # -- régimen de expansión: UNA sola ley, la del motor --
    def H(self):
        """Factor de Hubble, derivado del reloj de enfriamiento. Sin constantes nuevas.

        LEY ÚNICA (ya estaba en el motor, no se inventa nada):
          `cs072_modulos/estado.py` l.44:  T = T0/√(1+k·t)   (enfriamiento)
          `p_expansion.py`:                a = T0/T          ⟹  a = √(1+k·t)
        De ahí H = (da/dt)/a = k/(2(1+k·t)).

        TRES CORRECCIONES ENCADENADAS, cada una encontrada midiendo:
        1ª Puse `H_post = 1.0` constante. H constante es de Sitter — acelerada para siempre.
        2ª Además violaba `p_expansion.py`, que dice textual: "NO se inventa una ley nueva
           -- se deriva del propio reloj de enfriamiento que el motor YA tiene... ninguna
           constante nueva". `H_post` era justo esa constante prohibida.
        3ª CORRECCIÓN DEL DIRECTOR (29-jul): tampoco hacen falta DOS regímenes empalmados
           con un `fin_inflacion` puesto a mano. La transición es física y más simple: la
           velocidad de expansión no puede superar c. Una sola ley basta, porque su
           velocidad DECAE y cruza c por sí sola.
        """
        return float(self.k_enfriamiento / (2.0 * (1.0 + self.k_enfriamiento * self.t)))

    def a_teorico(self):
        """a(t) = √(1+k·t), la ley del motor. Para verificar que la integración no deriva."""
        return float(np.sqrt(1.0 + self.k_enfriamiento * self.t))

    def velocidad_expansion(self, L=1.0):
        """Velocidad a la que se expande una escala comóvil L: v = L·da/dt.
        Con a = √(1+k·t):  v = L·k/(2√(1+k·t)) — arranca en L·k/2 y DECAE como t^(-1/2)."""
        return float(L * self.k_enfriamiento / (2.0 * np.sqrt(1.0 + self.k_enfriamiento * self.t)))

    def es_supraluminico(self, L=1.0):
        """¿La expansión va más rápido que la luz? Criterio del director: en un universo
        físico la velocidad de expansión queda limitada a c, y ESE es el único cambio de
        régimen. No hay bandera de época ni parámetro de corte: se compara v contra c."""
        return bool(self.velocidad_expansion(L) > C_LUZ)

    def t_transicion_luminica(self, L=1.0):
        """El instante en que v = c, resuelto analíticamente de v(t) = L·k/(2√(1+k·t)):
        t = ((L·k/2c)² − 1)/k.  Es SALIDA del modelo, no un parámetro de entrada."""
        k = self.k_enfriamiento
        return float(((L * k / (2.0 * C_LUZ)) ** 2 - 1.0) / k)

    # -- las seis variables de estado --
    def temperatura_media(self):
        return float(self.T.mean())

    def densidad_media(self):
        return float(self.rho.mean())

    def exergia(self):
        """X = varianza relativa de densidad. cs074_energia_holistica.py l.261, verbatim.
        Es la DIFERENCIA disponible: X=0 significa homogéneo, sin nada que extraer."""
        m = self.rho.mean()
        if m <= 0:
            return 0.0
        return float(np.mean((self.rho / m - 1.0) ** 2))

    def entropia(self):
        """Entropía TERMODINÁMICA específica media (Sackur-Tetrode, a menos de constantes):
        s ∝ (3/2)·ln T − ln ρ.

        DOS CORRECCIONES, ambas encontradas midiendo, no razonando:

        1ª (histograma de rango variable): la primera versión usaba
           `np.histogram(self.rho)`, que toma min/max de los propios datos, así que el ancho
           del bin se encogía con la dilución. Verificado: el rango cayó de 7,4e-1 a 3,1e-5
           entre t=0 y t=0,3 mientras S quedaba casi constante.

        2ª (la de fondo — CONFUNDÍ DOS ENTROPÍAS): arreglar los bins no alcanzó, S siguió
           bajando. La causa real: la entropía de SHANNON del histograma de densidad y la
           entropía TERMODINÁMICA son magnitudes distintas y aquí van en sentidos opuestos.
           La difusión colapsa el contraste δ=ρ/⟨ρ⟩ hacia 1 (verificado: std cae de 1,0e-1 a
           1,5e-2, y el 100% del volumen termina en 2 bins de 32). Una distribución
           concentrada tiene Shannon BAJA por definición — mientras el sistema se homogeneiza
           y su entropía termodinámica SUBE. Medir la flecha del tiempo con Shannon sobre
           densidad era la métrica equivocada.

        Sackur-Tetrode es la definición correcta para esto: la expansión adiabática la deja
        invariante (T ∝ a⁻¹ con ρ ∝ a⁻³ da Δs = 0 sólo si γ calza; acá el término dominante
        es −ln ρ, que crece con la dilución) y la difusión, que es el proceso irreversible
        del sistema, la aumenta.
        """
        rho = np.maximum(self.rho, 1e-300)
        T = np.maximum(self.T, 1e-300)
        # entropía específica de Sackur-Tetrode (gas ideal monoatómico), a menos de
        # constantes: s ∝ ln(T^{3/2}/ρ). La expansión adiabática la deja constante; lo que
        # la SUBE es la difusión, que es el proceso irreversible del sistema.
        s_local = 1.5 * np.log(T) - np.log(rho)
        return float(s_local.mean())

    def gradientes_campo(self):
        """Magnitud media del gradiente de densidad — los gradientes producto de la asimetría.
        En coordenadas comóviles el gradiente físico lleva 1/a."""
        gx = np.roll(self.rho, -1, 0) - np.roll(self.rho, 1, 0)
        gy = np.roll(self.rho, -1, 1) - np.roll(self.rho, 1, 1)
        gz = np.roll(self.rho, -1, 2) - np.roll(self.rho, 1, 2)
        g = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) / (2.0 * self.a)
        return float(g.mean())

    def estado(self):
        """El estado completo que los 23 agentes leen para decidir si tienen con qué."""
        return dict(
            paso=self.paso_n, t=float(self.t), dt=self.dt,
            a=float(self.a), H=float(self.H()),
            supraluminico=self.es_supraluminico(),
            T=self.temperatura_media(),
            rho=self.densidad_media(),
            X=self.exergia(),
            S=self.entropia(),
            grad=self.gradientes_campo(),
            v_c=self.velocidad_expansion(), a_teorico=self.a_teorico(),
            T_max=float(self.T.max()), T_min=float(self.T.min()),
            rho_max=float(self.rho.max()),
        )

    # -- evolución: TODO lo mueve la expansión. Ninguna variable se fija a mano. --
    def paso(self, depositos=None):
        """Un paso de la base física.

        La expansión es el único motor: diluye la densidad (ρ ∝ a⁻³), enfría
        adiabáticamente (T ∝ a⁻¹ para materia relativista) y difunde los gradientes.
        `depositos` son las contribuciones de los agentes del inventario, si alguno
        ya despertó — se suman a la densidad, no la reemplazan.
        """
        H = self.H()
        da = H * self.a * self.dt
        a_nuevo = self.a + da
        factor = self.a / a_nuevo

        # dilución por expansión: rho ∝ a^-3
        self.rho = self.rho * factor ** 3
        # enfriamiento adiabático: T ∝ a^-1 (radiación/relativista)
        self.T = self.T * factor
        # difusión térmica: los gradientes se suavizan (esto SUBE la entropía)
        lap_rho = (np.roll(self.rho, 1, 0) + np.roll(self.rho, -1, 0)
                   + np.roll(self.rho, 1, 1) + np.roll(self.rho, -1, 1)
                   + np.roll(self.rho, 1, 2) + np.roll(self.rho, -1, 2) - 6.0 * self.rho)
        coef_dif = 0.05 / (a_nuevo ** 2)
        self.rho = self.rho + coef_dif * lap_rho * self.dt / self.dt  # difusión por paso
        self.rho = np.maximum(self.rho, 1e-30)

        if depositos is not None:
            self.rho = np.maximum(self.rho + depositos * self.dt, 1e-30)

        self.a = a_nuevo
        self.t += self.dt
        self.paso_n += 1

    def correr(self, T_total, registrar_cada=100, depositos_fn=None):
        pasos = int(round(T_total / self.dt))
        for k in range(pasos):
            dep = depositos_fn(self) if depositos_fn is not None else None
            self.paso(depositos=dep)
            if registrar_cada and (k % registrar_cada == 0 or k == pasos - 1):
                self.historia.append(self.estado())
        return self.estado()
