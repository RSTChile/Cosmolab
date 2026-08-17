#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Genoma — EL GENOMA DE LA CÉLULA MADRE COSMOSEMIÓTICA   ·   "QUIÉN SOY"
================================================================================

QUÉ ES ESTE ARCHIVO
-------------------
Esto NO es todavía el organismo corriendo. Es su GENOMA y el MOTOR que lo lee:
la configuración inicial abierta de la que un organismo se desarrolla, más la
maquinaria mínima (citoplasma, contrato de organelo, ciclo metabólico, leyes de
escala) que le da vida. Un organismo concreto se "transcribe" desde este genoma
expresando un subconjunto de organelos.

Está escrito para poder ENTENDERSE leyendo solo sus docstrings y comentarios,
sin reprocesar el código entero. Es deliberadamente autodescriptivo: el archivo
es, también, la memoria del organismo sobre sí mismo.

POR QUÉ EXISTE (el problema que resuelve)
-----------------------------------------
El linaje VSTCosmo (v70–v182, ~292 scripts) acumuló muchas capacidades, pero el
organismo consolidado actual (VST_Celula_Madre_001.py) las cablea de forma
MONOLÍTICA: un método gigante con `if flag` por todas partes, y términos de campo
inline. Eso impide (a) tener TODAS las capacidades en potencia, (b) expresarlas /
silenciarlas / recombinarlas limpio, y (c) que crezcan y se recombinen solas.
El genoma reorganiza todo eso bajo SEIS principios (abajo) para que el organismo
sea una verdadera CÉLULA MADRE: pluripotente, modular y evolucionable.

================================================================================
LOS SEIS PRINCIPIOS QUE ENCARNO  (bio-cosmosemióticos)
================================================================================
1. PLURIPOTENCIA. Debo contener EN POTENCIA todas las capacidades exploradas por
   el linaje, conmutables (latentes hasta que el contexto las exprese), MENOS las
   podadas con razón. Incluye la capacidad de DÍADA (comunicación entre células):
   no es "fuera de alcance", es potencial dormido.

2. MODULARIDAD-ORGANELO. Cada capacidad es un ORGANELO encapsulado (como una
   mitocondria o un nucléolo): membrana (interfaz), estado interno propio, función
   especializada, expresable/silenciable. Los organelos NO se meten en los internos
   de otros: se comunican SOLO por el citoplasma compartido (el `Milieu`).

3. ECONOMÍA. La naturaleza es económica: no se gasta lo que no se usa. Un organelo
   silenciado cuesta cero (estructural); uno expresado pero ocioso se auto-regula a
   la baja. La economía es PRESIÓN INTRÍNSECA, nunca un regulador externo (eso sería
   el "Pastor" podado: control desde fuera = Shannon encubierto).

4. ESCALAMIENTO ALOMÉTRICO (ley de Kleiber). A más complejidad, MENOR metabolismo
   por unidad y MAYORES tiempos vitales. Ratón: vive rápido, metabolismo alto.
   Elefante: vive lento y largo, metabolismo bajo. La maduración del organismo es
   su descenso por esta curva. Las τ (tiempos) y los costos NO son constantes: se
   escalan desde la complejidad M (ver `MedidorComplejidad`).

5. LOCI RESERVADOS. El genoma reserva SLOTS VACÍOS para capacidades anticipadas
   pero aún no desarrolladas. El primero: la GENÉTICA DEL ALTRUISMO de Scott Boorman
   (Boorman & Levitt, 1980), para el salto al primer PLURICELULAR. Debe estar
   reservado AHORA aunque vacío: si lo agregáramos recién al hacer el pluricelular,
   estaríamos IMPONIENDO la multicelularidad (Shannon a mayor escala). Reservado, el
   paso a pluricelular es VOLUNTARIO/emergente. (Una célula madre real porta genes
   que solo se expresan en sus descendientes diferenciados: esto es eso.)

6. ADAPTACIÓN Y EXAPTACIÓN. Cada organelo puede ADAPTARSE (plasticidad de sus
   parámetros) y EXAPTAR (su función puede ser co-optada para otra cosa). Las
   definiciones que determinan la persistencia de su diferencia (su identidad, su
   membrana, su función) NO están cerradas: el organelo está sujeto a las leyes
   evolutivas generales. Por eso el `Milieu` es público: una señal secretada para
   un fin queda disponible para ser co-optada por otro: el citoplasma compartido es
   la CONDICIÓN DE POSIBILIDAD de la exaptación. Corolario operativo: registramos
   la PROCEDENCIA de cada organelo, pero NUNCA congelamos su telos. El genoma
   describe DE DÓNDE PARTE el organismo, no a dónde llega.

================================================================================
DOS INVARIANTES QUE NO SE CONTRADICEN (aclaración importante)
================================================================================
· INVARIANTE DE INGENIERÍA: al modularizar (refactor), NOSOTROS no rompemos el
  baseline. El organismo con las capacidades actuales debe reproducir el control
  congelado v1 (VST_Organismo_Individual.py). Esto se valida con tests, no aquí.
· APERTURA BIOLÓGICA: una vez puesto el cimiento, el organismo PUEDE evolucionarse
  a sí mismo en runtime (adaptación/exaptación).
Uno dice "no rompas el cimiento mientras construyes"; el otro, "el cimiento, ya
puesto, puede cambiarse solo". Compatibles.

================================================================================
ESTADO DE ESTE ARCHIVO (honestidad sobre lo hecho vs. lo pendiente)
================================================================================
HECHO aquí:  motor del genoma (Milieu, Organelo, MedidorComplejidad/Kleiber,
             LocusReservado, Organismo/ciclo metabólico) + el MANIFIESTO del
             genoma (censo completo de organelos con su estado) + 3 organelos de
             ejemplo funcionando + el locus reservado de Boorman + autodescripción.
PENDIENTE:   portar CADA capacidad del organismo v2 a su organelo, validando
             contra el control v1 (invariante de ingeniería). Cada entry del
             manifiesto marcada PRESENTE/PARCIAL es código que aún vive en
             VST_Celula_Madre_001.py y hay que envolver como organelo.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import math


# ==============================================================================
# ESTADO DE UN GEN / ORGANELO EN EL GENOMA
# ==============================================================================
class Estado(Enum):
    """Cuán desarrollado está un organelo en este genoma.

    No es lo mismo "no expresado ahora" que "no existe": un organelo PRESENTE
    puede estar silenciado (expresar=False) y seguir siendo parte del genoma.
    RESERVADO es el cuarto estado clave (principio 5): el slot existe, declarado,
    pero todavía no tiene mecanismo — espera desarrollo futuro.
    """
    PRESENTE  = "presente"    # implementado y funcional (portado o nativo)
    PARCIAL   = "parcial"     # existe pero con limitaciones reconocidas
    AUSENTE   = "ausente"     # capacidad explorada en el linaje, aún no portada
    RESERVADO = "reservado"   # locus declarado pero VACÍO (a desarrollar)


# ==============================================================================
# MILIEU — EL CITOPLASMA COMPARTIDO  (sustrato de la comunicación y la exaptación)
# ==============================================================================
class Milieu:
    """El medio interno: un espacio de señales PÚBLICAS que todos los organelos
    comparten. Es a la vez el "citoplasma" (donde difunden las señales) y la razón
    por la que la exaptación es posible (principio 6): como toda señal secretada es
    pública, puede ser leída — co-optada — por un organelo distinto del que la
    secretó, incluso por uno que no existía cuando la señal se "inventó".

    Cómo funciona: es un diccionario nombre->valor con dos matices.
      · `secretar` ESCRIBE una señal (un organelo deposita su salida).
      · `leer` LEE una señal (con valor por defecto si aún nadie la secretó).
    Deliberadamente simple: la riqueza está en QUÉ señales circulan, no en el
    contenedor. El historial reciente se guarda para organelos que necesiten
    derivadas/tendencias (p.ej. un futuro organelo de Atención a dΩ/dt).
    """

    def __init__(self) -> None:
        # Señales vivas del paso actual. Clave = nombre de señal (p.ej. 'error',
        # 'Cb', 'orientacion', 'gradiente'); valor = su magnitud actual.
        self.senales: dict[str, Any] = {}
        # Memoria corta por señal (para tendencias/derivadas). Acotada por organelo
        # que la consuma; aquí solo se acumula de forma barata.
        self.historial: dict[str, list] = {}
        # Contabilidad metabólica del paso (principio 3): cuánta energía costó el
        # ciclo. Cada organelo suma aquí su gasto.
        self.gasto_metabolico: float = 0.0

    def secretar(self, nombre: str, valor: Any, guardar_historial: bool = False) -> None:
        """Un organelo deposita una señal en el citoplasma. Si `guardar_historial`,
        además la acumula en la memoria corta (para quien necesite su tendencia)."""
        self.senales[nombre] = valor
        if guardar_historial:
            self.historial.setdefault(nombre, []).append(valor)

    def leer(self, nombre: str, defecto: Any = 0.0) -> Any:
        """Lee una señal del citoplasma. Devuelve `defecto` si aún nadie la secretó
        (un organelo no debe asumir que otro ya corrió: lee defensivamente)."""
        return self.senales.get(nombre, defecto)

    def gastar(self, energia: float) -> None:
        """Registra gasto metabólico en el citoplasma (principio 3, economía)."""
        self.gasto_metabolico += float(energia)

    def nuevo_paso(self) -> None:
        """Reinicia la contabilidad de gasto al empezar un ciclo metabólico.
        Las señales NO se borran (persisten como estado del medio); solo se pone a
        cero el contador de energía del paso."""
        self.gasto_metabolico = 0.0


# ==============================================================================
# ORGANELO — EL CONTRATO COMÚN (la "membrana" y el metabolismo de cada capacidad)
# ==============================================================================
class Organelo:
    """Clase base de todo organelo. Define el CONTRATO uniforme que permite al
    organismo tratar cualquier capacidad genéricamente (y por tanto expresarla,
    silenciarla o recombinarla sin conocer sus internos).

    LA MEMBRANA (principio 2): un organelo se relaciona con el resto SOLO por tres
    actos, en este orden, una vez por ciclo:
        1. percibir(milieu)  -> lee del citoplasma lo que necesita (su "membrana
                                de entrada"). No muta el milieu.
        2. metabolizar(dt, tempo) -> actualiza su estado interno. `tempo` es el
                                factor de Kleiber s(M) que ESTIRA sus tiempos
                                propios al crecer la complejidad (principio 4).
        3. secretar(milieu)  -> deposita sus señales de salida en el citoplasma
                                (su "membrana de salida") y registra su gasto.

    PLASTICIDAD (principio 6): los parámetros del organelo viven en `self.plast`
    (un dict mutable), NO como constantes congeladas. Adaptación = cambiar `plast`;
    exaptación = que su salida sea co-optada por otro organelo vía el milieu.

    IDENTIDAD ABIERTA: `procedencia` registra de dónde vino (trazabilidad), pero
    NO se congela "para qué sirve". `criterio` es un descriptor del éxito actual,
    no una compuerta que fije su telos.

    ECONOMÍA (principio 3): `costo_base` es lo que cuesta trabajar (no existir). El
    organismo lo escala por r(M) (eficiencia de Kleiber) y lo registra en el milieu.
    Si está silenciado, NO entra al ciclo: cuesta cero por construcción.
    """

    def __init__(
        self,
        nombre: str,
        organelo_analogo: str,
        procedencia: str,
        descripcion: str,
        nodo_canonico: str = "",                  # nodo de la Teoría Cosmosemiótica que realiza
        lee: Optional[list[str]] = None,
        secreta: Optional[list[str]] = None,
        depende_de: Optional[list[str]] = None,
        costo_base: float = 0.0,
        plast: Optional[dict[str, Any]] = None,
        criterio: str = "",
        estado: Estado = Estado.PRESENTE,
        expresar: bool = True,
    ) -> None:
        self.nombre = nombre                      # identificador único en el genoma
        self.organelo_analogo = organelo_analogo  # la analogía biológica (didáctica)
        self.procedencia = procedencia            # de qué script/versión proviene
        self.nodo_canonico = nodo_canonico        # trazabilidad TEORÍA→código (no solo versión)
        self.descripcion = descripcion            # "qué hago y cómo" en prosa
        # Membrana declarada: qué señales lee y cuáles secreta. Es DOCUMENTACIÓN +
        # permite al organismo ordenar el ciclo y detectar co-opciones (exaptación).
        self.lee = list(lee or [])
        self.secreta = list(secreta or [])
        # Dependencias de orden: este organelo corre DESPUÉS de estos otros (porque
        # lee señales que ellos secretan). Ej.: Ritual depende de Cb.
        self.depende_de = list(depende_de or [])
        self.costo_base = float(costo_base)       # gasto al trabajar (economía)
        self.plast = dict(plast or {})            # parámetros MUTABLES (plasticidad)
        self.criterio = criterio                  # descriptor de éxito actual
        self.estado = estado                      # PRESENTE/PARCIAL/AUSENTE/RESERVADO
        self.expresar = expresar                  # ¿está expresado (no silenciado)?

    # ---- Los tres actos de la membrana. La base son no-ops documentados; cada
    # ---- organelo concreto sobreescribe lo que necesite. ----
    def percibir(self, milieu: "Milieu") -> None:
        """Lee del citoplasma lo necesario y lo guarda en estado temporal interno.
        La base no lee nada (un organelo sin entradas, p.ej. un marcapasos puro)."""
        pass

    def metabolizar(self, dt: float, tempo: float) -> None:
        """Avanza el estado interno un paso `dt`. `tempo` (=s(M), Kleiber) estira
        los tiempos propios: a mayor complejidad, el organelo "vive más lento".
        Convención de uso: donde el organelo tenga una τ, usar `tau * tempo`."""
        pass

    def secretar(self, milieu: "Milieu") -> None:
        """Deposita las señales de salida en el citoplasma. La base no secreta."""
        pass

    def descripcion_de_si(self) -> str:
        """Autodescripción legible de este organelo (para el censo 'quién soy')."""
        membrana = f"lee[{', '.join(self.lee) or '—'}] → secreta[{', '.join(self.secreta) or '—'}]"
        nodo = f"    nodo canónico: {self.nodo_canonico}\n" if self.nodo_canonico else ""
        return (f"· {self.nombre} ({self.organelo_analogo}) [{self.estado.value}]"
                f"{'' if self.expresar else ' (silenciado)'}\n"
                f"    {self.descripcion}\n"
                f"{nodo}"
                f"    membrana: {membrana}\n"
                f"    procedencia: {self.procedencia} · costo_base={self.costo_base}")


# ==============================================================================
# LOCUS RESERVADO — UN SLOT GENÉTICO VACÍO (principio 5)
# ==============================================================================
class LocusReservado(Organelo):
    """Un organelo DECLARADO PERO SIN MECANISMO: el genoma reserva el lugar y la
    membrana tentativa de una capacidad anticipada, sin implementarla todavía.

    Por qué un locus vacío y no "lo agregamos cuando haga falta": si la capacidad
    apareciera recién en el momento de necesitarla, sería estructura IMPUESTA por
    nosotros desde fuera (Shannon encubierto). Reservada de antemano, su desarrollo
    futuro es continuación del linaje — emergencia, no intervención.

    Un locus reservado nunca se expresa (`expresar=False`), no metaboliza ni
    secreta: solo ocupa su lugar en el genoma y documenta su propósito. Cuando se
    desarrolle, se sustituye por un Organelo real con esta misma membrana.
    """

    def __init__(self, nombre: str, organelo_analogo: str, proposito: str,
                 membrana_tentativa: str, nota: str, procedencia: str = "(a desarrollar)",
                 lee: Optional[list[str]] = None, secreta: Optional[list[str]] = None) -> None:
        super().__init__(
            nombre=nombre, organelo_analogo=organelo_analogo, procedencia=procedencia,
            descripcion=f"[LOCUS RESERVADO — VACÍO] {proposito}",
            lee=lee, secreta=secreta, costo_base=0.0,
            criterio="(sin criterio: a especificar al desarrollarlo)",
            estado=Estado.RESERVADO, expresar=False,
        )
        self.proposito = proposito                  # qué hará cuando se desarrolle
        self.membrana_tentativa = membrana_tentativa  # sus conexiones previstas
        self.nota = nota                            # guarda anti-imposición

    # Un locus vacío es inerte por definición: percibir/metabolizar/secretar = no-op
    # (heredados de la base). No se le da mecanismo: ese es justamente el punto.


# ==============================================================================
# MEDIDOR DE COMPLEJIDAD — LAS LEYES DE KLEIBER  (principio 4)
# ==============================================================================
class MedidorComplejidad:
    """Calcula la complejidad M del organismo y deriva de ella los DOS moduladores
    globales que implementan la ley de Kleiber. Es, en sí mismo, el "estado
    fisiológico de conjunto" del organismo (y el lugar natural donde más adelante
    revivirá la `CbGlobal` perdida: una presión GLOBAL, no por motor).

    NOTA DE PROCEDENCIA TEÓRICA: la ley de Kleiber (escalamiento alométrico) es una
    EXTENSIÓN nuestra, NO un nodo nombrado de la Teoría Cosmosemiótica canónica. Es
    compatible con la economía y con el PRE (Principio de Reserva Estructural,
    O-N8.19), pero se declara honestamente como añadido, no como derivación del canon.

    Definiciones (todo EMERGE de M; nada se tunea a mano — anti-Shannon):
      · M  = "masa"/complejidad = nº de organelos expresados, ponderado por su
             costo_base (un organelo más pesado aporta más masa). M0 = baseline
             de la célula madre mínima (M con un solo organelo de costo unitario).
      · B  = tasa metabólica TOTAL ∝ M^(3/4)   (crece sublinealmente: economías de
             escala — el todo gasta más que una célula, pero menos que la suma).
      · b  = B/M ∝ M^(-1/4)   metabolismo POR UNIDAD: BAJA al crecer (eficiencia).
      · s  = factor de TEMPO = (M/M0)^(1/4): MULTIPLICA todas las τ → a más
             complejidad, tiempos más largos (el organismo "vive más lento").
      · r  = factor de EFICIENCIA = (M/M0)^(-1/4): escala el costo por organelo.

    Lectura: la célula madre mínima es el ratón (M pequeño → s≈1 rápido, r≈1 caro
    por unidad). Al diferenciarse y volverse pluricelular es el elefante (M grande
    → s grande/lento, r pequeño/eficiente).
    """

    def __init__(self, M0: float = 1.0) -> None:
        self.M0 = float(M0)   # complejidad de referencia (célula madre mínima)
        self.M = float(M0)    # complejidad actual (se recalcula cada ciclo)

    def medir(self, organelos: list["Organelo"]) -> float:
        """M = suma de (costo_base) sobre los organelos EXPRESADOS, con piso M0 (un
        organismo nunca tiene complejidad < la mínima). Los silenciados/reservados
        no suman: solo cuenta lo que efectivamente se expresa (coherente con la
        economía: la masa es lo que metaboliza)."""
        masa = sum(max(o.costo_base, 0.0) for o in organelos
                   if o.expresar and o.estado in (Estado.PRESENTE, Estado.PARCIAL))
        self.M = max(self.M0, masa)
        return self.M

    def tempo(self) -> float:
        """s(M) = (M/M0)^(1/4): factor por el que se ESTIRAN las τ (Kleiber)."""
        return (self.M / self.M0) ** 0.25

    def eficiencia(self) -> float:
        """r(M) = (M/M0)^(-1/4): factor por el que se escala el COSTO por unidad."""
        return (self.M / self.M0) ** -0.25

    def metabolismo_total(self) -> float:
        """B(M) ∝ M^(3/4): tasa metabólica total (referencial/diagnóstica)."""
        return self.M ** 0.75


# ==============================================================================
# INVARIANTES DE VIABILIDAD (κ) — umbrales orientativos, CALIBRABLES por dominio
# Canon C-N2.8.8 (las 5 condiciones del Teorema) + κ_H (analizabilidad, C-N2.8.16).
# C-N2.8.14a: los κ son universales en FUNCIÓN, pero sus VALORES se calibran por
# dominio. Estos son valores de arranque para la célula madre, no constantes físicas.
# ==============================================================================
KAPPA = {
    'kP':     0.0,    # κ_P persistencia: historia > kP  (hay tiempo biológico acumulado)
    'kDelta': 0.0,    # κ_Δ diferencia:   Δ_struct > kDelta  (hay al menos 2 estados)
    'kO':     1.0e9,  # κ_O oscilación:   0 < |e_R| < kO  (error acotado, no desbordado)
    'kV':     0.0,    # κ_V acoplamiento: A_sys_env > kV  (no desacoplado del entorno)
    'kLF':    0.05,   # κ_LF libertad:    LF >= kLF  (mínimo de libertad funcional)
    'kH':     0.0,    # κ_H analizabilidad: H_conductual > kH  (hay conducta medible)
}


# ==============================================================================
# ORGANISMO — EL HOST QUE TRANSCRIBE EL GENOMA Y CORRE EL CICLO METABÓLICO
# ==============================================================================
class Organismo:
    """Una célula: contiene un conjunto de organelos (expresados desde el genoma),
    un citoplasma (`Milieu`) y un medidor de complejidad (Kleiber). NO conoce los
    internos de ningún organelo: solo corre el ciclo metabólico sobre el milieu.

    El CICLO METABÓLICO (un paso de vida del organismo):
      1. medir complejidad M y derivar tempo s(M) y eficiencia r(M).
      2. ordenar los organelos expresados por dependencia (los que secretan antes
         que los que leen su salida): un orden topológico de la membrana.
      3. para cada organelo, en ese orden: percibir → metabolizar(dt, s) → secretar.
      4. cobrar su costo (costo_base · r(M)) al citoplasma (economía).
    Silenciar un organelo = no incluirlo en el ciclo = sin efecto NI costo, por
    construcción (la conmutabilidad es estructural, no un `if` disperso).
    """

    def __init__(self, nombre: str = "celula_madre", M0: float = 1.0) -> None:
        self.nombre = nombre
        self.milieu = Milieu()                       # el citoplasma
        self.complejidad = MedidorComplejidad(M0)    # las leyes de Kleiber
        self.organelos: dict[str, Organelo] = {}     # organelos por nombre
        self.loci_reservados: dict[str, LocusReservado] = {}  # slots vacíos
        self.t = 0.0                                  # tiempo de vida acumulado

    # ---- Construcción del organismo desde el genoma ----
    def expresar(self, org: Organelo) -> "Organismo":
        """Añade un organelo al organismo. Si es un locus reservado, va al registro
        de loci (no entra al ciclo). Devuelve self para encadenar."""
        if isinstance(org, LocusReservado):
            self.loci_reservados[org.nombre] = org
        else:
            self.organelos[org.nombre] = org
        return self

    def silenciar(self, nombre: str) -> None:
        """Silencia un organelo expresado (queda en el genoma pero fuera del ciclo)."""
        if nombre in self.organelos:
            self.organelos[nombre].expresar = False

    def _orden_metabolico(self) -> list[Organelo]:
        """Ordena los organelos expresados respetando `depende_de` (orden topológico
        simple). Si hay un ciclo de dependencias (no debería), cae a orden de
        inserción y sigue — preferimos correr que abortar."""
        activos = [o for o in self.organelos.values()
                   if o.expresar and o.estado in (Estado.PRESENTE, Estado.PARCIAL)]
        resueltos: list[Organelo] = []
        nombres_resueltos: set[str] = set()
        pendientes = list(activos)
        # Mientras haya pendientes cuyas dependencias ya estén resueltas, emítelos.
        progreso = True
        while pendientes and progreso:
            progreso = False
            quedan = []
            for o in pendientes:
                deps_internas = [d for d in o.depende_de if d in self.organelos]
                if all(d in nombres_resueltos for d in deps_internas):
                    resueltos.append(o); nombres_resueltos.add(o.nombre); progreso = True
                else:
                    quedan.append(o)
            pendientes = quedan
        # Si quedaron pendientes (ciclo de deps), añádelos al final sin romper.
        resueltos.extend(pendientes)
        return resueltos

    def vivir_un_paso(self, dt: float) -> dict[str, Any]:
        """Un ciclo metabólico completo. Devuelve un pequeño parte del paso
        (complejidad, tempo, eficiencia, gasto) para diagnóstico/log."""
        self.milieu.nuevo_paso()
        # 1) Kleiber: medir M y derivar los moduladores globales.
        M = self.complejidad.medir(list(self.organelos.values()))
        s = self.complejidad.tempo()        # estira las τ de cada organelo
        r = self.complejidad.eficiencia()   # abarata el costo por organelo
        # 2-3-4) ciclo en orden de dependencia: percibir → metabolizar → secretar.
        for org in self._orden_metabolico():
            org.percibir(self.milieu)
            org.metabolizar(dt, s)
            org.secretar(self.milieu)
            self.milieu.gastar(org.costo_base * r)   # economía: cobrar el trabajo
        self.t += dt
        return {"t": self.t, "M": M, "tempo_s": s, "eficiencia_r": r,
                "gasto": self.milieu.gasto_metabolico}

    # ---- Salud cosmosemiótica: Λ_Cos, OI e invariantes (lecturas del canon) ----
    def salud(self) -> dict[str, Any]:
        """Lee del milieu las variables canónicas y computa la "salud del cierre".

        Λ_Cos = (Δ_struct · LF) / |e_R| · A_sys-env   (C-N2.8.12): razón cosmosemiótica;
                alta = diferencia + libertad + acoplamiento; baja = el error domina.
        OI    = w_H·H + w_ME·ME + w_XE·XE + w_LF·LF − w_IRDE·IRDE·1(LF≥κ_LF)  (O-N9.14):
                Índice de Organismicidad Integrada. Umbrales: ≥0.7 pleno · 0.4-0.7
                protoorganismo · <0.4 no organismal. Pesos ORIENTATIVOS, calibrables.
        invariantes: las 5+1 condiciones de viabilidad (κ) — True/False por condición.

        Honestidad: si faltan organelos que secreten H/ME/XE/LF, esos términos son 0 y
        el OI sale bajo — correcto: una célula madre mínima aún NO es organismo pleno.
        """
        m = self.milieu
        d   = m.leer('delta_struct', 0.0)     # Δ_struct
        LF  = m.leer('LF', 0.0)               # libertad funcional (Bloque 7)
        eR  = abs(m.leer('e_R', 0.0))         # |error operativo|
        A   = m.leer('A_sys_env', 0.0)        # acoplamiento sistema-entorno
        # Λ_Cos (guarda anti-división por cero en e_R)
        lam = (d * LF) / (eR + 1e-9) * A
        # Invariantes de viabilidad (κ)
        inv = {
            'κ_P persistencia':   m.leer('historia', 0.0) > KAPPA['kP'],
            'κ_Δ diferencia':     d > KAPPA['kDelta'],
            'κ_O error acotado':  0.0 < eR < KAPPA['kO'],
            'κ_V acoplamiento':   A > KAPPA['kV'],
            'κ_LF libertad':      LF >= KAPPA['kLF'],
            'κ_H analizabilidad': m.leer('H_conductual', 1.0) > KAPPA['kH'],
        }
        # OI — Índice de Organismicidad Integrada
        H    = m.leer('H_homeostasis', 0.0)   # homeostasis (termorregulación/rango)
        ME   = m.leer('ME', 0.0)              # memoria externalizada (S_shared)
        XE   = m.leer('XE', 0.0)              # exaptación realizada
        IRDE = m.leer('IRDE', 0.0)           # desviación ética
        componentes_oi = [H, ME, min(1.0, XE), min(LF, 1.0)]
        soporte_oi = sum(componentes_oi) / max(1, len(componentes_oi))
        desviacion_oi = IRDE * min(LF, 1.0) if LF >= KAPPA['kLF'] else 0.0
        OI = soporte_oi * (1.0 - max(0.0, min(1.0, desviacion_oi)))
        OI = max(0.0, min(1.0, OI))
        nivel = ('organismo pleno' if OI >= 0.7
                 else 'protoorganismo' if OI >= 0.4
                 else 'no organismal')
        return {'Lambda_Cos': lam, 'OI': OI, 'nivel_OI': nivel, 'invariantes': inv}

    # ---- Autodescripción: "QUIÉN SOY" ----
    def quien_soy(self) -> str:
        """Genera el censo legible del organismo: su complejidad actual, sus
        organelos expresados/silenciados y sus loci reservados. Pensado para
        retomar el estado sin reprocesar el código."""
        M = self.complejidad.medir(list(self.organelos.values()))
        lineas = [
            "=" * 78,
            f"QUIÉN SOY — organismo '{self.nombre}'",
            "=" * 78,
            f"  complejidad M={M:.2f}  ·  tempo s(M)={self.complejidad.tempo():.3f}  "
            f"·  eficiencia r(M)={self.complejidad.eficiencia():.3f}  "
            f"·  metabolismo total B≈{self.complejidad.metabolismo_total():.2f}",
            f"  organelos expresados: {sum(1 for o in self.organelos.values() if o.expresar)}"
            f" / {len(self.organelos)}   ·   loci reservados: {len(self.loci_reservados)}",
        ]
        # Salud cosmosemiótica (Λ_Cos, OI, invariantes) — solo si ya hubo ciclos.
        s = self.salud()
        lineas.append(f"  Λ_Cos (salud del cierre) = {s['Lambda_Cos']:.3f}   ·   "
                      f"OI (organismicidad) = {s['OI']:.3f} → {s['nivel_OI']}")
        invs = "  ".join(f"{'✓' if ok else '✗'} {k.split()[0]}" for k, ok in s['invariantes'].items())
        lineas.append(f"  invariantes de viabilidad:  {invs}")
        lineas += [
            "-" * 78,
            "ORGANELOS:",
        ]
        for o in self.organelos.values():
            lineas.append(o.descripcion_de_si())
        if self.loci_reservados:
            lineas.append("-" * 78)
            lineas.append("LOCI RESERVADOS (vacíos, a desarrollar):")
            for L in self.loci_reservados.values():
                lineas.append(f"· {L.nombre} ({L.organelo_analogo}) — {L.proposito}")
                lineas.append(f"    membrana tentativa: {L.membrana_tentativa}")
                lineas.append(f"    nota: {L.nota}")
        lineas.append("=" * 78)
        return "\n".join(lineas)


# ==============================================================================
# ORGANELOS DE EJEMPLO  (3 reales, para demostrar que el contrato funciona)
# Portados desde VST_Celula_Madre_001.py respetando lo que su código computa.
# El resto del organismo se portará igual, validando contra el control v1.
# ==============================================================================
class OrganeloPresionDesacople(Organelo):
    """Presión de desacople — la TENSIÓN de no estar acoplado al medio (arousal).

    RESOLUCIÓN DE DERIVA nombre≠función: en CM001 esto se llamaba "Cb"
    (ConscienciaBasica), pero su FÓRMULA computa la presión de desacople integrada
    (e_R·(1−A_sys_env)), NO la consciencia básica canónica C_b=|R₁| (registrar el
    propio estado representacional, O-N5.1). Son cosas distintas. Aquí se nombra por
    lo que computa; la C_b canónica vive en el Bloque 5 (OrganeloConscienciaBasica).
    Esta señal ('presion_desacople') es la que gatea juego/ritual (arousal), como en
    CM001. NO es interioridad-contenido; es tensión.

    QUÉ HACE: integra con fuga la presión = error·(1−acople). Sube cuando hay error
    alto Y bajo acople a la vez; se descarga sola con τ.
    CÓMO: leaky integrator  nivel += (presion − nivel/τ)·dt, saturado en [0, max].
    PROCEDENCIA: ConscienciaBasica, VST_Celula_Madre_001.py:422-429 (V155).
    NODO CANÓNICO: deriva de A_sys-env (O-N2.1) y e_R (O-N4.1); gatea la genealogía LF.
    KLEIBER: su τ efectiva = τ · tempo (se alarga al crecer la complejidad).
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="presion_desacople", organelo_analogo="quimiorreceptor de estrés (arousal)",
            procedencia="CM001:422-429 (ex-'Cb'/ConscienciaBasica, V155)",
            nodo_canonico="deriva de O-N2.1 (A_sys-env) y O-N4.1 (e_R) — NO es C_b=|R₁|",
            descripcion=("Presión de desacople (arousal): integrador con fuga de "
                         "error·(1−acople). Tensión, no contenido. Gatea juego/ritual."),
            lee=["e_R", "A_sys_env"], secreta=["presion_desacople", "presion_inst"],
            depende_de=[],   # se alimenta de señales sensorimotoras de base
            costo_base=1.0,
            plast={"tau": 10.0, "max": 500.0},
            criterio="0 < max(presion) < max sostenido (vivo, no saturado)",
            estado=Estado.PRESENTE,
        )
        self.nivel = 0.0       # estado interno: presión de desacople acumulada
        self._presion = 0.0    # presión instantánea calculada en percibir

    def percibir(self, milieu: "Milieu") -> None:
        # Lee error representacional y acople sistema-entorno del citoplasma.
        e_R = milieu.leer("e_R", 0.0)
        A = milieu.leer("A_sys_env", 1.0)
        self._presion = e_R * (1.0 - A)   # la conjunción error∧desacople

    def metabolizar(self, dt: float, tempo: float) -> None:
        tau = self.plast["tau"] * tempo     # Kleiber: τ se estira con M
        self.nivel = max(0.0, min(self.plast["max"], self.nivel + (self._presion - self.nivel / tau) * dt))

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("presion_desacople", self.nivel, guardar_historial=True)
        milieu.secretar("presion_inst", self._presion)


class OrganeloFatiga(Organelo):
    """Fatiga metabólica = TIEMPO BIOLÓGICO.

    QUÉ HACE: acumula dos cosas — `historia` (monótona, irreversible: la identidad
    que el cuerpo gana viviendo, nunca baja) y `fatiga_activa` (recuperable en
    reposo). La fatiga degrada ganancia, ensancha zona muerta y mete temblor.
    CÓMO: historia += |delta|; en trabajo fatiga_activa += costo, en reposo decae
    con τ_recuperacion. PROCEDENCIA: FatigaMetabolica, v2:391-406 (V155/V180c).
    NOTA HONESTA: el linaje tiene un residuo abierto (V150): la recuperación
    post-reposo a veces sube en vez de bajar (−6%). Se porta tal cual; se arregla
    como adaptación, no en silencio.
    KLEIBER: τ_recuperacion · tempo.
    """

    def __init__(self) -> None:
        super().__init__(
            nombre="fatiga", organelo_analogo="mitocondria (balance energético)",
            procedencia="v2:391-406 (FatigaMetabolica, V155/V180c)",
            descripcion=("Tiempo biológico: historia irreversible (identidad) + "
                         "fatiga activa recuperable. Degrada ganancia/precisión."),
            lee=["delta_real", "costo_trabajo", "en_reposo"],
            secreta=["historia", "fatiga_activa", "factor_gain"],
            depende_de=[],
            costo_base=1.0,
            plast={"k_gain": 0.00015, "tau_recuperacion": 300.0, "techo": 20000.0},
            criterio="historia>0 monótona; fatiga_activa recupera en reposo",
            estado=Estado.PARCIAL,   # PARCIAL por el residuo de recuperación (V150)
        )
        self.historia = 0.0
        self.fatiga_activa = 0.0
        self._factor_gain = 1.0

    def percibir(self, milieu: "Milieu") -> None:
        self._delta = abs(milieu.leer("delta_real", 0.0))
        self._costo = milieu.leer("costo_trabajo", 0.0)
        self._reposo = bool(milieu.leer("en_reposo", True))

    def metabolizar(self, dt: float, tempo: float) -> None:
        self.historia += self._delta   # irreversible: el tiempo no vuelve
        if not self._reposo:
            self.fatiga_activa += self._costo
        else:
            tau = self.plast["tau_recuperacion"] * tempo
            self.fatiga_activa *= math.exp(-dt / tau)
        self.fatiga_activa = min(self.fatiga_activa, self.plast["techo"])
        # factor de ganancia: cae exponencialmente con la fatiga acumulada.
        self._factor_gain = max(0.2, min(1.0, math.exp(-self.plast["k_gain"] * self.fatiga_activa)))

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar("historia", self.historia, guardar_historial=True)
        milieu.secretar("fatiga_activa", self.fatiga_activa)
        milieu.secretar("factor_gain", self._factor_gain)


class OrganeloMarcapasos(Organelo):
    """Marcapasos sensorimotor mínimo (ejemplo de organelo SIN entradas: solo
    secreta). Aquí es un stub didáctico que inyecta señales base (e_R, A_sys_env,
    en_reposo) para que Cb y fatiga tengan de qué alimentarse en la demo. En el
    organismo real, estas señales las produce el aparato motor (gradiente, error).
    Sirve para mostrar que un organelo puede ser pura fuente (como un nodo
    marcapasos cardíaco) y que el milieu desacopla productores de consumidores."""

    def __init__(self, e_R: float = 8.0, A_sys_env: float = 0.4) -> None:
        super().__init__(
            nombre="marcapasos", organelo_analogo="nodo marcapasos",
            procedencia="(stub de demostración del genoma)",
            descripcion="Fuente de señales base sensorimotoras (demo). No lee nada.",
            lee=[], secreta=["e_R", "A_sys_env", "delta_struct", "delta_real", "costo_trabajo", "en_reposo"],
            costo_base=0.5, estado=Estado.PRESENTE,
        )
        self.plast = {"e_R": e_R, "A_sys_env": A_sys_env}

    def secretar(self, milieu: "Milieu") -> None:
        # Inyecta un entorno mínimo: error medio, acople parcial, algo de trabajo.
        milieu.secretar("e_R", self.plast["e_R"])
        milieu.secretar("A_sys_env", self.plast["A_sys_env"])
        milieu.secretar("delta_struct", 0.30)   # diferencia estructural mínima del medio (Δ_struct>0)
        milieu.secretar("delta_real", 0.05)
        milieu.secretar("costo_trabajo", 0.05)
        milieu.secretar("en_reposo", False)


# ==============================================================================
# LOCUS DESARROLLADO — GENÉTICA DEL ALTRUISMO (Boorman & Levitt 1980) → PLURICELULAR (O-N22)
# Era un LocusReservado VACÍO (principio 5 / PRE O-N8.19); DESARROLLADO 24-jun-2026: el genoma
# dijo "se sustituye por un Organelo real con esta misma membrana" — aquí está. Reemplazo
# retro-compatible: locus_altruismo_boorman() ahora devuelve el organelo REAL (ver alias al final).
# ==============================================================================
def beta_crit(delta_A: float, e_R: float, LF: float) -> float:
    """O-N22.2 — UMBRAL DE COOPERACIÓN. Δ_struct_crit = {1 − [ΔA/(ΔA+e_R)]^(1/LF)}^(1/2).
    Baja con LF↑ y con e_R↓ (cooperar es más fácil con más libertad / menos error).
    Guardas: LF≤0 ó ΔA+e_R≤0 ⇒ 1.0 (umbral inalcanzable). Función pura (testeable aislada)."""
    dA = max(0.0, float(delta_A)); er = max(0.0, float(e_R)); lf = float(LF)
    if lf <= 0.0 or (dA + er) <= 0.0:
        return 1.0
    ratio = min(1.0, max(0.0, dA / (dA + er)))
    return math.sqrt(max(0.0, 1.0 - ratio ** (1.0 / lf)))


def _sigmoid_altruismo(x: float) -> float:
    if x < -60.0:
        return 0.0
    if x > 60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class OrganeloAltruismo(Organelo):
    """Gobernanza de la cooperación inter-celular (locus de Boorman DESARROLLADO).

    Decide la disposición a cooperar/agregarse leyendo el estado del OTRO, vía:
      · β_crit (O-N22.2): umbral que el Δ_struct social debe superar.
      · Hamilton: ΔA_sys-env / e_R > 1 / S_shared   (≡ r·σ > τ).
      · Ψ_alma (O-N3.4b): el otro tratado como SUJETO (otro.Cb>0) ∧ LF≥κ_LF. SIN esto
        NO hay altruismo voluntario (desalmamiento U0) → la multicelularidad NO se impone.
      · Simbiosis (O-N9.9): mutualismo σ=(+,+) sostenido τ>τ_min ∧ costo_desacople>min.
    `coopera`=True es la condición CONJUNTIVA que habilita V'=V₁⊕V₂ (C-N8/O-N22.7); la
    agregación la ejecuta el nivel multicélula, no este organelo (que sólo gobierna).
    Membrana = la del locus reservado original + diagnósticos O-N22. Constantes CALIBRABLES."""

    def __init__(self, plast=None) -> None:
        base = dict(
            tau_min=8.0,               # O-N9.9: mutualismo sostenido (s) mínimo antes de fusión
            costo_desacople_min=0.30,  # O-N9.9: separar debe costar >30% de A
            umbral_coopera=0.50,       # disposición mínima para gatear coopera
            k_sigmoid=4.0,             # pendiente de emergencia de la disposición
            tau_disp=0.5,              # τ propia de la relajación (la estira Kleiber)
            peso_costo=0.5,            # cuánto penaliza costo_cooperar a la disposición
        )
        if plast:
            base.update(plast)
        super().__init__(
            nombre="altruismo",
            organelo_analogo="gen regulador de cooperación (vía altruismo)",
            procedencia="VST_Genoma O-N22 (desarrollo del locus PRE/O-N8.19 · Boorman&Levitt 1980)",
            descripcion=("Gobernanza de la cooperación inter-celular: disposición a cooperar/agregarse "
                         "vía β_crit (O-N22.2) + Hamilton (ΔA/e_R>1/S_shared) + Ψ_alma (el otro como "
                         "sujeto). Habilita el salto VOLUNTARIO al pluricelular (coopera ⇒ V'=V₁⊕V₂). "
                         "Anti-Shannon: si el otro no es sujeto, la disposición NO emerge."),
            nodo_canonico="O-N22 (Dinámica del Altruismo) · C-N8 · O-N9.9 · O-N3.4b · PRE O-N8.19",
            lee=["otro.valencia", "otro.Cb", "costo_cooperar", "estado_reproductivo",
                 "delta_struct", "LF", "e_R", "A_sys_env", "ME", "A_sys_env_solo"],
            secreta=["disposicion_cooperar", "altruismo.beta_crit", "altruismo.supera_umbral",
                     "altruismo.hamilton_ok", "altruismo.tau_simbiosis", "altruismo.costo_desacople",
                     "altruismo.psi_alma", "altruismo.coopera"],
            costo_base=0.8,
            plast=base,
            criterio="coopera=True sólo si β>β_crit ∧ Hamilton ∧ Ψ_alma ∧ τ>τ_min ∧ costo_desacople>min",
            estado=Estado.PRESENTE,
            expresar=True,
        )
        self.tau = 0.0; self.disposicion = 0.0; self.beta_crit = 1.0
        self.supera_umbral = False; self.hamilton_ok = False
        self.costo_desacople = 0.0; self.psi_alma = 0.0; self.coopera = False
        self._p: dict = {}

    def percibir(self, milieu: "Milieu") -> None:
        g = milieu.leer
        A_solo = g("A_sys_env_solo", None)
        self._p = dict(
            otro_valencia=float(g("otro.valencia", 0.0)), otro_Cb=float(g("otro.Cb", 0.0)),
            costo_cooperar=float(g("costo_cooperar", 0.0)), estado_reproductivo=float(g("estado_reproductivo", 1.0)),
            delta_struct=float(g("delta_struct", 0.0)), LF=float(g("LF", 0.0)),
            eR=abs(float(g("e_R", 0.0))), A=float(g("A_sys_env", 0.0)),
            S_shared=float(g("ME", g("S_shared", 0.0))),
            A_solo=(float(A_solo) if A_solo is not None else None),
        )

    def metabolizar(self, dt: float, tempo: float) -> None:
        p = self.plast; q = self._p
        eR = max(q["eR"], 1e-9); LF = max(q["LF"], 0.0); S = max(q["S_shared"], 1e-9)
        self.beta_crit = beta_crit(q["A"], q["eR"], LF if LF > 0 else 1e-9)
        self.supera_umbral = q["delta_struct"] > self.beta_crit
        self.hamilton_ok = (q["A"] / eR) > (1.0 / S)
        otro_es_sujeto = (q["otro_Cb"] > 0.0) and (q["otro_valencia"] >= 0.0)
        self.psi_alma = (min(1.0, q["otro_Cb"]) if (otro_es_sujeto and LF >= KAPPA["kLF"]) else 0.0)
        mutualismo = (q["A"] > 0.0) and (q["otro_valencia"] > 0.0)
        self.tau = (self.tau + dt) if mutualismo else 0.0
        if q["A_solo"] is not None and q["A"] > 1e-9:
            self.costo_desacople = max(0.0, (q["A"] - q["A_solo"]) / q["A"])
        else:
            self.costo_desacople = 0.0
        favorable = self.supera_umbral and self.hamilton_ok and (self.psi_alma > 0.0)
        margen = (q["delta_struct"] - self.beta_crit) + ((q["A"] / eR) - (1.0 / S))
        objetivo = self.psi_alma * _sigmoid_altruismo(p["k_sigmoid"] * margen) if favorable else 0.0
        objetivo *= max(0.0, 1.0 - p["peso_costo"] * q["costo_cooperar"])
        objetivo *= max(0.0, min(1.0, q["estado_reproductivo"]))
        tau_disp = max(1e-6, p["tau_disp"] * max(tempo, 1e-6))
        self.disposicion += (objetivo - self.disposicion) * (dt / tau_disp)
        self.disposicion = max(0.0, min(1.0, self.disposicion))
        self.coopera = bool(
            favorable and self.tau > p["tau_min"] * max(tempo, 1e-6)
            and self.costo_desacople > p["costo_desacople_min"] and self.disposicion > p["umbral_coopera"]
        )

    def secretar(self, milieu: "Milieu") -> None:
        s = milieu.secretar
        s("disposicion_cooperar", self.disposicion, guardar_historial=True)
        s("altruismo.beta_crit", self.beta_crit)
        s("altruismo.supera_umbral", 1.0 if self.supera_umbral else 0.0)
        s("altruismo.hamilton_ok", 1.0 if self.hamilton_ok else 0.0)
        s("altruismo.tau_simbiosis", self.tau)
        s("altruismo.costo_desacople", self.costo_desacople)
        s("altruismo.psi_alma", self.psi_alma)
        s("altruismo.coopera", 1.0 if self.coopera else 0.0)


def organelo_altruismo(plast=None) -> "OrganeloAltruismo":
    """Factory del locus de altruismo DESARROLLADO (O-N22). Misma membrana que el locus reservado."""
    return OrganeloAltruismo(plast=plast)


def locus_altruismo_boorman() -> "OrganeloAltruismo":
    """COMPAT: nombre histórico del locus. Antes devolvía un LocusReservado VACÍO; tras el
    desarrollo (24-jun-2026, O-N22) devuelve el ORGANELO REAL con la misma membrana, de modo que
    todo el linaje que ya lo expresaba (Bloque05/07/08, célula funcional, __main__) lo hereda."""
    return organelo_altruismo()


# ==============================================================================
# MANIFIESTO DEL GENOMA — EL CENSO COMPLETO  (todas las capacidades del linaje)
# ==============================================================================
# Cada entrada describe una capacidad y su ESTADO en el genoma. Es la referencia
# legible de "qué tengo y qué me falta", cruzada con INVENTARIO_RESCATE.md.
# (Las que aún no son Organelo de pleno derecho figuran aquí como metadatos: son
# el contrato de lo que falta portar/validar contra v1.)
@dataclass
class GenEspec:
    """Una línea del genoma: la ficha de una capacidad (esté portada o no)."""
    nombre: str
    organelo_analogo: str
    procedencia: str
    estado: Estado
    descripcion: str
    grupo: str = ""              # cómo se clasificó en el inventario de rescate
    nota: str = ""
    nodo_canonico: str = ""      # nodo de la Teoría Cosmosemiótica que realiza

# El censo completo. Agrupado por estado para lectura rápida.
GENOMA: list[GenEspec] = [
    # ---- PRESENTES: capacidades ya vivas en el organismo v2 ----
    GenEspec("orientacion_gradiente", "tropismo/eje motor", "V132/V147→v2",
             Estado.PRESENTE, "Driver único de conducta: ωA−ωB. Orienta al gradiente.", "espina"),
    GenEspec("lateralidad", "asimetría L/R + cuerpo calloso", "V122→v2",
             Estado.PRESENTE, "4 hemisferios acoplados; lateralidad + R₂ coexisten.", "espina"),
    GenEspec("presion_desacople", "quimiorreceptor de estrés (arousal)", "V155→CM001",
             Estado.PRESENTE, "Presión de desacople (arousal): gatea juego/ritual. NO es C_b canónico.",
             "espina", nodo_canonico="deriva O-N2.1/O-N4.1"),
    GenEspec("consciencia_basica", "registro del propio estado representacional", "Bloque 5",
             Estado.PRESENTE, "C_b=|R₁|: registra el propio estado representacional (cuántos estímulos procesa).",
             "consciencia funcional", nodo_canonico="O-N5.1"),
    GenEspec("meta_representacion", "auto-modelo (R₂)", "Bloque 5",
             Estado.PRESENTE, "R₂=R(R): representación de la representación. Funda LF_struct (O-N13.8).",
             "consciencia funcional", nodo_canonico="O-N5.2"),
    GenEspec("self", "operador de identidad", "Bloque 5",
             Estado.PRESENTE, "Self=operador(R₂): gestiona las meta-representaciones; coherencia de identidad.",
             "consciencia funcional", nodo_canonico="O-N5.3"),
    GenEspec("fatiga", "mitocondria", "V155/V180c→v2",
             Estado.PARCIAL, "Tiempo biológico; residuo de recuperación −6% (V150).", "espina",
             "PARCIAL: arreglar recuperación como adaptación, no en silencio."),
    GenEspec("memoria_ausencia", "memoria de corto plazo", "V153/V155→v2",
             Estado.PRESENTE, "Confianza decae en silencio; mantiene último setpoint.", "espina"),
    GenEspec("juego", "conducta exploratoria", "V156/V165→v2",
             Estado.PRESENTE, "Acción 'como si' cuando Cb alta; inhibido por ritual.", "espina"),
    GenEspec("valencia_deliberacion", "núcleo de valor (R_op/R_af)", "V176/V181→v2",
             Estado.PRESENTE, "Valencia diferencial + deliberación con costo; veto (No).", "espina"),
    GenEspec("memoria_episodica", "hipocampo", "V180c→v2",
             Estado.PARCIAL, "Evitación de trauma real; recall explícito NO demostrado (0/50).", "espina",
             "PARCIAL: la conducta es real, el mecanismo de recall está por demostrar."),
    GenEspec("ritual", "ritmo estabilizador", "V165→v2",
             Estado.PRESENTE, "Detecta patrón temporal; estabiliza; inhibe juego.", "espina"),
    GenEspec("meta_Rᴿ", "auto-monitor (corteza observadora)", "V167→v2",
             Estado.PRESENTE, "Observa desajuste y registra; NO inhibe (por diseño).", "espina"),
    GenEspec("campo_phi", "citoplasma del hemisferio (medio excitable)", "V122→v2",
             Estado.PRESENTE, "Sustrato del campo Φ por hemisferio: difusión (laplaciano) + "
             "reacción biestable Φ(1−Φ²) + forzamiento de borde. El MEDIO que todo lo demás modula.",
             "campo", "Estaba implícito; es el organelo base sobre el que viven W/atractor/oscilador."),
    GenEspec("campo_W", "retículo (plasticidad)", "v72b→v2",
             Estado.PRESENTE, "Plasticidad hebbiana del campo Φ.", "campo"),
    GenEspec("campo_atractor", "atractor de historia", "v72c/v80h→v2",
             Estado.PRESENTE, "Phi_int_historia reinyectado (memoria estructural).", "campo"),
    GenEspec("campo_oscilador", "oscilador/modos propios", "v70→v2",
             Estado.PRESENTE, "Resonancia por nodo (medio, no driver).", "campo"),
    GenEspec("campo_dualW", "retículo dual (identidad/contexto)", "v80h→v2",
             Estado.PRESENTE, "Plasticidad dual W_prof(τ≈20s)/W_rec(τ≈2s); reemplaza a campo_W si se expresa.",
             "campo (condicional)", "Implementado en CM001 pero por defecto SILENCIADO (flag OFF); pasó smoke."),
    GenEspec("campo_inhib_lateral", "inhibición lateral", "V122→v2",
             Estado.PARCIAL, "Congela al lento; sin efecto en sonda aislada.", "campo",
             "PARCIAL: falta atribución limpia (no se ejercita sin el otro hemisferio)."),
    GenEspec("campo_lambda", "medidor nativo Λ/LF", "V122→v2",
             Estado.PARCIAL, "Mide multiplicidad de atractores; salió PLANO.", "campo",
             "PARCIAL: latente no expresada en conducta (Λ/LF plano OFF==ON)."),

    # ---- AUSENTES: capacidades validadas/exploradas, aún no portadas (a recuperar) ----
    GenEspec("predictor_trayectoria", "corteza predictiva", "V135-V138",
             Estado.AUSENTE, "Anticipación: extrapola pos+vel·h(+½·acc·h²). VALIDADO (MAE<12°).", "(A) candidato fuerte",
             "El más valioso: única familia (A) con criterio ✅. Eliminado V139 sin test."),
    GenEspec("relajacion_a_centro", "tono de reposo", "V134-V139",
             Estado.AUSENTE, "En ausencia, relaja el setpoint a un centro. VALIDADO (O-N9.1).", "(A) candidato",
             "Hoy solo decae la confianza; no relaja a centro."),
    GenEspec("Cb_global", "estado fisiológico de conjunto", "V174",
             Estado.AUSENTE, "Presión de desacople GLOBAL (no por motor). Gatea ritual/juego.", "(A) candidato",
             "Reaparecerá ligada al MedidorComplejidad (estado de conjunto)."),
    GenEspec("act_perm", "permeabilidad activa de membrana", "v81",
             Estado.AUSENTE, "Modula α (acople al estímulo) desde el propio campo.", "(A) candidato",
             "Coherente con 'campo como medio'."),
    GenEspec("membrana_sensorial", "membrana receptora", "V111",
             Estado.AUSENTE, "Preprocesa la entrada (inst+envolvente+derivada+tanh).", "(A)",
             "Hoy la entrada llega cruda a la frontera."),
    GenEspec("olvido_selectivo", "poda sináptica", "v80h",
             Estado.AUSENTE, "Olvido modulado por eficiencia en la W dual.", "(A)",
             "No portado a v2 (requiere GED, ausente). La W dual entró con olvido fijo."),
    GenEspec("atencion_derivadas", "atención (a dΩ/dt)", "V118 (prometida)",
             Estado.AUSENTE, "Atención a TENDENCIAS, la capacidad real que el nombre prometía.", "(A) reconstruir",
             "La implementación histórica divergía (coseno-softmax). Reconstruir la capacidad real."),
    GenEspec("exploracion_actuadores", "exploración activa", "v97.2 (prometida)",
             Estado.AUSENTE, "Perturbar/explorar actuadores (lo que el nombre prometía).", "(A) reconstruir",
             "El ExploradorActuadores histórico solo registraba; reconstruir la capacidad real."),

    # ---- DÍADA: capacidad relacional, latente (principio 1: NO es fuera de alcance) ----
    GenEspec("memoria_relacional", "reconocimiento del otro", "V182A5",
             Estado.AUSENTE, "Confianza sigmoide por banda hacia el otro. VALIDADO ON/OFF.", "díada (latente)",
             "Sustrato de la cultura acumulativa. Membrana hacia el MILIEU EXTERNO."),
    GenEspec("buffer_acoplamiento", "sincronía con el otro", "V182A-v3",
             Estado.AUSENTE, "Buffer de la trayectoria del otro (val,Cb,D) + convergencia.", "díada (latente)",
             "Degradado a escalar en v7; recuperar la versión con estado del otro."),

    # ---- RESERVADO (principio 5): sistema semiótico EMERGENTE / protolenguaje (hipótesis Alexis 24-jun-2026) ----
    GenEspec("sistema_semiotico_emergente", "protolenguaje (signo emergente)", "RESERVADO (a DEMOSTRAR, no diseñar)",
             Estado.RESERVADO, "La comunicación NO es protocolo diseñado ni símbolos predefinidos: EMERGE de la "
             "HISTORIA COMPARTIDA (orden: experiencia→memoria→sentido→necesidad→comunicación→símbolos). El tono es "
             "sólo SUSTRATO físico; el significado es la red de asociaciones de la propia historia (codebook). NO se "
             "implementa, se DEMUESTRA de forma falsable: (1) aprendido no impuesto; (2) especificidad funcional "
             "(info mutua emisión↔contexto>0); (3) convergencia A↔B; (4) DEPENDENCIA DE HISTORIA COMPARTIDA "
             "(control NULL/SHUFFLED = CRÍTICO; sin historia → no convención, si aparece igual → artefacto); "
             "(5) arbitrariedad (otra historia → otro signo); (6) informatividad (necesidad y suficiencia). "
             "Sustrato = codebook + memoria + metabolismo + necesidad + órgano de comunicación + memoria_relacional.",
             "díada (locus reservado)", nodo_canonico="V182C convención · O-N22 (no-impuesto) · PRE O-N8.19"),

    # ---- DESARROLLADO: locus de altruismo (era reservado, principio 5 → O-N22) ----
    GenEspec("altruismo", "gen regulador de cooperación (vía altruismo)", "VST_Genoma O-N22 (Boorman&Levitt 1980)",
             Estado.PRESENTE, "Genética del altruismo → pluricelular VOLUNTARIO. DESARROLLADO (O-N22): "
             "β_crit + Hamilton + Ψ_alma (gate de sujeto, anti-imposición) + simbiosis O-N9.9 → coopera (V'=V₁⊕V₂).",
             "locus desarrollado", nodo_canonico="O-N22 · C-N8 · O-N9.9 · O-N3.4b · PRE O-N8.19"),
]

# ---- PODADOS CON RAZÓN: explícitamente FUERA del genoma (que nadie los reviva) ----
# No son GenEspec: se listan para memoria, con su razón de exclusión.
PODADOS_CON_RAZON: dict[str, str] = {
    "Pastor": "Regulador EXTERNO de parámetros (v11). Revivirlo = control desde fuera = Shannon. La homeostasis ya vive en la dinámica del campo.",
    "modos_espectrales_riqueza_entropia": "Diagnóstico (v72c) que nunca alimentó criterio. GED sobrevive subsumido.",
    "cuerpo_calloso_escalar": "Sustrato escalar 1-g.d.l. (V123) descartado; la función gated sobrevive inline. (Matiz: la rectificación direccional max(0,diff) se perdió — registrable, no urgente.)",
    "inhibicion_reciproca": "Winner-take-all duro sobre Ω (V123). Ningún término inhibitorio sobrevive V146–V182.",
    "ganglio_G": "No coordinaba: era la rebanada con más aristas en una lista a mano. Divergente.",
    "memoria_largo_plazo_v10": "Media con contador siempre 1 → nunca promediaba. Trivial.",
}


def transcribir_celula_madre_minima() -> Organismo:
    """Transcribe un organismo MÍNIMO de demostración desde el genoma: marcapasos
    + Cb + fatiga + el locus de altruismo reservado. Es el 'ratón' (M pequeño):
    pocas piezas, metabolismo alto, tempo rápido. Sirve para probar el motor y la
    autodescripción. La transcripción del organismo COMPLETO (todos los organelos
    portados desde v2, validados contra v1) es el siguiente paso."""
    o = Organismo(nombre="celula_madre_minima", M0=1.0)
    o.expresar(OrganeloMarcapasos())
    o.expresar(OrganeloPresionDesacople())
    o.expresar(OrganeloFatiga())
    o.expresar(locus_altruismo_boorman())
    return o


# ==============================================================================
# DEMO / AUTOVERIFICACIÓN  — corre solo si se ejecuta el archivo directamente.
# ==============================================================================
if __name__ == "__main__":
    # 1) Transcribe la célula madre mínima y la deja vivir (la salud se mide sobre
    #    un organismo en marcha: antes del primer ciclo el milieu está vacío).
    org = transcribir_celula_madre_minima()
    for _ in range(10):
        parte = org.vivir_un_paso(0.1)

    # 2) "Quién soy" — ahora con salud (Λ_Cos/OI/invariantes) ya significativa.
    print(org.quien_soy())

    # 3) Parte metabólico (Kleiber + economía) y estado interno de los organelos.
    print(f"\nCICLO METABÓLICO (10 pasos de dt=0.1):  tras {parte['t']:.1f}s  "
          f"M={parte['M']:.2f}  s(M)={parte['tempo_s']:.3f}  r(M)={parte['eficiencia_r']:.3f}  "
          f"gasto/paso={parte['gasto']:.3f}")
    print(f"  presion_desacople={org.organelos['presion_desacople'].nivel:.3f}  "
          f"historia={org.organelos['fatiga'].historia:.3f}  "
          f"fatiga_activa={org.organelos['fatiga'].fatiga_activa:.3f}")

    # 3) Resumen del censo del genoma completo (por estado).
    print("\nCENSO DEL GENOMA COMPLETO (manifiesto):")
    for est in Estado:
        items = [g for g in GENOMA if g.estado is est]
        print(f"  {est.value.upper():9}: {len(items):2}  ->  "
              f"{', '.join(g.nombre for g in items)}")
    print(f"  PODADOS (fuera del genoma, no revivir): {', '.join(PODADOS_CON_RAZON)}")
