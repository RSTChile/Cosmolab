#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_Homeostasis — HOMEOSTASIS SCALE-INVARIANT Y EXPANSIÓN MULTICELULAR  ·  "QUIÉN SOY"
================================================================================

QUÉ ES ESTE ARCHIVO
-------------------
La homeostasis del organismo cosmosemiótico, implementada como un organelo
REUTILIZABLE Y SCALE-INVARIANT, más la capa MULTICELULAR que emerge de aplicarlo a
una escala mayor. Importa el motor del genoma (VST_Genoma.py) y los bloques 5/7/8.

EL PRINCIPIO (por qué homeostasis "se expande a multicelular por definición")
-----------------------------------------------------------------------------
La homeostasis es mantener una variable regulada dentro de su rango viable
(C-N5.1: x(t)∈[x_min,x_max]; O-N6.1: fitness=f(A_sys-env)). Es la H del Índice de
Organismicidad (OI, O-N9.14). Por UNIVERSALIDAD ESTRUCTURAL (C-N2.8.14: los
invariantes son universales en función, su valor se calibra por dominio), la MISMA
lógica homeostática opera a tres escalas con el mismo código:
  · ORGANELO  — cada organelo se autorregula (economía, fuga de fatiga).
  · CÉLULA    — un homeostato mantiene el medio interno en rango (da la H del OI).
  · MULTICELULAR — un homeostato COLECTIVO mantiene el acoplamiento del conjunto.
Por eso el salto a multicelular es "por definición": un organismo multicelular ES
un homeostato sobre sus células (igual que la célula es un homeostato sobre sus
organelos). El ejemplo canónico de las hormigas (C-N2.8.14a): termorregulación
COLECTIVA, memoria externalizada (feromonas = S_shared), organismicidad de colonia.

LO QUE LA MULTICELULARIDAD DESBLOQUEA EN EL OI
----------------------------------------------
El OI tenía dos componentes en cero a nivel célula: H (homeostasis) y ME (memoria
externalizada = S_shared, Dicc.146). La homeostasis activa H. La MULTICELULARIDAD
activa ME: el medio compartido (el "aire" de la díada) ES la memoria externalizada.
Con H y ME activos, el OI colectivo puede cruzar a ORGANISMO PLENO (≥0.7).

GUARDA ANTI-SHANNON (no negociable)
-----------------------------------
Que el código pueda AGREGAR células en un colectivo NO significa que el colectivo sea
un organismo VOLUNTARIO. La cohesión voluntaria (Ψ_alma sujeto-sujeto vs desalmamiento,
O-N3.4b) está gobernada por el LOCUS DE ALTRUISMO DE BOORMAN, que sigue RESERVADO Y
VACÍO (PRE, O-N8.19). Por tanto, esta `Multicelula` implementa la DEFINICIÓN estructural
del organismo multicelular (homeostato colectivo + S_shared), pero su cohesión es por
ahora ANDAMIAJE IMPUESTO: el organismo multicelular EMERGENTE/voluntario espera el
desarrollo del locus. Se reporta el OI colectivo marcando esta condición.

KLEIBER A ESCALA (O-N nuestro, extensión)
-----------------------------------------
El multicelular tiene M (complejidad) mayor → por la ley de Kleiber, MENOR metabolismo
por unidad (r↓) y tiempos más largos (s↑): la colonia es "el elefante" frente a la
célula "ratón". El demo lo muestra.
================================================================================
"""

from __future__ import annotations
from typing import Any

from VST_Genoma import (
    Organelo, Estado, Organismo, Milieu, KAPPA, MedidorComplejidad,
)
from escala import Escala


# ==============================================================================
# ORGANELO HOMEOSTASIS — SCALE-INVARIANT  (C-N5.1 · O-N6.1 · componente H del OI)
# ==============================================================================
class OrganeloHomeostasis(Organelo):
    """Homeostasis EMERGENTE estilo DAISYWORLD (Lovelock) — SIN setpoint ni ganancia impuestos.

    ANTES (doble Shannon, derivaba): un controlador PID arrastraba una variable FANTASMA `x`
    hacia un setpoint x0=0.5 (meta impuesta) que además no estaba acoplada a nada real. Cuando la
    perturbación crecía, `x` se alejaba y H colapsaba.

    AHORA: la estabilidad EMERGE del acoplamiento de DOS procesos OPUESTOS (las "margaritas"),
    como en Daisyworld nadie fija 22°C — solo hay margaritas negras/blancas, reproducción
    diferencial y acoplamiento, y la temperatura se autorregula:
      · margarita ORDEN (b, ICR-like): prospera cuando hay DESORDEN (x alto) y lo REDUCE (sentido).
      · margarita ENTROPÍA (w, IRDE-like): prospera cuando hay RIGIDEZ (x bajo) y lo AUMENTA (variación).
    `x` = desorden interno, DRIVEN por el ESTRÉS REAL del organismo (e_R + desacople A_sys_env), no por
    un fantasma. El equilibrio EMERGE de la SIMETRÍA de la competencia (no se impone) y resiste la
    perturbación de forma robusta (la respuesta no-lineal de las margaritas, mucho más rígida que un kp).
    H = CALIDAD DE REGULACIÓN EMERGENTE = BALANCE de las dos margaritas (ambas vivas ⇒ regulación
    bidireccional activa; una colapsa ⇒ regulación fallando). SIN meta, SIN distancia a setpoint.
    Constantes FISIOLÓGICAS declaradas (α crecimiento, μ mortalidad, κ acoplamiento, rango viable),
    no ajustes "para estabilizar". SCALE-INVARIANT (célula y colectivo). Kleiber: α se ralentiza con s(M).
    """

    def __init__(self, variable: str = "x_interna", x_min: float = 0.0, x_max: float = 1.0,
                 # SALIDA RENOMBRADA 6-ago-2026. Se llamaba `H_homeostasis` y no medía homeostasis:
                 # `H = tension · viable · tendencia · estab_A`, y tres de esos cuatro factores son
                 # funciones de `A_sys_env`, el ACOPLAMIENTO CON EL ENTORNO. El comentario de más
                 # abajo lo dice literal: «¿la competencia ICR↔IRDE está SOSTENIENDO el acoplamiento
                 # del organismo con su entorno?». Eso es ADAPTACIÓN —seguir agarrado al nicho
                 # mientras el mundo cambia—, pregunta legítima pero distinta: la homeostasis es
                 # mantener las variables INTERNAS en rango A PESAR del entorno. El nombre pasa al
                 # índice que sí mide eso (VST_IndiceHomeostasis) y esta señal se llama como lo que
                 # hace. `_daisy` porque es la versión Daisyworld, anterior a la de
                 # HomeostasisEmergente (`acople_sostenido`, canónica): son dos generaciones del
                 # MISMO instrumento, y tenerlas peleando un solo nombre era la raíz del enredo —
                 # ésta corría primero cada paso y le pisaba el valor a la otra antes de que nadie
                 # pudiera leerlo.
                 perturb: str = "e_R", perturb_escala: float = 0.01,
                 salida: str = "acople_sostenido_daisy",
                 alpha: float = 2.0, mu: float = 0.3, kappa: float = 0.6,
                 escala_dA: float = 0.002, ema_A: float = 0.05, banda: float = 0.15,
                 x0=None, kp=None,             # x0/kp: COMPAT, IGNORADOS (Daisyworld no tiene setpoint ni ganancia)
                 A_min=None) -> None:          # A_min: COMPAT, IGNORADO desde 8-ago-2026 (ver `A_piso` en metabolizar)
        super().__init__(
            nombre=f"homeostasis::{variable}", organelo_analogo="Daisyworld (regulación emergente)",
            procedencia="Lovelock Daisyworld / C-N5.1 / O-N6.1 (componente H del OI)",
            nodo_canonico="C-N5.1 (x∈[x_min,x_max]) · O-N6.1 · O-N9.14 (H del OI) · emergencia (no controlador)",
            descripcion=(f"Regula '{variable}' por ACOPLAMIENTO de dos procesos opuestos (margaritas "
                         f"orden/entropía), SIN setpoint. H∈[0,1]=calidad de regulación emergente. Scale-invariant."),
            lee=[perturb, "A_sys_env"],
            secreta=[salida, variable, f"{variable}_en_rango", f"{variable}_orden", f"{variable}_entropia",
                     f"{variable}_acopl_tendencia", f"{variable}_autoencierro", f"{variable}_anestesia",
                     f"{variable}_A_estabilidad", f"{variable}_estres", f"{variable}_perturb_habitual",
                     f"{variable}_perturb_exceso", f"{variable}_esfuerzo", f"{variable}_costo_activo"],
            depende_de=[],
            costo_base=1.0,
            plast={"x_min": x_min, "x_max": x_max, "perturb_escala": perturb_escala,
                   "alpha": alpha, "mu": mu, "kappa": kappa,
                   # `A_piso` NO es un número propio: ES `banda`. Se guarda con nombre aparte para
                   # que se pueda leer (y replayar) el piso sin tocar la tolerancia, pero nace de ella.
                   "escala_dA": escala_dA, "A_piso": banda, "ema_A": ema_A, "banda": banda},
            criterio="H alto sostenido (las dos margaritas vivas y balanceándose: regulación activa)",
            estado=Estado.PRESENTE,
        )
        self.variable = variable
        self.perturb = perturb
        self.salida = salida
        self.esc_perturb = Escala()        # lo habitual de la perturbación PARA ESTE organismo (ver percibir)
        self._eR_habitual = 0.0; self._exceso = 0.0
        self.x = (x_min + x_max) / 2.0     # condición INICIAL (no es setpoint: nada la arrastra aquí)
        self.b = 0.5                       # margarita ORDEN (ICR-like)
        self.w = 0.5                       # margarita ENTROPÍA (IRDE-like)
        self.H = 1.0
        self._estres = 0.0; self._eR = 0.0; self._A = 1.0; self._A_prev = None; self._dA_ema = 0.0
        self._b_prev = self.b; self._w_prev = self.w
        self.autoencierro = 0.0; self.anestesia = 0.0
        self._A_ema = None; self._varA_ema = 0.0; self.estab_A = 1.0

    def percibir(self, milieu: "Milieu") -> None:
        # "Luminosidad" externa de Daisyworld = ESTRÉS REAL: error operativo + desacople.
        # A_sys-env (O-N2.1) = la variable UNIFICADORA del modelo (acoplamiento con el nicho).
        self._eR = abs(milieu.leer(self.perturb, 0.0))
        self._A = max(0.0, min(1.0, milieu.leer("A_sys_env", 1.0)))
        # ── EL ESTRÉS ES EL ERROR QUE NO SE ESTÁ CERRANDO (6-ago-2026) ──────────────────
        # ANTES: `estrés = perturb_escala · e_R`, el error CRUDO. Y `e_R` es un ángulo en
        # grados —`|setpoint − orientación|` sobre un rango de ±90°— o sea que rastrear una
        # fuente que se mueve produce, siempre, unos grados de error. Diez grados de desvío
        # es lo que hace cualquier cosa que sigue algo; un búho localizando presa no lo hace
        # mejor. Con la forma cruda eso se leía como estrés permanente de 0,1.
        #
        # MEDIDO, y es lo que obliga a cambiarlo: en Abraxas el estrés nunca bajó de 0,0960
        # en 2.461 pasos, y el único día tranquilo de la semana fue el único día en que el
        # organismo NO OYÓ NADA (4-ago: las dos energías en 0,000 y `e_R` clavado en su piso
        # de 0,500 el 52,5 % del tiempo, contra ~9,5 los días con mundo). Bajo la definición
        # vieja, la única forma de no estar estresado era estar sordo: tener mundo ERA el
        # estrés. Y encima se cobraba — el costo homeostático corría en cada paso.
        #
        # AHORA sólo cuenta el error que EXCEDE lo que este organismo consigue sostener
        # habitualmente. El halcón que corrige su rumbo no está estresado; el que no logra
        # cerrar la brecha, sí. Un organismo que rastrea bien vuelve a tener cero, y el
        # estrés sube de verdad cuando la fuente se le escapa, que es cuando debe costar.
        #
        # NINGÚN NÚMERO NUEVO, Y NINGUNO MENOS: `perturb_escala` sigue siendo el mismo y
        # sigue aplicándose a grados —de hecho 0,01 ≈ 1/90, la fracción del propio rango de
        # movimiento, que es una lectura estructural legítima—. Lo único que se mueve es
        # DÓNDE ESTÁ EL CERO. La referencia sale de `escala.py` (una sola implementación
        # para todo el organismo, con su tasa declarada) y no de una constante local.
        #
        # POR QUÉ RELATIVIZAR AQUÍ NO ES EL TRINQUETE DE LA ADVERTENCIA 3: la referencia
        # tiene que ser AJENA a lo que el criterio decide. Verificado — `x_interna` no
        # realimenta el lazo de orientación: sólo la leen el índice de homeostasis, el costo
        # metabólico y la UI. El estrés no puede bajarse a sí mismo el listón moviendo el
        # `e_R` contra el que se compara.
        #
        # Y POR QUÉ NO ES LA ADVERTENCIA 2: esto es una PERCEPCIÓN, no una condición de
        # viabilidad. El ojo se adapta a la oscuridad; lo que no se relativiza es el mínimo
        # de fotones bajo el cual no ve. Aquí las condiciones absolutas siguen absolutas y
        # en otra parte: la reserva en `H_var_reserva`, el acoplamiento en `A_min`.
        #
        # LÍMITE CONOCIDO: este organelo no está en la lista de persistibles, así que la
        # escala se reinicia en cada arranque y el organismo pasa sus primeros pasos sin
        # saber qué error es «el suyo». Mientras no haya historia se ABSTIENE (exceso 0, ver
        # abajo) en vez de inventar estrés, que es lo que manda `escala.py`; pero lo correcto
        # es que la escala sobreviva al apagón, como ya sobreviven las de memoria.
        # SEMBRAR LA ESCALA EN EL PRIMER ERROR QUE VE, y hubo que aprenderlo mirándolo nacer:
        # `Escala` arranca su media en 0,0 y la mueve un 5 % por observación, así que sin
        # sembrarla el organismo pasa su primer minuto de vida creyendo que su error habitual
        # es CERO — todo su error le parece exceso. MEDIDO en el arranque de las 23:38:
        # `perturb_habitual` cayó a 0,9875 con `e_R` en 21,15 y el exceso marcó 20,16, o sea
        # un estrés de 0,216 contra los 0,110 de la fórmula vieja. Nacer no puede ser el
        # momento más angustioso de la vida por un artefacto del promedio.
        if not self.esc_perturb.n:
            self.esc_perturb.restore({"n": 1, "media": self._eR, "dispersion": 0.0})
        else:
            self.esc_perturb.observar(self._eR)      # se observa DESPUÉS: el paso no se compara consigo mismo
        # Y MIENTRAS NO HAYA HISTORIA, ABSTENERSE: es el contrato del propio módulo («quien
        # clasifique debe abstenerse en vez de inventar»). Un recién nacido todavía no sabe
        # cuál es su error normal; no puede estar estresado por no saberlo.
        habitual = self.esc_perturb.media if self.esc_perturb.madura else self._eR
        self._eR_habitual = habitual
        self._exceso = max(0.0, self._eR - habitual)
        self._estres = self.plast["perturb_escala"] * self._exceso + 0.02 * (1.0 - self._A)

    def metabolizar(self, dt: float, tempo: float) -> None:
        p = self.plast
        alpha = p["alpha"] / tempo         # Kleiber: las margaritas crecen más lento a mayor complejidad
        mu = p["mu"]; kappa = p["kappa"]
        A = self._A
        if self._A_prev is None:
            self._A_prev = A
        self._dA_ema = (1.0 - p["ema_A"]) * self._dA_ema + p["ema_A"] * (A - self._A_prev)
        # DESACOPLE = cuánto CAE el acoplamiento (O-N2.1). Señal que MOVILIZA el ORDEN (recobrar el nicho).
        desacople = max(0.0, min(1.0, -self._dA_ema / p["escala_dA"]))
        libre = max(0.0, 1.0 - self.b - self.w)
        # ICR↔IRDE ACOPLADOS POR A_sys-env (O-N2.1):
        #   ORDEN (b, ICR): prospera ante DESORDEN (x alto) O ante DESACOPLE de A (hay que recobrar el nicho).
        #   ENTROPÍA (w, IRDE): prospera ante RIGIDEZ (x bajo) SÓLO si el acoplamiento está SANO (no en crisis).
        g_b = alpha * max(self.x, desacople) * libre - mu
        g_w = alpha * (1.0 - self.x) * A * libre - mu
        self.b = max(0.0, min(1.0, self.b + self.b * g_b * dt))
        self.w = max(0.0, min(1.0, self.w + self.w * g_w * dt))
        # x = desorden interno: SUBE con el estrés real; ORDEN lo baja, ENTROPÍA lo sube.
        self.x += (self._estres - kappa * self.b + kappa * self.w) * dt
        self.x = max(p["x_min"] - 0.1, min(p["x_max"] + 0.1, self.x))
        # --- H = REGULACIÓN EMERGENTE ponderada por A_sys-env (la variable UNIFICADORA, O-N2.1) ---
        #   tensión: ICR↔IRDE vivas y BALANCEADAS (ni colapso de rama ni igualdad fija).
        #   viable: A>0 (no colapsado por desacople).  tendencia: A se MANTIENE/mejora (sin setpoint, sólo dA/dt).
        #   "¿la competencia ICR↔IRDE está SOSTENIENDO el acoplamiento del organismo con su entorno?"
        tension = 1.0 - abs(self.b - self.w) / (self.b + self.w + 1e-6)
        # ── EL PISO DE VIABILIDAD DEL ACOPLE DEJA DE SER UN NÚMERO SUELTO (8-ago-2026) ──
        # ERA `A_min = 0.1`, y MEDIDO no decidía nada: sobre 157.421 pasos (108 CSV, 3–8 ago)
        # `A_sys_env` nunca bajó de 0,1638 —en los tres últimos días, de 0,1885—, así que
        # A/A_min ≥ 1,638 SIEMPRE y `viable` valió exactamente 1,0 en el 100,00 % de los pasos.
        # Un factor que vale 1 en 157.421 de 157.421 pasos no es un factor: es una constante
        # escondida dentro de un producto de cuatro.
        #
        # NO SE RELATIVIZA CONTRA LA PROPIA HISTORIA, y eso no es negociable: `viable` es una
        # CONDICIÓN DE VIDA, no una percepción (advertencia 2 de escala.py; VST_IndiceHomeostasis
        # cita este mismo piso como el ejemplo de lo que «debe seguir siendo absoluto aunque el
        # organismo se acostumbre a estar mal»). Dividir A por su media móvil haría que un
        # organismo crónicamente desacoplado leyera «acople normal» justo mientras pierde el nicho.
        #
        # LA SALIDA ES ESTRUCTURAL, NO OTRA CIFRA A OJO: el piso pasa a ser `banda`, que este mismo
        # organelo YA declara y usa en las MISMAS UNIDADES que A (estab_A = 1 − sd(A)/banda, o sea
        # `banda` es la anchura de fluctuación de A que se considera tolerable). Lo que se afirma es
        # una RELACIÓN, no un número: un acoplamiento más pequeño que la tolerancia que el propio
        # organelo concede a sus fluctuaciones ya no es un agarre. Si mañana se mide `banda`, el
        # piso se mueve con ella: donde había dos números que calibrar por separado —y que podían
        # contradecirse sin que nadie lo notara— queda uno.
        #
        # QUÉ SE MUEVE: el piso 0,10 → 0,15 y `viable` SIGUE valiendo 1 en el 100 % del corpus —que
        # es lo que debe pasar con un piso de muerte en un organismo que sigue vivo: se mide en que
        # no se cruza—. Lo que cambia es el margen al colapso sobre el peor acople jamás registrado:
        # de 63,8 % a 9,2 %. Un número menos y una relación más. REPLAYADO fila por fila sobre el
        # corpus entero (DT=0,1 · tempo de Kleiber 2,1226): esta H no se mueve ni un dígito —0,00 %
        # de pasos distintos, |ΔH| máximo 0,000000—. Cambia la calibración, no la conducta.
        # (`analisis/deg_constantes_degeneradas.py`).
        viable = max(0.0, min(1.0, A / max(1e-6, p["A_piso"])))
        tendencia = max(0.0, min(1.0, 1.0 + self._dA_ema / p["escala_dA"]))
        # BANDA VIABLE EMERGENTE (montañista, O-N9.14 "A_sys-env EN RANGO ESTABLE"): el centro y la
        # dispersión de A salen de SU PROPIA historia — nadie los fija. estab_A alta = A se SOSTIENE
        # dentro de su banda, sea cual sea el nivel (0.65 o 0.95: ambos sanos si son estables).
        if self._A_ema is None:
            self._A_ema = A
        desv = A - self._A_ema
        self._A_ema += p["ema_A"] * desv
        self._varA_ema = (1.0 - p["ema_A"]) * self._varA_ema + p["ema_A"] * desv * desv
        self.estab_A = max(0.0, 1.0 - (self._varA_ema ** 0.5) / max(1e-6, p["banda"]))
        # H = ¿la competencia ICR↔IRDE SOSTIENE A_sys-env estable dentro de su banda viable? (viabilidad a través del cambio)
        self.H = max(0.0, min(1.0, tension * viable * tendencia * self.estab_A))
        # PATOLOGÍAS (diagnóstico O-N2.1): AUTOENCIERRO = el orden SUBE mientras A CAE (se ordena perdiendo nicho);
        # ANESTESIA = la entropía CAE porque el sistema deja de registrar perturbación (sin estrés = falsa calma).
        self.autoencierro = max(0.0, (self.b - self._b_prev) / max(dt, 1e-6)) * desacople
        self.anestesia = max(0.0, (self._w_prev - self.w) / max(dt, 1e-6)) * max(0.0, 1.0 - min(1.0, self._estres / 0.05))
        self._A_prev = A; self._b_prev = self.b; self._w_prev = self.w

    def secretar(self, milieu: "Milieu") -> None:
        milieu.secretar(self.salida, self.H)
        milieu.secretar(self.variable, self.x)
        p = self.plast
        milieu.secretar(f"{self.variable}_en_rango", p["x_min"] <= self.x <= p["x_max"])
        milieu.secretar(f"{self.variable}_orden", self.b)        # margarita ORDEN (ICR-like)
        milieu.secretar(f"{self.variable}_entropia", self.w)     # margarita ENTROPÍA (IRDE-like)
        milieu.secretar(f"{self.variable}_acopl_tendencia", self._dA_ema)   # dA_sys-env/dt suavizada (O-N2.1)
        milieu.secretar(f"{self.variable}_autoencierro", self.autoencierro) # orden sube ∧ A cae
        milieu.secretar(f"{self.variable}_anestesia", self.anestesia)       # entropía cae sin perturbación
        milieu.secretar(f"{self.variable}_A_estabilidad", self.estab_A)     # A estable en su banda emergente (montañista)
        # LO QUE EMPUJA, VISIBLE. El estrés es la entrada que mueve toda esta regulación y no
        # se publicaba: se veía a las margaritas reaccionar y no A QUÉ. Sin estas tres columnas
        # la pregunta «¿por qué está trabajando?» sólo se podía contestar reconstruyendo el
        # cálculo a mano desde e_R y A_sys_env, que es como se descubrió el problema del cero.
        milieu.secretar(f"{self.variable}_estres", self._estres)              # lo que empuja el desorden
        milieu.secretar(f"{self.variable}_perturb_habitual", self._eR_habitual)  # el error que este organismo sostiene
        milieu.secretar(f"{self.variable}_perturb_exceso", self._exceso)      # lo que NO está logrando cerrar
        # ── EL TRABAJO DE REGULAR (6-ago-2026) ──────────────────────────────────────────
        # Sostener el equilibrio CUESTA, y no puede costar lo mismo estar en calma que
        # defenderse de una perturbación: con frío se queman más calorías para sostener la
        # temperatura. Hasta hoy este organelo cobraba `costo_base` fijo, proporcional a la
        # masa (Kleiber) y ciego al esfuerzo — flotar en calma y pelear costaban igual.
        #
        # EL ESFUERZO ES EL CONTRA-EMPUJE APLICADO. En este Daisyworld `x` se mueve por
        # `(estrés − κ·b + κ·w)`: el estrés empuja, la margarita ORDEN empuja en contra y la
        # de ENTROPÍA a favor. En calma las dos se equilibran y el contra-empuje neto es
        # ~0; bajo perturbación una desplaza a la otra y el neto crece. Por eso el trabajo
        # es |b − w|: cero cuando nada hay que corregir, máximo cuando una rama tiene que
        # sostener sola el equilibrio.
        #
        # VERIFICADO, y hacía falta verificarlo (`analisis/probar_esfuerzo.py`). En calma
        # REAL —e_R=0, A=1— las dos margaritas no se apagan: se estacionan en b = w =
        # 0,35000 exactas, y el neto queda en 0,00000 indefinidamente. Y el esfuerzo crece
        # monótono con lo que empuja el entorno: 0,000 → 0,083 → 0,186 → 0,357 → 0,693 para
        # e_R = 0, 5, 10, 20, 40. Es lo que pide la norma: el costo escala con la
        # perturbación que se está corrigiendo.
        #
        # SE PROBÓ LA ALTERNATIVA `b + w` («las dos ramas efectoras trabajando, aunque se
        # cancelen: tiritar y sudar a la vez quema más, no cero») Y ESTÁ MAL EN ESTE MODELO:
        # da 0,700 en calma absoluta y BAJA al subir la perturbación (0,700 → 0,634 → 0,501
        # → 0,358), porque aquí la coexistencia equilibrada ES el estado de reposo, no un
        # ciclo fútil. Habría cobrado más por descansar que por defenderse.
        #
        # OJO CON LEER EL PISO COMO DEFECTO. Medido en Abraxas el 6-ago, el esfuerzo nunca
        # baja de 0,1235 (p05) y `x_interna_entropia` redondea a 0,0000 el 66,72 % de los
        # pasos. No es que la fórmula no sepa llegar a cero: es que este organismo NO TIENE
        # UN SOLO PASO DE CALMA — su estrés reconstruido nunca baja de 0,0960, porque `e_R`
        # vive en ~10. Y la margarita ENTROPÍA está casi extinta porque los picos la
        # aplastan (a e_R=20 el estacionario es w = 0,00082) y su crecimiento es
        # multiplicativo, o sea que volver de ahí toma muchísimo más de lo que tarda en
        # caer. El organelo está informando bien un organismo crónicamente estresado.
        #
        # SIN NINGUNA CONSTANTE NUEVA, y por una cancelación real: el contra-empuje vale
        # κ·|b − w| y el máximo posible es κ·1, así que el factor es |b − w| y κ se va. El
        # costo activo queda en [0, costo_base]: en calma no se paga nada EXTRA (el basal se
        # sigue cobrando aparte, como debe ser: el metabolismo basal existe en reposo), y en
        # defensa máxima se paga una vez el costo del órgano.
        #
        # ABSOLUTO, NO RELATIVO A LA PROPIA HISTORIA. Se probó escalarlo con `rel_contra`
        # contra el esfuerzo habitual y está mal: tiritar quema calorías se tirite seguido o
        # no. Un organismo que se defiende siempre leería «esfuerzo normal» y dejaría de
        # pagarlo — la advertencia 2 de escala.py aplicada al gasto.
        #
        # AQUÍ SÓLO SE PUBLICA; EL COBRO ESTÁ EN `VST_Metabolismo` y se puso DESPUÉS de
        # medir la escala, que era la única forma de no inventar otro número: `costo_base`
        # crudo daba 0,2714 contra un `met_gasto` de 0,0105 —veintiséis veces el metabolismo
        # entero—, así que la referencia pasó a ser `basal`, lo que cuesta meramente existir.
        # Ya cobrando, medido sobre 2.461 pasos: el costo homeostático es el 7,5 % del gasto
        # (mediana), p95 22,8 %, máximo 46,4 %. Poner precio antes de medir la escala es
        # exactamente cómo se llegó a los 168 números sin origen.
        milieu.secretar(f"{self.variable}_esfuerzo", abs(self.b - self.w))
        milieu.secretar(f"{self.variable}_costo_activo", self.costo_base * abs(self.b - self.w))


# ==============================================================================
# CÉLULA HOMEOSTÁTICA — homeostasis aplicada a la propia célula
# ==============================================================================
def transcribir_celula_homeostatica() -> Organismo:
    """El organismo completo (B5+B7+B8) MÁS homeostasis a nivel célula. La H activa el
    componente de homeostasis del OI: la célula sube de protoorganismo hacia organismo,
    aunque le falta ME (memoria externalizada) — que solo aparece en el colectivo."""
    from VST_Bloque08_DinamicaEvolutiva import transcribir_organismo_completo
    o = transcribir_organismo_completo()
    # homeostato celular: regula un estado interno frente al error operativo
    o.expresar(OrganeloHomeostasis(variable="x_interna", x0=0.5, x_min=0.0, x_max=1.0,
                                   kp=0.5, perturb="e_R", perturb_escala=0.01))
    o.nombre = "celula_madre_homeostatica"
    return o


# ==============================================================================
# MULTICÉLULA — el organismo a la escala siguiente (homeostasis colectiva + S_shared)
# ==============================================================================
class Multicelula:
    """Un organismo MULTICELULAR: varias células que comparten un MEDIO EXTERNO (el
    "aire" = S_shared) sobre el cual opera una homeostasis COLECTIVA. Por universalidad
    estructural, es un Organismo a la escala siguiente: un homeostato sobre sus células,
    como la célula es un homeostato sobre sus organelos.

    MEMBRANA INTER-CELULAR: tras vivir, cada célula EXPORTA un resumen de su estado
    (LF, XE, acoplamiento, H) al aire compartido. El aire ES la memoria externalizada
    (ME = S_shared): por primera vez ME>0 en el OI.

    HOMEOSTASIS COLECTIVA: un OrganeloHomeostasis (el mismo código, otra escala) regula
    el acoplamiento colectivo (media de las células) → H colectivo.

    KLEIBER COLECTIVO: M_colectivo = Σ M_células → la colonia es más lenta y eficiente
    por unidad (el elefante).

    GUARDA: la cohesión aquí es ANDAMIAJE IMPUESTO. El organismo multicelular VOLUNTARIO
    (Ψ_alma sujeto-sujeto vs desalmamiento) espera el locus de altruismo de Boorman
    (PRE, reservado). Se reporta el OI colectivo MARCANDO esta condición.
    """

    # OI colectivo E018: componentes simetricos, sin pesos orientativos.

    def __init__(self, celulas: list[Organismo], M0: float = 1.0) -> None:
        self.celulas = celulas
        self.aire = Milieu()                          # medio compartido = S_shared
        self.complejidad = MedidorComplejidad(M0)     # Kleiber a escala colectiva
        # homeostato colectivo: regula el acoplamiento colectivo A hacia un rango viable
        self.homeostato = OrganeloHomeostasis(
            variable="A_colectivo", x0=0.7, x_min=0.3, x_max=1.0, kp=0.5,
            perturb="estres_colectivo", perturb_escala=0.2, salida="H_colectivo")
        self.t = 0.0
        self.M_colectivo = M0

    def vivir_un_paso(self, dt: float) -> None:
        # 1) cada célula vive su propio ciclo metabólico (es autónoma)
        for c in self.celulas:
            c.vivir_un_paso(dt)
        # 2) MEMBRANA: cada célula exporta su resumen al aire compartido (S_shared)
        for i, c in enumerate(self.celulas):
            self.aire.secretar(f"cel{i}_LF", c.milieu.leer("LF", 0.0))
            self.aire.secretar(f"cel{i}_XE", c.milieu.leer("XE", 0.0))
            self.aire.secretar(f"cel{i}_A", c.milieu.leer("A_sys_env", 0.0))
            self.aire.secretar(f"cel{i}_H", c.milieu.leer("H_homeostasis", 0.0))
        # 3) agregación colectiva (medias) + estrés colectivo (desvío del acoplamiento)
        n = len(self.celulas)
        A_col = sum(c.milieu.leer("A_sys_env", 0.0) for c in self.celulas) / n
        self.aire.secretar("estres_colectivo", max(0.0, 0.7 - A_col))  # cuánto baja del objetivo
        # 4) homeostato COLECTIVO sobre el aire (mismo organelo, otra escala)
        self.homeostato.percibir(self.aire)
        self.homeostato.metabolizar(dt, self._tempo_colectivo())
        self.homeostato.secretar(self.aire)
        self.t += dt

    def _tempo_colectivo(self) -> float:
        # Kleiber colectivo: M = Σ M_células
        self.M_colectivo = max(self.complejidad.M0,
                               sum(c.complejidad.M for c in self.celulas))
        return (self.M_colectivo / self.complejidad.M0) ** 0.25

    def salud_colectiva(self) -> dict[str, Any]:
        """OI colectivo (O-N9.14) sobre el medio compartido. ME>0 porque el aire ES la
        memoria externalizada (S_shared)."""
        n = len(self.celulas)
        H_col = self.aire.leer("H_colectivo", 0.0)
        LF_col = sum(c.milieu.leer("LF", 0.0) for c in self.celulas) / n
        XE_col = sum(c.milieu.leer("XE", 0.0) for c in self.celulas) / n
        # ME = riqueza de la memoria externalizada compartida (nº de aportes en el aire)
        aportes = len([k for k in self.aire.senales if k.startswith("cel")])
        ME = min(1.0, aportes / (n * 2.0))   # ~1 cuando las células comparten su estado
        componentes_oi = [H_col, ME, min(1.0, XE_col), min(1.0, LF_col)]
        OI = sum(componentes_oi) / max(1, len(componentes_oi))
        OI = max(0.0, min(1.0, OI))
        nivel = ("organismo pleno" if OI >= 0.7
                 else "protoorganismo" if OI >= 0.4 else "no organismal")
        s = self._tempo_colectivo()
        return {"H_col": H_col, "ME": ME, "XE_col": XE_col, "LF_col": LF_col,
                "OI": OI, "nivel_OI": nivel, "M_colectivo": self.M_colectivo,
                "tempo_s": s, "eficiencia_r": (self.M_colectivo / self.complejidad.M0) ** -0.25}

    def quien_soy(self) -> str:
        s = self.salud_colectiva()
        return (
            "=" * 80 + "\n"
            f"QUIÉN SOY — organismo MULTICELULAR ({len(self.celulas)} células)\n"
            + "=" * 80 + "\n"
            f"  Kleiber colectivo: M={s['M_colectivo']:.1f}  s(M)={s['tempo_s']:.2f} (más lento)  "
            f"r(M)={s['eficiencia_r']:.3f} (más eficiente/unidad)\n"
            f"  componentes OI:  H_col={s['H_col']:.3f}  ME(S_shared)={s['ME']:.3f}  "
            f"XE_col={s['XE_col']:.3f}  LF_col={s['LF_col']:.3f}\n"
            f"  → OI colectivo = {s['OI']:.3f} → {s['nivel_OI'].upper()}\n"
            f"  ⚠ cohesión = ANDAMIAJE IMPUESTO; organismo VOLUNTARIO pendiente del locus de "
            f"Boorman (PRE, reservado).\n"
            + "=" * 80
        )


# ==============================================================================
# DEMO / AUTOVERIFICACIÓN
# ==============================================================================
if __name__ == "__main__":
    # --- 1) Homeostasis aplicada a la propia célula: activa H del OI ---
    cel = transcribir_celula_homeostatica()
    for _ in range(2000):
        cel.vivir_un_paso(0.1)
    h = cel.organelos["homeostasis::x_interna"]
    s = cel.salud()
    print("=" * 80)
    print("HOMEOSTASIS A NIVEL CÉLULA (B5+B7+B8 + homeostasis):")
    print(f"  x_interna={h.x:.3f} (rango [0,1])  H={h.H:.3f}")
    print(f"  → OI={s['OI']:.3f} → {s['nivel_OI'].upper()}  (H activo; ME aún 0: falta el colectivo)")

    # --- 2) Expansión multicelular POR DEFINICIÓN: misma homeostasis, otra escala ---
    print()
    colonia = Multicelula([transcribir_celula_homeostatica() for _ in range(3)])
    for _ in range(2000):
        colonia.vivir_un_paso(0.1)
    print(colonia.quien_soy())
    sc = colonia.salud_colectiva()
    print(f"\n  COMPARACIÓN DE ESCALA:")
    print(f"    célula:       OI={s['OI']:.3f} ({s['nivel_OI']})   — ME=0")
    print(f"    multicelular: OI={sc['OI']:.3f} ({sc['nivel_OI']}) — ME={sc['ME']:.2f} (S_shared activa)")
    print(f"    La homeostasis es la MISMA (scale-invariant); el salto a organismo lo da ME (S_shared).")
