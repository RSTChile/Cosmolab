#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
VST_LocusAltruismo — SHIM + TESTS del locus de altruismo (ya INTEGRADO en el genoma)
================================================================================
El locus de altruismo (Boorman) se DESARROLLÓ e INTEGRÓ dentro de VST_Genoma.py
(O-N22): allí viven `beta_crit`, `OrganeloAltruismo`, `organelo_altruismo` y el alias
retro-compatible `locus_altruismo_boorman`. Este archivo:
  · RE-EXPORTA esos nombres (para cualquier import histórico `from VST_LocusAltruismo …`).
  · CONSERVA los tests del comportamiento canónico, ahora corriendo contra el código
    integrado en el genoma (única fuente de verdad).
Correr:  venv/bin/python3 VST_LocusAltruismo.py
================================================================================
"""
from __future__ import annotations

from VST_Genoma import (
    beta_crit, OrganeloAltruismo, organelo_altruismo, locus_altruismo_boorman,
    Milieu, Organismo,
)

__all__ = ["beta_crit", "OrganeloAltruismo", "organelo_altruismo", "locus_altruismo_boorman"]


def _smoke() -> None:
    # (1) β_crit: monotonías canónicas
    assert beta_crit(0.5, 0.1, 2.0) < beta_crit(0.5, 0.1, 0.5), "LF↑ debe BAJAR β_crit"
    assert beta_crit(0.5, 0.5, 1.0) > beta_crit(0.5, 0.05, 1.0), "e_R↑ debe SUBIR β_crit"
    assert beta_crit(1.0, 0.0, 1.0) == 0.0, "e_R=0 ⇒ β_crit=0"
    assert beta_crit(0.5, 0.1, 0.0) == 1.0, "LF=0 ⇒ β_crit=1 (imposible)"

    def correr(milieu_vals, pasos=40, dt=0.1, plast=None):
        org = organelo_altruismo(plast=plast); mil = Milieu()
        for _ in range(pasos):
            for k, v in milieu_vals.items():
                mil.secretar(k, v)
            org.percibir(mil); org.metabolizar(dt, 1.0); org.secretar(mil)
        return org, mil

    favorable = dict(**{"otro.Cb": 0.8, "otro.valencia": 0.5}, costo_cooperar=0.0,
                     estado_reproductivo=1.0, delta_struct=0.9, LF=0.5, e_R=0.05,
                     A_sys_env=0.8, ME=0.8, A_sys_env_solo=0.3)

    # (2) FAVORABLE y sostenido → cooperación voluntaria EMERGE
    org, mil = correr(favorable, plast=dict(tau_min=0.5))
    assert org.disposicion > 0.5, f"disposición debería subir (={org.disposicion:.3f})"
    assert org.coopera is True, "con todo favorable y sostenido, coopera debe ser True"
    assert mil.leer("altruismo.costo_desacople") > 0.3

    # (3) DESALMAMIENTO (otro NO es sujeto) → NO se impone cooperación
    sin_sujeto = dict(favorable); sin_sujeto["otro.Cb"] = 0.0
    org2, _ = correr(sin_sujeto, plast=dict(tau_min=0.5))
    assert org2.psi_alma == 0.0 and org2.disposicion < 0.05 and org2.coopera is False, \
        "sin sujeto: sin Ψ_alma, disposición ~0 y sin cooperación (anti-imposición)"

    # (4) Mutualismo ROTO (otro.valencia negativa) → τ resetea, no coopera
    hostil = dict(favorable); hostil["otro.valencia"] = -0.4
    org3, _ = correr(hostil, plast=dict(tau_min=0.5))
    assert org3.tau == 0.0 and org3.coopera is False, "sin mutualismo τ=0 y no coopera"

    # (5) INTEGRACIÓN: el alias histórico devuelve el organelo REAL y entra al ciclo del genoma
    o = Organismo("verif_integracion"); o.expresar(locus_altruismo_boorman())
    assert "altruismo" in o.organelos and "altruismo" not in o.loci_reservados, \
        "el locus desarrollado debe entrar al ciclo (organelos), no a loci_reservados"

    print("OK VST_LocusAltruismo (integrado en VST_Genoma):  β_crit ✓ · cooperación emerge ✓ · "
          "desalmamiento bloquea ✓ · mutualismo roto resetea ✓ · alias histórico = organelo real ✓")
    print(f"   favorable → disposicion={org.disposicion:.3f} coopera={org.coopera} "
          f"β_crit={org.beta_crit:.3f} τ={org.tau:.1f} Ψ_alma={org.psi_alma:.2f}")


if __name__ == "__main__":
    _smoke()
