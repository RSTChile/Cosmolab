# 15 — Cronología de la masa (canónica, Alexis 2026-07-22)

## Instrucción

**No puede haber masa** hasta:

1. **Después** del Higgs / ruptura del medio de orden (no en fase simétrica caliente), **y**
2. **Después** de la formación del **primer átomo**, **y**
3. **Solo cuando** la **gravedad actúa sobre el hidrógeno** y **este aumenta su densidad**.

Cualquier `Rm`, `m = y·factor·Σρ`, “masa térmica” o “jerarquía k1/k3” medidos **antes** de esa época **no es masa** en el sentido de la Teoría. Son, como mucho, lecturas de orden / geometría / acople prematuro — y **no** deben venderse como claim de masa.

## Épocas y observables admitidos

| Época | Relato | Qué SÍ medir | Qué NO medir / NO reclamar |
|-------|--------|--------------|----------------------------|
| **E0** pre-ruptura | T alta, medio simétrico, “derretido” | simetría, T(a), ρ∝a⁻³, estiramiento ∇_phys, ausencia de VEV | masa, bosón, arrastre inercial |
| **E1** ruptura / VEV | el medio “congela” (hielo) | aparición de VEV, muros, orden; proxy de excitación del vacío **sin** llamarlo masa | masa de partículas / dominios |
| **E2** post-Higgs pre-átomo | mediadores, carga, confín, etc. | lecturas de **relación** (carga, apantallamiento, topología) | masa |
| **E3** primer átomo | estructura H (o análogo) | existencia / estabilidad del átomo relacional | masa gravitatoria aún no |
| **E4** gravedad + H densificado | gravedad actúa; densidad de H sube | **ahí** nace el claim de **masa** (inercial/gravitatoria del relato) | — |

## Consecuencia para el trabajo 2026-07-21/22

- `Higgs_TEST_REAL_v3/v4` y `suite_crono_higgs` que juzgaban `sep(Rm, NULL)` como “señal de masa”: **reclasificar**. Como mucho: *herencia de orden en el tejido post-VEV*, no masa.
- Veredicto útil de la suite: **cronología de orden** y **prohibición de efectos de “masa” precoz**; el fallo `early_mass_fail` se relee como: el instrumento **no debió** exponer un observable de masa tan temprano.
- Programa correcto: cerrar E0→E1 sin masa → E2 sin masa → E3 átomo → E4 masa solo con gravedad+densidad H.

## Suite ejecutada 2026-07-22 (`suite_epocas_masa`)

Veredicto: **`MASS_E4_OK_BUT_LEAK_INSTRUMENT`**

- **`mass_obs` (legítima):** 0 en E0–E3; solo E4 con gravedad; OFF gravedad → 0 (6/6 seeds).
- **`leak_sep` (fórmula v3):** fuga sistemática en E0–E3 → **no es masa**; instrumento ilegal para ese nombre.
- Artefactos: `codigo/suite_epocas_masa/`, `results/suite_epocas_masa/RESUMEN_SUITE_EPOCAS_MASA.md`.

## Suite v2 endurecida 2026-07-22 (`suite_epocas_masa_v2`)

Veredicto: **`E3_OK_E4_CAUSAL_WEAK`**

- **E3 estricto** (núcleo+halo+cohesión+persistencia): rate **1.00**.
- **Masa:** solo `grav_mode=real`; OFF/SHUFFLE/INVERT → mass=0; pre-E4 mass=0.
- **Hueco abierto:** densificación REAL ≈ densificación SHUFFLE → el clumping genérico no prueba gravedad-sobre-H; falta dinámica atómica (N-body / enlaces entre centroides) donde REAL gane al null de pozos barajados.
- Artefactos: `results/suite_epocas_masa_v2/RESUMEN_SUITE_EPOCAS_MASA_v2.md`.

## Suite v3 N-body atómico 2026-07-22 (`suite_epocas_masa_v3`)

Veredicto: **`E3_OK_E4_PARTIAL_bind_sep_weak`**

- Gravedad **entre centroides** de átomos H; SHUFFLE = permutar **quién es fuente de atracción**.
- mass OFF/SHUFFLE/INVERT = **0** (nulls limpios); mass solo REAL.
- Enlace causal REAL≻SHUFFLE: rate **0.30**, media bindR/S **1.11** (umbral 1.25) — progreso vs v2, aún no robusto al 55%+.
- Artefactos: `results/suite_epocas_masa_v3/RESUMEN_SUITE_EPOCAS_MASA_v3.md`.

## Suite v4 linaje de fusión 2026-07-22/23 (`suite_epocas_masa_v4`)

Veredicto: **`E3_OK_E4_PARTIAL_lineage_weak`**

- Añade **FORCE_CUTOFF**, **pares mutuos** (E_mutual por ID ≥5 pasos) y **linaje** (co-membresía + fusiones).
- mass nulls **siguen limpios**; mass solo REAL.
- Stack causal pre-registrado (mutual_bind R/S + linaje): rate **0.30** (no PASS 0.55).
- **Hallazgo nuevo:** co-membresía REAL≻SHUFFLE (R/S≈**1.42**, rate lineage **0.90**). En cambio E_mutual media SHUFFLE ≥ REAL — la energía de pares mutuos **no** es el discriminante; el **linaje de quién se queda con quién** sí.
- No se redefinió el juez tras ver el dato para fabricar PASS.
- Artefactos: `results/suite_epocas_masa_v4/RESUMEN_SUITE_EPOCAS_MASA_v4.md`.

## Suite v5 juez por linaje 2026-07-23 (`suite_epocas_masa_v5`)

**Pre-registro:** `codigo/suite_epocas_masa/PROTOCOLO_V5_LINAJE_PREREGISTRO.md`  
Veredicto: **`E3_OK_E4_PARTIAL_lineage`**

- Juez **primario** = linaje; mutual_bind solo diagnóstico.
- rate e4_lineage_pass **0.40** (cuello: mass_obs=0 con E_mutual).
- Artefactos: `results/suite_epocas_masa_v5/`.

## Suite v6 mass∝linaje 2026-07-23 (`suite_epocas_masa_v6`) — ⚠ NO CIERRE

**Pre-registro:** `PROTOCOLO_V6_MASS_LINAJE_PREREGISTRO.md`  
**JSON reportó:** `E3_OK_E4_LINEAGE_CAUSAL_OK` rate 0.80  

**Auditoría director (mismo día):** **RETRACTADO como prueba de masa.**  
`mass_obs` se define con **las mismas variables** que `lineage_wins` (co_member, n_long_co).  
`e4_lineage_pass` ≡ `lineage_ok` semilla a semilla; el umbral mass≥0.3 nunca decide.  
El 0.40→0.80 es la tasa de lineage_ok ya conocida, no un hallazgo nuevo de masa.  
Además v6 se introdujo **después** de que v5 fallara el pre-registro del motor 1a7.  

**Documento de estatuto:** `Cosmogenesis/HALLAZGO_ABIERTO_etapa7_v6_masa_es_linaje_CS.md`  
**Claim residual honesto:** linaje REAL≻SHUFFLE (de v4/v5). **No claim:** masa independiente.

## Motor unificado 1→7 (2026-07-23) — ⚠ etapa 7 abierta

- Orquesta CS074-rcruz + TEST_RHO + épocas.  
- Etapas **1–6:** firmes (NULL muerden).  
- Etapa **7 / chain_pass=True con v6:** **no es cierre** (ver hallazgo abierto).  
- Primera corrida con v5: chain_pass **false** (ese resultado cuenta).

## Anti-Shannon

- No fijar 1/1836, 125 GeV, 246 GeV como jueces de éxito.
- Anclas 10¹⁵ K / 10⁻¹² s = relato/reporte, no perillas.
- La masa en E4 debe **emerger** del proceso gravedad↔densidad H, no de un `if época == masa`.
