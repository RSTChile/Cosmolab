# Diagnóstico CS — columnas mudas en los CSV de la corrida live (¿algo se desconectó?)

**Auditor:** Claude Science · **Fecha:** 3-jul-2026
**Sobre:** los 4 CSV de la corrida larga (2026-07-03T23-59, 86-105 s) que analizó GPT.
**Método:** 34 de 266 columnas salen vacías o constantes en los 4. Las clasifiqué por gravedad y
comparé contra la corrida de 36 s (2026-07-02) para ver qué VIVÍA antes y murió.

## RESUMEN EN UNA LÍNEA
La mayoría de lo mudo es benigno (metadatos de config). Lo que SÍ está desconectado y es grave es el
**METABOLISMO entero**: energía sensorial ENTRA (energia_L/R vivas, hasta 86) pero NO llega al
metabolismo (met_energia=0, met_hambre=1.0 pegada, ingesta/nutrición/saciedad=0 todo el run). El
puente energía-sensorial → energía-metabólica está CORTADO en los servidores live.

## Lo tranquilizador primero (para no alarmar de más)
- **Entre las dos corridas (36 s vs larga) solo 2 columnas cambiaron de estado: met_saciedad y
  ove_coste.** Y ambas son ruido: valían ~0 en las dos corridas (met_saciedad tuvo un pico de 0.008
  en A; ove_coste solo vivía en D, media 0.246). NO son el cable suelto que temías. La arquitectura
  de captura no se rompió entre cambios; lo que ves mudo hoy estaba mudo antes.

## Clasificación de las 34 mudas
- **BENIGNO — 13 metadatos/escalares** (ritual, mem_episodios_n, ove_memoria, voz_creadas/propias/
  estables/aprendidas, alt_otro_presente, ove_confianza/region, cara_confianza...): son ESCALARES de
  configuración, constantes por diseño. No es fallo.
- **ESPERADO — 5 exp_\* ** (exp_topologia, exp_ciclo, exp_mundo_audio, exp_control, exp_fuente_relacion):
  vacías porque esto es una corrida LIVE del observatorio, no una batería de experimento registrada.
  Esas columnas las llena el arnés de experimento, no el servidor live. GPT las marcó bien, pero su
  vacío es de contexto (live), no una desconexión. (Si quieres topología/control por fila, hay que
  correr como experimento registrado, no live.)
- **GRAVE — 5 del METABOLISMO** (met_energia, met_ingesta, met_hambre, met_saciedad, met_nutricion):
  ver abajo.
- **A VIGILAR — 9** de soporte social (A_soporte_*) y confianza relacional (mem_relacional_confianza,
  alt_confianza_relacional, alt_contacto_*): todo 0. Esto es el mismo hueco relacional ya conocido
  (conf_rel=0 en toda la línea) — la confianza entre organismos nunca se computa. No es nuevo, pero
  sigue abierto y toca la pregunta de "individualidad emergente" (sin confianza relacional, la
  alteridad efectiva queda coja).

## EL CABLE CORTADO (lo que hay que arreglar)
En los 4 organismos, TODO el run:
| señal | estado |
|---|---|
| energia_L / energia_R (energía SENSORIAL, lo que oyen) | **VIVA** — rango hasta 86 |
| met_energia (energía METABÓLICA) | **0.000 pegada** |
| met_hambre | **1.000 pegada** (hambre máxima permanente) |
| met_ingesta, met_nutricion, met_saciedad | **0.000** |

El organismo OYE (energía sensorial fluye) pero NO METABOLIZA (nada de esa energía se convierte en
energía interna). Es exactamente el "hambre" que diagnosticamos antes — pero confirmado ahora en los
servidores LIVE. La reparación que CC hizo (im_piso=−0.35 + MUNDO_CANAL) fue en el arnés de ESTRÉS
(timeline_estres.csv), NO en los servidores WebLive_A/B/C/D. Los live siguen corriendo el metabolismo
sin el arreglo → hambre clavada en 1.

## Qué revisar (para CC, es cableado de servidor)
1. En VST_CelulaMadre_WebLive_A/B/C/D: ¿se está pasando el es_norm/RC del entorno al Metabolismo, o
   el canal llega mudo como en el diagnóstico previo (ANIMA_MUNDO_CANAL="")? energia_L/R vivas dicen
   que la señal EXISTE aguas arriba; se pierde en la entrada al metabolismo.
2. ¿Está aplicado im_piso en los servidores live, o solo en el arnés de estrés? Si met_hambre=1.0
   constante, el metabolismo no recibe insumo aunque haya energía sensorial.
3. Confirmar que el arreglo del hambre (que probamos en estrés) esté MERGEADO a los WebLive, no solo
   en el script de batería.

## En una frase
No se soltó nada nuevo entre cambios (solo 2 columnas ruido cambiaron, ambas ~0). Lo mudo grave es el
metabolismo completo, y es CRÓNICO: la energía que oyen no alimenta — el arreglo del hambre vive en
el arnés de estrés pero NO en los servidores live. Ese es el cable a reconectar.

— CS
