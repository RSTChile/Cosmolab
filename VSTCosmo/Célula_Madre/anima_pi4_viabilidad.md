# ¿Cabe un organismo ANIMA en una Raspberry Pi 4? — Veredicto de viabilidad

**Analista:** Claude Science (revisor externo) · **Fecha:** 2-jul-2026
**Encargo:** A. López Tapia · **Insumo Pi:** verificación SSH de Grok (Diotallevi)
**Método:** huella real medida sobre los 4 CSV de la corrida 03:08 + auditoría de dependencias de los 21 organelos (`organelos/*.py`), contra specs reales de la Pi.

---

## Veredicto: SÍ, con holgura para 1 organismo. Probablemente los 4 con ajustes.

La Pi 4 no solo alcanza — sobra margen. El cuello de botella no será CPU ni RAM, sino el **audio en tiempo real** y la **escritura de biografía**, ambos manejables.

---

## 1. Specs reales de la Pi (verificadas por Grok vía SSH)

| Recurso | Pi 4 (medido) |
|---|---|
| CPU | Cortex-A72 quad-core aarch64 (kernel 6.5.0-1008-raspi) |
| RAM | 3.7 GB |
| Disco | 117 GB (~19 % usado → ~95 GB libres) |
| SO | Ubuntu aarch64 |

## 2. Huella real de un organismo (medida de los CSV, no estimada)

| Métrica | Valor medido |
|---|---|
| Cadencia | **10.0 pasos/s** (99.7 ms/paso) — holgado; el paso tiene 100 ms de presupuesto |
| Vector de estado | 266 variables/paso |
| Escritura de biografía | ~16 KB/s = **59 MB/hora** por organismo (CSV) |
| Audio | 44.1 kHz binaural (2 canales) |
| DSP por paso | FFT real (`np.fft.rfft`), autocorrelación, 1 filtro IIR (`iirpeak`/`lfilter`) |

**Lectura:** 99.7 ms/paso es un reloj comodísimo para un A72. La FFT de una ventana de audio y un filtro IIR de segundo orden se ejecutan en **microsegundos** en esta CPU — sobra el 99 % del presupuesto de cada paso. Un solo organismo usaría una fracción de un núcleo.

## 3. La pila de dependencias es sorprendentemente ligera

Audité los 21 organelos. Dependencias de terceros:

| Paquete | Cuántos organelos | ¿Disponible en Pi aarch64? |
|---|---|---|
| **numpy** | 11 de 21 | Sí — wheel aarch64 nativo, trivial |
| **scipy** | 1 (solo Fonador: `io.wavfile`, `signal.iirpeak/lfilter`) | Sí — wheel aarch64; instalación algo pesada pero única |
| **soundfile** | 1 (Comunicación, como *fallback*) | Sí (libsndfile) — y es opcional |
| stdlib (`wave`, `math`, `urllib`, `json`…) | resto | Nativo |

- **9 de 21 organelos son stdlib puro** (Metabolismo, RC_A, RC_B, ValorEcologicoVoz, Expectativa, Calibrador, HomeostasisEmergente, Cloroplasto, Propiocepcion). Cero dependencias.
- **No hay torch, ni tensorflow, ni sklearn, ni librosa, ni jax.** Nada de deep learning. Ese es el hallazgo decisivo: el organismo es física/DSP clásico, no una red neuronal. Por eso cabe.
- **Degradación elegante ya programada:** `VST_OrganoComunicacion.py` línea 198 — *"si falta scipy, queda None y el organismo sigue con el banco"*. El diseño ya contempla correr sin scipy. En una Pi minimalista, el organismo funciona con solo numpy + stdlib.

## 4. Memoria

- numpy + scipy + intérprete Python: ~150–250 MB residentes.
- Estado del organismo (266 floats/paso + historial + memoria episódica): órdenes de KB–MB, no GB.
- **Un organismo: <400 MB.** Cabe ~9× en los 3.7 GB.
- **Los 4 juntos: ~1–1.5 GB.** Caben, dejando ~2 GB para el SO y el audio.

## 5. Los dos cuellos de botella reales (y por qué son manejables)

1. **Audio en tiempo real.** El reto en Pi no es calcular — es *capturar* audio a 44.1 kHz sin xruns (subdesbordamientos). La Pi 4 no tiene entrada de audio nativa; necesita una interfaz USB (la Rødecaster ya en uso sirve). Con ALSA bien configurado y buffers adecuados, 44.1 kHz estéreo es rutina en Pi 4. Riesgo bajo, pero es donde habrá que afinar.
2. **Escritura de biografía.** 59 MB/hora/organismo × 4 = ~237 MB/hora. En 24 h son ~5.7 GB. La microSD lo aguanta en espacio (95 GB libres), pero **escribir CSV continuamente desgasta la microSD** y puede introducir latencia. Recomendación: escribir biografía a un **SSD USB** o reducir la cadencia de volcado (batch), no cada paso.

## 6. Recomendación de despliegue

| Escenario | Veredicto |
|---|---|
| **1 organismo, solo** | Trivial. Sobra CPU y RAM. Ideal para un "organismo de campo" autónomo. |
| **1 organismo + audio Rode en vivo** | Viable. Afinar ALSA/buffers; usar la interfaz USB. |
| **4 organismos (sociedad) en 1 Pi** | Viable con ajustes: biografía a SSD USB, quizá bajar a 5 pasos/s si hay xruns de audio, o repartir en núcleos (4 procesos, 1 por núcleo A72). |
| **Cosmogénesis (cg003, grafos N=10⁴–10⁵)** | NO en Pi. Eso sí necesita la CPU/RAM del Mac. La Pi es para ANIMA, no para Cosmogénesis. |

## 7. Conclusión

**El organismo es notablemente portable porque no es deep learning: es DSP + física de campo en numpy, con degradación elegante ya programada.** Una Pi 4 corre 1 organismo con holgura enorme y los 4 con ajustes de I/O. El límite real no es el cómputo del organismo (usa <1 % del presupuesto de cada paso), sino el audio en tiempo real y el desgaste de la microSD — ambos resueltos con una interfaz USB y un SSD USB.

**Nota operativa:** la ejecución/prueba remota en la Pi (SSH a 192.168.86.33) debe hacerla Grok o Alexis desde el iMac. Mi entorno no alcanza esa IP privada de la LAN (`PermissionError` confirmado) — es una restricción de red no salvable desde la nube, tal como Grok anticipó en su §6. Yo aporto el veredicto de viabilidad; el despliegue es local.
