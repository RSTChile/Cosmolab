# diagnostico_acoplamiento.py
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def cargar_wav_seguro(nombre):
    """Carga WAV manejando diferentes formatos de retorno."""
    resultado = wav.read(nombre)
    
    # scipy.io.wavfile.read puede retornar 2 o 3 valores
    if len(resultado) == 2:
        sr, data = resultado
    elif len(resultado) == 3:
        sr, data, _ = resultado
    else:
        raise ValueError(f"No se pudo leer {nombre}")
    
    # Convertir a float
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    
    return sr, data

def a_mono(audio):
    """Convierte estéreo a mono promediando canales."""
    if audio is None:
        return None
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        return audio.mean(axis=1)
    return audio

def acoplamiento_simple(audio, fs, ventana_ms=30, hop_ms=10):
    """Calcula acoplamiento como proporción energía voz / energía total."""
    ventana = int(fs * ventana_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    
    if len(audio) < ventana:
        return np.array([0.5])
    
    n_ventanas = (len(audio) - ventana) // hop + 1
    acoplamientos = []
    
    for i in range(n_ventanas):
        inicio = i * hop
        fragmento = audio[inicio:inicio + ventana]
        ventana_hann = np.hanning(len(fragmento))
        fragmento = fragmento * ventana_hann
        
        fft = np.fft.rfft(fragmento)
        potencia = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(fragmento), 1/fs)
        
        # Energía en banda de viento (0-200 Hz)
        mask_viento = freqs <= 200
        energia_viento = np.sum(potencia[mask_viento]) if np.any(mask_viento) else 0
        
        # Energía en banda de voz (100-4000 Hz)
        mask_voz = (freqs >= 100) & (freqs <= 4000)
        energia_voz = np.sum(potencia[mask_voz]) if np.any(mask_voz) else 0
        
        energia_total = energia_voz + energia_viento + 0.01
        acop = energia_voz / energia_total
        acoplamientos.append(acop)
    
    return np.array(acoplamientos)

# Cargar archivos
print("Cargando archivos...")
try:
    sr_v, voz = cargar_wav_seguro('Voz_Estudio.wav')
    print(f"  Voz_Estudio: {len(voz)/sr_v:.3f}s, {sr_v}Hz, {voz.dtype}")
except Exception as e:
    print(f"  Error con Voz_Estudio: {e}")
    sr_v, voz = None, None

try:
    sr_v1, voz_viento1 = cargar_wav_seguro('Voz+Viento_1.wav')
    print(f"  Voz+Viento_1: {len(voz_viento1)/sr_v1:.3f}s, {sr_v1}Hz")
except Exception as e:
    print(f"  Error con Voz+Viento_1: {e}")
    sr_v1, voz_viento1 = None, None

try:
    sr_v2, voz_viento2 = cargar_wav_seguro('Voz+Viento_2.wav')
    print(f"  Voz+Viento_2: {len(voz_viento2)/sr_v2:.3f}s, {sr_v2}Hz")
except Exception as e:
    print(f"  Error con Voz+Viento_2: {e}")
    sr_v2, voz_viento2 = None, None

try:
    sr_w, viento = cargar_wav_seguro('Viento.wav')
    print(f"  Viento: {len(viento)/sr_w:.3f}s, {sr_w}Hz")
except Exception as e:
    print(f"  Error con Viento: {e}")
    sr_w, viento = None, None

# Verificar que todos tienen la misma frecuencia
frecuencias = []
for sr in [sr_v, sr_v1, sr_v2, sr_w]:
    if sr is not None:
        frecuencias.append(sr)
frecuencias = set(frecuencias)

if len(frecuencias) != 1:
    print(f"\nAdvertencia: Frecuencias diferentes: {frecuencias}")
    # Usar la primera no nula como referencia
    fs = next(sr for sr in [sr_v, sr_v1, sr_v2, sr_w] if sr is not None)
    print(f"  Usando fs = {fs} Hz")
else:
    fs = sr_v

# Convertir a mono y normalizar
print("\nProcesando...")
voz = a_mono(voz) if voz is not None else None
voz_viento1 = a_mono(voz_viento1) if voz_viento1 is not None else None
voz_viento2 = a_mono(voz_viento2) if voz_viento2 is not None else None
viento = a_mono(viento) if viento is not None else None

# Recortar todas a la misma duración (la más corta)
arrays = [voz, voz_viento1, voz_viento2, viento]
nombres = ["Voz Estudio", "Voz+Viento_1", "Voz+Viento_2", "Viento"]
datos_validos = [(arr, nom) for arr, nom in zip(arrays, nombres) if arr is not None]

if len(datos_validos) > 0:
    min_len = min(len(arr) for arr, _ in datos_validos)
    print(f"Recortando a {min_len} muestras")
    
    idx = 0
    if voz is not None:
        voz = voz[:min_len]
    if voz_viento1 is not None:
        voz_viento1 = voz_viento1[:min_len]
    if voz_viento2 is not None:
        voz_viento2 = voz_viento2[:min_len]
    if viento is not None:
        viento = viento[:min_len]

# Calcular acoplamiento
print("Calculando acoplamiento...")
acop_voz = acoplamiento_simple(voz, fs) if voz is not None else None
acop_v1 = acoplamiento_simple(voz_viento1, fs) if voz_viento1 is not None else None
acop_v2 = acoplamiento_simple(voz_viento2, fs) if voz_viento2 is not None else None
acop_viento = acoplamiento_simple(viento, fs) if viento is not None else None

# Graficar
plt.figure(figsize=(14, 7))

t_max = 5  # mostrar primeros 5 segundos

if acop_voz is not None:
    t_voz = np.arange(len(acop_voz)) * 10 / 1000
    mask = t_voz < t_max
    plt.plot(t_voz[mask], acop_voz[mask], label='Voz Estudio', linewidth=1)

if acop_v1 is not None:
    t_v1 = np.arange(len(acop_v1)) * 10 / 1000
    mask = t_v1 < t_max
    plt.plot(t_v1[mask], acop_v1[mask], label='Voz + Viento 1', linewidth=1, alpha=0.7)

if acop_v2 is not None:
    t_v2 = np.arange(len(acop_v2)) * 10 / 1000
    mask = t_v2 < t_max
    plt.plot(t_v2[mask], acop_v2[mask], label='Voz + Viento 2', linewidth=1, alpha=0.7)

if acop_viento is not None:
    t_w = np.arange(len(acop_viento)) * 10 / 1000
    mask = t_w < t_max
    plt.plot(t_w[mask], acop_viento[mask], label='Viento Puro', linewidth=1, alpha=0.7)

plt.axhline(y=0.3, color='r', linestyle='--', label='Umbral colapso (0.3)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Acoplamiento (energía voz / energía total)')
plt.title('Diagnóstico de Acoplamiento - Primeros 5 segundos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diagnostico_acoplamiento.png', dpi=150)
print("Gráfico guardado: diagnostico_acoplamiento.png")

# Estadísticas
print("\n" + "=" * 50)
print("ESTADÍSTICAS DE ACOPLAMIENTO")
print("=" * 50)

def stats(nombre, acop):
    if acop is not None and len(acop) > 0:
        print(f"{nombre:20} media={np.mean(acop):.3f}  min={np.min(acop):.3f}  max={np.max(acop):.3f}")
    else:
        print(f"{nombre:20} ERROR - no se pudo calcular")

stats("Voz Estudio", acop_voz)
stats("Voz + Viento 1", acop_v1)
stats("Voz + Viento 2", acop_v2)
stats("Viento Puro", acop_viento)

# Verificar si hay voz (acoplamiento alto)
if acop_voz is not None and len(acop_voz) > 0:
    if np.mean(acop_voz) > 0.6:
        print("\n✅ Voz Estudio detectada correctamente (acoplamiento alto)")
    else:
        print(f"\n❌ Voz Estudio NO detectada (acoplamiento medio={np.mean(acop_voz):.3f}). Revisar cálculo.")
else:
    print("\n❌ No se pudo calcular acoplamiento para Voz Estudio")

print("\n¡Diagnóstico completado!")