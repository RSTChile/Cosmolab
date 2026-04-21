import numpy as np
import matplotlib.pyplot as plt
import csv

# =========================
# CONFIG
# =========================
DT = 0.01
SIM_TIME = 30.0
GRAVITY = 9.81

# Caja (Test 1)
CEILING = 0.5
FLOOR = 0.0
MASS = 1.2
MAX_THRUST = 20.0
HOVER_THRUST = MASS * GRAVITY # 11.772

# =========================
# DRON SIMPLE (1D vertical)
# =========================
class Drone:
    def __init__(self):
        self.z = 0.0
        self.vz = 0.0
        self.mass = MASS
        self.max_thrust = MAX_THRUST

    def step(self, thrust):
        thrust = np.clip(thrust, 0, self.max_thrust)
        az = (thrust / self.mass) - GRAVITY
        self.vz += az * DT
        self.z += self.vz * DT

        # colisiones con restitución parcial
        if self.z <= FLOOR:
            self.z = FLOOR
            self.vz = max(0, self.vz) * 0.1
        if self.z >= CEILING:
            self.z = CEILING
            self.vz = -abs(self.vz) * 0.3

# =========================
# SENSORES
# =========================
class Sensors:
    def __init__(self):
        self.prev_acc = 0.0
        self.prev_vz = 0.0

    def read(self, drone):
        # IMU
        acc = (drone.vz - self.prev_vz) / DT
        jerk = (acc - self.prev_acc) / DT
        self.prev_acc = acc
        self.prev_vz = drone.vz

        # ToF
        tof_down = max(0.0, drone.z)
        tof_up = max(0.0, CEILING - drone.z)

        return {
            "vz": drone.vz,
            "acc": acc,
            "jerk": jerk,
            "tof_down": tof_down,
            "tof_up": tof_up
        }

# =========================
# PID BASELINE
# =========================
class PID:
    def __init__(self):
        self.target = 1.5
        self.kp = 15
        self.kd = 8
        self.integral = 0.0

    def step(self, s):
        error = self.target - s["tof_down"]
        self.integral += error * DT
        self.integral = np.clip(self.integral, -5, 5)
        thrust = self.kp * error - self.kd * s["vz"] + 0.5 * self.integral + HOVER_THRUST
        return np.clip(thrust, 0, MAX_THRUST)

# =========================
# CONTROL VIABILIDAD V2
# =========================
class Viability:
    def __init__(self):
        self.kappa_delta = 0.04
        self.kappa_lf = 0.15
        self.prev_A = 0.0
        self.thrust_base = HOVER_THRUST
        self.banda_min = 0.18
        self.banda_max = 0.32

    def delta_struct(self, s):
        # Fix: Δ_struct mide nerviosismo, no velocidad
        return 0.4 * abs(s["acc"]) + 0.1 * abs(s["jerk"])

    def A_sys(self, thrust, s):
        # Respuesta = cambio en aceleración por unidad de thrust
        if abs(thrust - HOVER_THRUST) < 0.1:
            return 1.0
        return np.clip(abs(s["acc"]) / (abs(thrust - HOVER_THRUST) + 1e-3), 0, 1)

    def LF(self, thrust):
        return 1 - (thrust / MAX_THRUST)

    def step(self, s):
        delta = self.delta_struct(s)
        thrust = self.thrust_base

        A = self.A_sys(thrust, s)
        e_R = -(A - self.prev_A)
        self.prev_A = A
        lf = self.LF(thrust)

        # 1. Regla: si el sistema está muerto, genera diferencia suave
        if delta < self.kappa_delta:
            thrust += np.random.uniform(-0.5, 0.5)

        # 2. Regla: amortiguación siempre activa. Mata energía
        thrust -= 3.0 * s["vz"]

        # 3. Regla: sin margen de empuje, baja
        if lf < self.kappa_lf:
            thrust -= 3.0

        # 4. Regla: techo cerca, baja
        if s["tof_up"] < 0.15:
            thrust -= 5.0

        # 5. Regla: suelo cerca, sube
        if s["tof_down"] < 0.08:
            thrust += 5.0

        # 6. Regla: zona muerta. Si estás bien y quieto, no toques nada
        if self.banda_min < s["tof_down"] < self.banda_max and abs(s["vz"]) < 0.08:
            thrust = self.thrust_base

        thrust = np.clip(thrust, 0, MAX_THRUST)
        return thrust, delta, A, e_R, lf

# =========================
# SIMULACIÓN
# =========================
def run(controller_type="viab"):
    drone = Drone()
    sensors = Sensors()

    if controller_type == "pid":
        ctrl = PID()
    else:
        ctrl = Viability()

    log = []

    for t in np.arange(0, SIM_TIME, DT):
        s = sensors.read(drone)

        if controller_type == "pid":
            thrust = ctrl.step(s)
            delta = A = e_R = lf = 0
        else:
            thrust, delta, A, e_R, lf = ctrl.step(s)

        drone.step(thrust)

        log.append([
            t, drone.z, drone.vz, thrust, delta, A, e_R, lf
        ])

    return np.array(log)

# =========================
# EJECUCIÓN
# =========================
print("Corriendo PID...")
log_pid = run("pid")
print("Corriendo Viabilidad...")
log_viab = run("viab")

# =========================
# GUARDAR CSV
# =========================
with open("log_viab.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t","z","vz","thrust","delta","A","e_R","lf"])
    writer.writerows(log_viab)

# =========================
# MÉTRICAS
# =========================
def metricas(log, nombre):
    z = log[:,1]
    thrust = log[:,3]
    colisiones_techo = np.sum(z >= CEILING - 0.001)
    banda = z[(z > 0.18) & (z < 0.32)]
    print(f"\n{nombre}:")
    print(f" Altura media: {np.mean(z):.3f} m")
    print(f" Altura min-max: {np.min(z):.3f} - {np.max(z):.3f} m")
    print(f" Banda estable 0.18-0.32m: {len(banda)/len(z)*100:.1f}% del tiempo")
    print(f" Thrust medio: {np.mean(thrust):.2f} N")
    print(f" Thrust std: {np.std(thrust):.2f} N")
    print(f" Colisiones techo: {colisiones_techo}")

metricas(log_pid, "PID")
metricas(log_viab, "VIABILIDAD")

# =========================
# GRÁFICAS
# =========================
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.title("Test 1: Caja 0.5m - Altura")
plt.plot(log_pid[:,0], log_pid[:,1], label="PID", linewidth=2)
plt.plot(log_viab[:,0], log_viab[:,1], label="Viabilidad", linewidth=2)
plt.axhline(CEILING, linestyle="--", color="r", label="Techo")
plt.axhline(0.18, linestyle=":", color="g", label="Banda viable")
plt.axhline(0.32, linestyle=":", color="g")
plt.ylabel("z [m]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3,1,2)
plt.title("Thrust")
plt.plot(log_pid[:,0], log_pid[:,3], label="PID")
plt.plot(log_viab[:,0], log_viab[:,3], label="Viabilidad")
plt.axhline(HOVER_THRUST, linestyle="--", color="k", alpha=0.5, label="Hover")
plt.ylabel("Thrust [N]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3,1,3)
plt.title("Viabilidad: LF y Δ_struct")
plt.plot(log_viab[:,0], log_viab[:,7], label="LF", color="orange")
plt.plot(log_viab[:,0], log_viab[:,4], label="Δ_struct", color="purple")
plt.axhline(0.15, linestyle="--", color="orange", alpha=0.5, label="κ_LF")
plt.axhline(0.04, linestyle="--", color="purple", alpha=0.5, label="κ_Δ")
plt.xlabel("Tiempo [s]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultado_test1.png", dpi=150)
plt.show()

print("\nListo. Revisa resultado_test1.png y log_viab.csv")