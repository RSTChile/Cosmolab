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

# =========================
# DRON SIMPLE (1D vertical)
# =========================
class Drone:
    def __init__(self):
        self.z = 0.0
        self.vz = 0.0
        self.mass = 1.2
        self.max_thrust = 20.0

    def step(self, thrust):
        thrust = np.clip(thrust, 0, self.max_thrust)
        az = (thrust / self.mass) - GRAVITY
        self.vz += az * DT
        self.z += self.vz * DT

        # colisiones
        if self.z <= FLOOR:
            self.z = FLOOR
            self.vz = 0
        if self.z >= CEILING:
            self.z = CEILING
            self.vz = -abs(self.vz) * 0.3

# =========================
# SENSORES
# =========================
class Sensors:
    def __init__(self):
        self.prev_acc = 0

    def read(self, drone):
        # IMU
        acc = drone.vz / DT
        jerk = (acc - self.prev_acc) / DT
        self.prev_acc = acc

        # ToF
        tof_down = drone.z
        tof_up = CEILING - drone.z

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

    def step(self, s):
        error = self.target - s["tof_down"]
        thrust = self.kp * error - self.kd * s["vz"] + 12
        return thrust

# =========================
# CONTROL VIABILIDAD
# =========================
class Viability:
    def __init__(self):
        self.kappa_delta = 0.05
        self.kappa_lf = 0.1
        self.prev_A = 0.0

    def delta_struct(self, s):
        return abs(s["vz"]) + 0.3 * abs(s["acc"]) + 0.1 * abs(s["jerk"])

    def A_sys(self, thrust, s):
        # correlación simple acción → respuesta
        return min(1.0, abs(s["vz"]) / (abs(thrust) + 1e-3))

    def LF(self, thrust):
        return 1 - (thrust / 20.0)

    def step(self, s):
        delta = self.delta_struct(s)
        thrust = 12.0  # base hover

        A = self.A_sys(thrust, s)
        e_R = -(A - self.prev_A)
        self.prev_A = A

        lf = self.LF(thrust)

        # ===== REGLAS κ =====

        # 1. muerto → generar diferencia
        if delta < self.kappa_delta:
            thrust += np.random.uniform(-2, 2)

        # 2. sin margen → bajar
        if lf < self.kappa_lf:
            thrust -= 3

        # 3. techo cerca → bajar
        if s["tof_up"] < 0.2:
            thrust -= 5

        # 4. suelo muy cerca → subir
        if s["tof_down"] < 0.1:
            thrust += 5

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
log_pid = run("pid")
log_viab = run("viab")

# =========================
# GUARDAR CSV
# =========================
with open("log_viab.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t","z","vz","thrust","delta","A","e_R","lf"])
    writer.writerows(log_viab)

# =========================
# GRÁFICAS
# =========================
plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.title("Altura")
plt.plot(log_pid[:,0], log_pid[:,1], label="PID")
plt.plot(log_viab[:,0], log_viab[:,1], label="Viabilidad")
plt.axhline(CEILING, linestyle="--", color="r")
plt.legend()

plt.subplot(2,1,2)
plt.title("Thrust")
plt.plot(log_pid[:,0], log_pid[:,3], label="PID")
plt.plot(log_viab[:,0], log_viab[:,3], label="Viabilidad")
plt.legend()

plt.tight_layout()
plt.show()