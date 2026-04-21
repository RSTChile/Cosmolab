import numpy as np
import matplotlib.pyplot as plt
import csv

# =========================
# CONFIG
# =========================
DT = 0.01
SIM_TIME = 30.0
GRAVITY = 9.81
MASS = 1.2
MAX_THRUST = 20.0
HOVER_THRUST = MASS * GRAVITY

# Mundo
CEILING = 10.0 # sin techo para este test
FLOOR = 0.0

# Viento: ráfaga lateral en +Y
WIND_START = 10.0
WIND_END = 15.0
WIND_SPEED = 8.0 # m/s
WIND_DRAG_COEF = 0.8 # fuerza = coef * v_rel^2

# =========================
# DRON 2D (Z, Y)
# =========================
class Drone2D:
    def __init__(self):
        self.z = 0.0
        self.vz = 0.0
        self.y = 0.0
        self.vy = 0.0
        self.mass = MASS
        self.max_thrust = MAX_THRUST

    def step(self, thrust_z, thrust_y, wind_y):
        # Fuerza viento: arrastre cuadrático
        v_rel_y = wind_y - self.vy
        f_wind_y = WIND_DRAG_COEF * np.sign(v_rel_y) * v_rel_y**2

        # Dinámica Z
        thrust_z = np.clip(thrust_z, 0, self.max_thrust)
        az = (thrust_z / self.mass) - GRAVITY
        self.vz += az * DT
        self.z += self.vz * DT

        # Dinámica Y
        ay = (thrust_y + f_wind_y) / self.mass
        self.vy += ay * DT
        self.y += self.vy * DT

        if self.z <= FLOOR:
            self.z = FLOOR
            self.vz = max(0, self.vz) * 0.1

# =========================
# SENSORES
# =========================
class Sensors2D:
    def __init__(self):
        self.prev_acc_z = 0.0
        self.prev_acc_y = 0.0
        self.prev_vz = 0.0
        self.prev_vy = 0.0

    def read(self, drone):
        acc_z = (drone.vz - self.prev_vz) / DT
        acc_y = (drone.vy - self.prev_vy) / DT
        jerk_z = (acc_z - self.prev_acc_z) / DT
        jerk_y = (acc_y - self.prev_acc_y) / DT

        self.prev_acc_z = acc_z
        self.prev_acc_y = acc_y
        self.prev_vz = drone.vz
        self.prev_vy = drone.vy

        tof_down = max(0.0, drone.z)
        tof_up = max(0.0, CEILING - drone.z)

        return {
            "vz": drone.vz, "vy": drone.vy,
            "acc_z": acc_z, "acc_y": acc_y,
            "jerk_z": jerk_z, "jerk_y": jerk_y,
            "tof_down": tof_down, "tof_up": tof_up,
            "y": drone.y, "z": drone.z
        }

# =========================
# PID BASELINE 2D
# =========================
class PID2D:
    def __init__(self):
        self.target_z = 2.0
        self.target_y = 0.0
        self.kp_z, self.kd_z = 15, 8
        self.kp_y, self.kd_y = 10, 6
        self.integral_z = 0.0
        self.integral_y = 0.0

    def step(self, s):
        err_z = self.target_z - s["tof_down"]
        self.integral_z += err_z * DT
        thrust_z = self.kp_z * err_z - self.kd_z * s["vz"] + 0.5 * self.integral_z + HOVER_THRUST

        err_y = self.target_y - s["y"]
        self.integral_y += err_y * DT
        thrust_y = self.kp_y * err_y - self.kd_y * s["vy"] + 0.3 * self.integral_y

        return np.clip(thrust_z, 0, MAX_THRUST), np.clip(thrust_y, -10, 10)

# =========================
# CONTROL VIABILIDAD 2D
# =========================
class Viability2D:
    def __init__(self):
        self.kappa_delta = 0.05
        self.kappa_lf = 0.15
        self.prev_A = 0.0
        self.thrust_base = HOVER_THRUST
        self.banda_min = 1.8
        self.banda_max = 2.2
        self.hist_z = []
        self.t_en_banda = 0.0

    def delta_struct(self, s):
        return 0.4 * (abs(s["acc_z"]) + abs(s["acc_y"])) + 0.1 * (abs(s["jerk_z"]) + abs(s["jerk_y"]))

    def A_sys(self, thrust_z, thrust_y, s):
        resp_z = abs(s["acc_z"]) / (abs(thrust_z - HOVER_THRUST) + 1e-3)
        resp_y = abs(s["acc_y"]) / (abs(thrust_y) + 1e-3)
        return np.clip(0.7 * resp_z + 0.3 * resp_y, 0, 1)

    def LF(self, thrust_z):
        return 1 - (thrust_z / MAX_THRUST)

    def step(self, s):
        delta = self.delta_struct(s)
        thrust_z = self.thrust_base
        thrust_y = 0.0

        A = self.A_sys(thrust_z, thrust_y, s)
        e_R = -(A - self.prev_A)
        self.prev_A = A
        lf = self.LF(thrust_z)

        if delta < self.kappa_delta:
            thrust_z += np.random.uniform(-0.4, 0.4)
            thrust_y += np.random.uniform(-0.2, 0.2)

        thrust_z -= 2.5 * s["vz"]
        thrust_y -= 2.0 * s["vy"] # amortiguación lateral

        if lf < self.kappa_lf:
            thrust_z -= 3.0
        if s["tof_down"] < 0.5:
            thrust_z += 5.0

        # Banda altura viable
        en_banda = self.banda_min < s["tof_down"] < self.banda_max
        if en_banda:
            self.t_en_banda += DT
            self.hist_z.append(s["tof_down"])
            if len(self.hist_z) > 200:
                self.hist_z.pop(0)
        else:
            self.t_en_banda = 0.0
            self.hist_z = []

        if en_banda and abs(s["vz"]) < 0.1 and self.t_en_banda > 1.0:
            z_centro = np.mean(self.hist_z)
            error_banda = z_centro - s["tof_down"]
            thrust_z += 4.0 * error_banda
            thrust_z = np.clip(thrust_z, HOVER_THRUST - 1.0, HOVER_THRUST + 1.0)

        return np.clip(thrust_z, 0, MAX_THRUST), np.clip(thrust_y, -10, 10), delta, A, e_R, lf

# =========================
# SIMULACIÓN
# =========================
def run(controller_type="viab"):
    drone = Drone2D()
    sensors = Sensors2D()

    if controller_type == "pid":
        ctrl = PID2D()
    else:
        ctrl = Viability2D()

    log = []

    for t in np.arange(0, SIM_TIME, DT):
        wind_y = WIND_SPEED if WIND_START <= t <= WIND_END else 0.0
        s = sensors.read(drone)

        if controller_type == "pid":
            thrust_z, thrust_y = ctrl.step(s)
            delta = A = e_R = lf = 0
        else:
            thrust_z, thrust_y, delta, A, e_R, lf = ctrl.step(s)

        drone.step(thrust_z, thrust_y, wind_y)
        log.append([t, drone.z, drone.y, drone.vz, drone.vy, thrust_z, thrust_y, delta, A, e_R, lf, wind_y])

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
with open("log_viab_2d-4.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t","z","y","vz","vy","thrust_z","thrust_y","delta","A","e_R","lf","wind_y"])
    writer.writerows(log_viab)

# =========================
# MÉTRICAS
# =========================
def metricas(log, nombre):
    y = log[:,2]
    vz = log[:,3]
    vy = log[:,4]
    thrust_z = log[:,5]
    wind = log[:,11]

    idx_viento = (wind > 0)
    deriva_max = np.max(np.abs(y[idx_viento])) if np.any(idx_viento) else 0
    corriente_media = np.mean(thrust_z[idx_viento]) if np.any(idx_viento) else np.mean(thrust_z)
    energia_total = np.sum(thrust_z) * DT / 3600 * 12 # Wh aprox con 12V

    print(f"\n{nombre}:")
    print(f" Deriva máxima en viento: {deriva_max:.3f} m")
    print(f" Thrust medio en viento: {corriente_media:.2f} N")
    print(f" Energía total 30s: {energia_total:.2f} Wh")
    print(f" Velocidad Y pico: {np.max(np.abs(vy)):.2f} m/s")

metricas(log_pid, "PID")
metricas(log_viab, "VIABILIDAD")

# =========================
# GRÁFICAS
# =========================
plt.figure(figsize=(12,10))

plt.subplot(4,1,1)
plt.title("Test 2: Viento 8m/s - Posición Y")
plt.plot(log_pid[:,0], log_pid[:,2], label="PID", linewidth=2)
plt.plot(log_viab[:,0], log_viab[:,2], label="Viabilidad", linewidth=2)
plt.axvspan(WIND_START, WIND_END, alpha=0.2, color='red', label='Viento')
plt.ylabel("y [m]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4,1,2)
plt.title("Altura Z")
plt.plot(log_pid[:,0], log_pid[:,1], label="PID")
plt.plot(log_viab[:,0], log_viab[:,1], label="Viabilidad")
plt.ylabel("z [m]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4,1,3)
plt.title("Thrust Z")
plt.plot(log_pid[:,0], log_pid[:,5], label="PID")
plt.plot(log_viab[:,0], log_viab[:,5], label="Viabilidad")
plt.axhline(HOVER_THRUST, linestyle="--", color="k", alpha=0.5, label="Hover")
plt.ylabel("Thrust Z [N]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4,1,4)
plt.title("Thrust Y / Corrección lateral")
plt.plot(log_pid[:,0], log_pid[:,6], label="PID")
plt.plot(log_viab[:,0], log_viab[:,6], label="Viabilidad")
plt.xlabel("Tiempo [s]")
plt.ylabel("Thrust Y [N]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultado_test2.png", dpi=150)
plt.show()

print("\nListo. Revisa resultado_test4.png y log_viab_2d-4.csv")