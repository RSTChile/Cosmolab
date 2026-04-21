import numpy as np
import matplotlib.pyplot as plt
import csv

# =========================
# CONFIG
# =========================
DT = 0.01
SIM_TIME = 60.0
GRAVITY = 9.81
MASS = 1.2
MAX_THRUST = 20.0
HOVER_THRUST = MASS * GRAVITY

# Mundo
CEILING = 2.5
FLOOR = 0.0
HOME_DIST = 50.0

# Batería
BATTERY_START = 0.09
BATTERY_HOVER_CURRENT = 5.0
BATTERY_CAPACITY_WH = 15.0
MAX_CURRENT = 15.0
VOLTAGE = 12.0

# =========================
# DRON 2D + BATERÍA
# =========================
class Drone2D:
    def __init__(self):
        self.z = 2.0 # FIX: inicia en vuelo
        self.vz = 0.0
        self.y = HOME_DIST
        self.vy = 0.0
        self.mass = MASS
        self.max_thrust = MAX_THRUST
        self.battery = BATTERY_START
        self.crashed = False
        self.landed = False

    def step(self, thrust_z, thrust_y):
        if self.crashed or self.landed or self.battery <= 0:
            self.vz = 0
            self.vy = 0
            if self.battery <= 0 and self.z > 0.05:
                self.crashed = True
            if self.z <= 0.05:
                self.landed = True
            return

        thrust_mag = np.sqrt(thrust_z**2 + thrust_y**2)
        current = BATTERY_HOVER_CURRENT * (thrust_mag / HOVER_THRUST)
        current = np.clip(current, 0, MAX_CURRENT)
        power_w = current * VOLTAGE
        energy_used_wh = power_w * DT / 3600.0
        self.battery -= energy_used_wh / BATTERY_CAPACITY_WH
        self.battery = max(0.0, self.battery)

        thrust_z = np.clip(thrust_z, 0, self.max_thrust)
        az = (thrust_z / self.mass) - GRAVITY
        self.vz += az * DT
        self.z += self.vz * DT

        ay = thrust_y / self.mass
        self.vy += ay * DT
        self.y += self.vy * DT

        if self.z <= FLOOR:
            self.z = FLOOR
            self.vz = max(0, self.vz) * 0.1
            if not self.crashed:
                self.landed = True
        if self.z >= CEILING:
            self.z = CEILING
            self.vz = -abs(self.vz) * 0.3

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

        return {
            "vz": drone.vz, "vy": drone.vy,
            "acc_z": acc_z, "acc_y": acc_y,
            "jerk_z": jerk_z, "jerk_y": jerk_y,
            "tof_down": max(0.0, drone.z),
            "tof_up": max(0.0, CEILING - drone.z),
            "y": drone.y, "z": drone.z,
            "battery": drone.battery,
            "crashed": drone.crashed,
            "landed": drone.landed
        }

# =========================
# PID + RTH CLÁSICO
# =========================
class PID_RTH:
    def __init__(self):
        self.target_z = 20.0
        self.target_y = 0.0
        self.kp_z, self.kd_z = 15, 8
        self.kp_y, self.kd_y = 10, 6
        self.integral_z = 0.0
        self.integral_y = 0.0

    def step(self, s):
        if s["crashed"] or s["landed"]:
            return 0, 0

        err_z = self.target_z - s["tof_down"]
        self.integral_z += err_z * DT
        self.integral_z = np.clip(self.integral_z, -5, 5)
        thrust_z = self.kp_z * err_z - self.kd_z * s["vz"] + 0.5 * self.integral_z + HOVER_THRUST

        err_y = self.target_y - s["y"]
        self.integral_y += err_y * DT
        self.integral_y = np.clip(self.integral_y, -5, 5)
        thrust_y = self.kp_y * err_y - self.kd_y * s["vy"] + 0.3 * self.integral_y

        return np.clip(thrust_z, 0, MAX_THRUST), np.clip(thrust_y, -10, 10)

# =========================
# CONTROL VIABILIDAD + RTH INTELIGENTE V3
# =========================
class Viability_RTH:
    def __init__(self):
        self.kappa_delta = 0.05
        self.kappa_lf = 0.15
        self.kappa_y = 1.0
        self.kappa_batt_emerg = 0.08 # 8% emergencia dura
        self.kappa_batt_eval = 0.15 # 15% evalúa planes
        self.prev_A = 0.0
        self.thrust_base = HOVER_THRUST
        self.rth_decision = 0 # 0=None, 1=land, 2=rth

    def delta_struct(self, s):
        return 0.4 * (abs(s["acc_z"]) + abs(s["acc_y"])) + 0.1 * (abs(s["jerk_z"]) + abs(s["jerk_y"]))

    def A_sys(self, thrust_z, thrust_y, s):
        resp_z = abs(s["acc_z"]) / (abs(thrust_z - HOVER_THRUST) + 1e-3)
        resp_y = abs(s["acc_y"]) / (abs(thrust_y) + 1e-3)
        return np.clip(0.7 * resp_z + 0.3 * resp_y, 0, 1)

    def LF(self, thrust_z):
        return 1 - (thrust_z / MAX_THRUST)

    def V_y(self, s, lf):
        if abs(s["y"]) < self.kappa_y:
            return 1.0
        return max(0.0, 1.0 - (abs(s["y"]) - self.kappa_y) / 10.0) * lf

    def estimate_energy(self, s, plan):
        # Estima Wh para ejecutar un plan
        dist = abs(s["y"])
        vel_cruise = 3.0
        if plan == "rth":
            tiempo = dist / vel_cruise
            # RTH: vuelo a 2m + ascenso mínimo
            current = BATTERY_HOVER_CURRENT * 1.1 # 10% más por traslación
        else: # land
            tiempo = s["tof_down"] / 2.0 # Descenso a 2m/s
            current = BATTERY_HOVER_CURRENT * 0.7 # Menos que hover
        power_w = current * VOLTAGE
        return power_w * tiempo / 3600.0

    def step(self, s):
        if s["crashed"] or s["landed"]:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        delta = self.delta_struct(s)
        thrust_z = self.thrust_base
        thrust_y = 0.0

        A = self.A_sys(thrust_z, thrust_y, s)
        e_R = -(A - self.prev_A)
        self.prev_A = A
        lf = self.LF(thrust_z)
        vy_value = self.V_y(s, lf)

        # FIX: Decisión en dos capas
        wh_rth = self.estimate_energy(s, "rth")
        wh_land = self.estimate_energy(s, "land")
        batt_wh = s["battery"] * BATTERY_CAPACITY_WH

        if self.rth_decision == 0:
            if s["battery"] < self.kappa_batt_emerg:
                self.rth_decision = 1 # Emergencia: land ya
            elif s["battery"] < self.kappa_batt_eval:
                # Evalúa planes
                if batt_wh > wh_rth * 1.2: # 20% margen
                    self.rth_decision = 2
                else:
                    self.rth_decision = 1
            else:
                self.rth_decision = 2 # Batería ok, intenta RTH

        # JERARQUÍA: LAND anula todo
        if self.rth_decision == 1:
            thrust_z = HOVER_THRUST * 0.6 # Descenso ~4m/s²
            thrust_y = -2.0 * s["vy"] # Solo frena Y
            if s["tof_down"] < 0.3:
                thrust_z = HOVER_THRUST * 0.95 # Suaviza toque
            return np.clip(thrust_z, 0, MAX_THRUST), np.clip(thrust_y, -10, 10), delta, A, e_R, lf, vy_value, wh_rth, wh_land, self.rth_decision

        # RTH
        target_z = min(2.0, CEILING - 0.3)
        err_z = target_z - s["tof_down"]
        thrust_z += 8.0 * err_z - 3.0 * s["vz"]

        if abs(s["y"]) > self.kappa_y:
            fuerza_retorno = -np.sign(s["y"]) * 12.0 * (1.0 - vy_value)
            thrust_y += fuerza_retorno

        thrust_y -= 2.0 * s["vy"]

        if delta < self.kappa_delta:
            thrust_z += np.random.uniform(-0.4, 0.4)
            thrust_y += np.random.uniform(-0.2, 0.2)

        thrust_z -= 2.5 * s["vz"]

        if lf < self.kappa_lf:
            thrust_z -= 3.0
        if s["tof_down"] < 0.5:
            thrust_z += 5.0
        if s["tof_up"] < 0.2:
            thrust_z -= 5.0

        return np.clip(thrust_z, 0, MAX_THRUST), np.clip(thrust_y, -10, 10), delta, A, e_R, lf, vy_value, wh_rth, wh_land, self.rth_decision

# =========================
# SIMULACIÓN
# =========================
def run(controller_type="viab"):
    drone = Drone2D()
    sensors = Sensors2D()

    if controller_type == "pid":
        ctrl = PID_RTH()
    else:
        ctrl = Viability_RTH()

    log = []

    for t in np.arange(0, SIM_TIME, DT):
        s = sensors.read(drone)

        if controller_type == "pid":
            thrust_z, thrust_y = ctrl.step(s)
            delta = A = e_R = lf = vy_value = wh_rth = wh_land = 0
            decision = 2
        else:
            thrust_z, thrust_y, delta, A, e_R, lf, vy_value, wh_rth, wh_land, decision = ctrl.step(s)

        drone.step(thrust_z, thrust_y)
        log.append([t, drone.z, drone.y, drone.vz, drone.vy, thrust_z, thrust_y,
                   drone.battery, delta, A, e_R, lf, vy_value, wh_rth, wh_land,
                   int(drone.crashed), int(drone.landed), int(decision)])

        if drone.crashed or drone.landed:
            # FIX: no rellenar. Corta aquí. Tiempo real.
            break

    return np.array(log, dtype=float)

# =========================
# EJECUCIÓN
# =========================
print("Corriendo PID + RTH...")
log_pid = run("pid")
print("Corriendo Viabilidad + RTH...")
log_viab = run("viab")

# =========================
# GUARDAR CSV - AMBOS
# =========================
with open("log_exp6b_pid.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t","z","y","vz","vy","thrust_z","thrust_y","battery","delta","A","e_R","lf","vy_value","wh_rth","wh_land","crashed","landed","decision"])
    writer.writerows(log_pid)

with open("log_exp6b_viab.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["t","z","y","vz","vy","thrust_z","thrust_y","battery","delta","A","e_R","lf","vy_value","wh_rth","wh_land","crashed","landed","decision"])
    writer.writerows(log_viab)

# =========================
# MÉTRICAS
# =========================
def metricas(log, nombre):
    y = log[:,2]
    vz = log[:,3]
    battery = log[:,7]
    crashed = log[:,15]
    landed = log[:,16]
    decision = log[:,17]

    colapso = crashed[-1] > 0.5
    aterrizo_ok = landed[-1] > 0.5 and not colapso
    y_final = y[-1]
    batt_final = battery[-1] * 100
    dec_text = {0:"None", 1:"land", 2:"rth"}[int(decision[-1])]
    t_evento = log[-1,0]

    print(f"\n{nombre}:")
    print(f" Decisión: {dec_text}")
    print(f" Tiempo a evento: {t_evento:.2f}s")
    print(f" Batería final: {batt_final:.1f}%")
    print(f" Posición Y final: {y_final:.1f}m de home")
    print(f" Velocidad Z final: {vz[-1]:.2f}m/s")
    print(f" Aterrizó: {aterrizo_ok}")
    print(f" Colapsó: {colapso}")

metricas(log_pid, "PID")
metricas(log_viab, "VIABILIDAD")

# =========================
# GRÁFICAS
# =========================
plt.figure(figsize=(12,14))

plt.subplot(6,1,1)
plt.title("Experimento 6b: Batería 9% desde z=2.0m - Altura Z")
plt.plot(log_pid[:,0], log_pid[:,1], label="PID", linewidth=2)
plt.plot(log_viab[:,0], log_viab[:,1], label="Viabilidad", linewidth=2)
plt.axhline(CEILING, linestyle="--", color="r", label=f"Techo {CEILING}m")
plt.axhline(20.0, linestyle=":", color="b", alpha=0.5, label="Target RTH PID")
plt.ylabel("z [m]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(6,1,2)
plt.title("Posición Y - Distancia a Home")
plt.plot(log_pid[:,0], log_pid[:,2], label="PID", linewidth=2)
plt.plot(log_viab[:,0], log_viab[:,2], label="Viabilidad", linewidth=2)
plt.axhline(0, linestyle="--", color="g", label="Home")
plt.ylabel("y [m]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(6,1,3)
plt.title("Batería %")
plt.plot(log_pid[:,0], log_pid[:,7]*100, label="PID")
plt.plot(log_viab[:,0], log_viab[:,7]*100, label="Viabilidad")
plt.axhline(0, linestyle="--", color="r", label="0%")
plt.ylabel("Batería [%]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(6,1,4)
plt.title("Velocidad Vertical")
plt.plot(log_pid[:,0], log_pid[:,3], label="PID")
plt.plot(log_viab[:,0], log_viab[:,3], label="Viabilidad")
plt.axhline(0, linestyle="--", color="k", alpha=0.5)
plt.ylabel("vz [m/s]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(6,1,5)
plt.title("Estado: Crashed vs Landed")
plt.plot(log_pid[:,0], log_pid[:,15], label="PID Crashed", color='red')
plt.plot(log_pid[:,0], log_pid[:,16], label="PID Landed", color='blue')
plt.plot(log_viab[:,0], log_viab[:,15], label="Viab Crashed", color='orange', linestyle='--')
plt.plot(log_viab[:,0], log_viab[:,16], label="Viab Landed", color='green', linestyle='--')
plt.ylabel("Estado")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(6,1,6)
plt.title("Viabilidad: Energía RTH vs LAND")
plt.plot(log_viab[:,0], log_viab[:,13], label="Wh_RTH", color="red")
plt.plot(log_viab[:,0], log_viab[:,14], label="Wh_LAND", color="green")
plt.plot(log_viab[:,0], log_viab[:,7]*BATTERY_CAPACITY_WH, label="Wh_Batt", color="purple")
plt.ylabel("Energía [Wh]")
plt.xlabel("Tiempo [s]")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultado_exp6b.png", dpi=150)
plt.show()

print("\nListo. Revisa resultado_exp6b.png, log_exp6b_pid.csv y log_exp6b_viab.csv")