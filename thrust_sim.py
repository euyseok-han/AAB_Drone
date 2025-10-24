import numpy as np
import matplotlib.pyplot as plt

# --- Physical constants ---
rho = 1.225  # air density (kg/m^3)
g = 9.81
mass = 0.025  # kg (25 g drone)
target_thrust = 0.05 * g  # N, roughly 50 g of thrust

# --- Motor/Prop parameters (fixed for all sweeps except x-axis) ---
prop_radius = 0.05        # m (5 cm)
A = np.pi * prop_radius**2
Rm = 0.2                  # Ohms
Kt = 0.0008               # Nm/A (example)
Kv_base = 1000.0          # RPM/V for baseline

# Derived constant k (thrust coefficient)
# T = k * ω²  → approximate k using prop area
k = 0.5 * rho * A * prop_radius**2  # crude estimate

# --- Helper functions ---

def omega_from_thrust(T):
    """Compute motor angular speed (rad/s) needed for thrust T"""
    return np.sqrt(T / k)

def thrust_from_omega(omega):
    """Thrust from angular speed"""
    return k * omega**2

def voltage_current_power(omega, Kv, Rm=Rm):
    """Return (V, I, P) for given angular speed and Kv"""
    Kv_rad = Kv * (2*np.pi/60)  # convert RPM/V to rad/s/V
    V = omega / Kv_rad + Rm * 1.0  # crude; ignoring I0
    I = (V - omega / Kv_rad) / Rm
    vh = np.sqrt(thrust_from_omega(omega) / (2 * rho * A))
    P = thrust_from_omega(omega) * vh_+ 
    return V, I, P

# --- 1. Thrust vs Voltage ---
voltages = np.linspace(1, 12, 100)
thrust_v = []
for V in voltages:
    # invert approximately to find omega for given V
    omega = V * Kv_base * (2*np.pi/60)
    thrust_v.append(thrust_from_omega(omega))
thrust_v = np.array(thrust_v)

# --- 2. Thrust vs Current ---
currents = np.linspace(0.1, 10, 100)
thrust_i = []
for I in currents:
    V = I*Rm + 5  # assume 5 V back-EMF baseline
    omega = (V - I*Rm) * Kv_base * (2*np.pi/60)
    thrust_i.append(thrust_from_omega(omega))
thrust_i = np.array(thrust_i)

# --- 3. Thrust vs Power ---
powers = np.linspace(0.1, 50, 100)
thrust_p = (powers**(2/3)) * ((2*rho*A)**(1/3))  # inverted P = T^(3/2)/sqrt(2ρA)

# --- 4. Thrust vs KV ---
kv_values = np.linspace(500, 4000, 100)
thrust_kv = []
fixed_voltage = 7.4  # 2S LiPo
for kv in kv_values:
    omega = fixed_voltage * kv * (2*np.pi/60)
    thrust_kv.append(thrust_from_omega(omega))
thrust_kv = np.array(thrust_kv)

# --- Plotting ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(voltages, thrust_v*1000)
plt.axhline(target_thrust*1000, color='r', linestyle='--', label='target 50g')
plt.xlabel('Voltage (V)')
plt.ylabel('Thrust (mN)')
plt.title('Thrust vs Voltage')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(currents, thrust_i*1000)
plt.axhline(target_thrust*1000, color='r', linestyle='--')
plt.xlabel('Current (A)')
plt.ylabel('Thrust (mN)')
plt.title('Thrust vs Current')

plt.subplot(2, 2, 3)
plt.plot(powers, thrust_p*1000)
plt.axhline(target_thrust*1000, color='r', linestyle='--')
plt.xlabel('Power (W)')
plt.ylabel('Thrust (mN)')
plt.title('Thrust vs Power')

plt.subplot(2, 2, 4)
plt.plot(kv_values, thrust_kv*1000)
plt.axhline(target_thrust*1000, color='r', linestyle='--')
plt.xlabel('Motor KV (RPM/V)')
plt.ylabel('Thrust (mN)')
plt.title('Thrust vs KV')

plt.tight_layout()
plt.show()
