import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim
import pandas as pd


# Read the CSV file
df = pd.read_csv("C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/storage_simulationData/disturbance/disturbance.csv")

# Extract columns as lists
time_values = df['time'].tolist()
disturbance_values = df['disturbance'].tolist()
freq_values = df['avg_freq_dev'].tolist()



# === 1) SYSTEM & CONTROLLER PARAMETERS ===
q_nl = 0.01        # no-load pu flow
Ppu   = 700/900    # Per-unit power
Dt    = 0.01       # Damping-term
A_t   = 1          # Gain turbine
Tw    = 1.1        # water_intertia time constant
Tr    = 5.7        # governor time constant
r     = 0.65       # temporary droop
Tf    = 0.05       # filter time constant
R     = 0.12       # permanent droop
Tg    = 0.5        # servo time constant
H     = 6          # inertia

# Derived parameters for Gt
q0 = (Ppu + A_t * q_nl) / A_t
T1 = (q0 - q_nl) * Tw
T2 = q0 * Tw / 2

#  calculate damping
Dg = q0 * Dt

#Gt(s) = A_t * (1 – s·T1) / (1 + s·T2)
num_gt = [-A_t * T1, A_t]
den_gt = [T2, 1]
Gt = lti(num_gt, den_gt)

#Gc(s) = (1 + s·Tr) / [r·Tr·Tf·s^2 + (r·Tr + R·Tr)·s + R]
num_gc = [Tr, 1]
den_gc = [r * Tr * Tf, (r * Tr + R * Tr), R]
Gc = lti(num_gc, den_gc)

#Gs(s) = 1 / (1 + s·Tg)
num_gs = [1]
den_gs = [Tg, 1]
Gs = lti(num_gs, den_gs)

num_12  = np.convolve(Gt.num, Gc.num)
den_12  = np.convolve(Gt.den, Gc.den)
num_123 = np.convolve(num_12, Gs.num)
den_123 = np.convolve(den_12, Gs.den)

num_gp = np.polyadd(num_123, Dg * den_123) * 15
den_gp = den_123

num_fcr = 4 * num_gp
den_fcr = den_gp



Kd_list = [0, 0.01] #, 0.005, 0.01, 0.02]

plt.figure(figsize=(8, 4))

for Kd in Kd_list:
    Gj = lti([50/3600], [2 * H, 50 * Kd])

    num_cl = np.convolve(Gj.num, den_fcr)
    den_cl = np.polyadd(
        np.convolve(Gj.den, den_fcr),
        np.convolve(Gj.num, num_fcr)
    )
    sys_cl = lti(num_cl, den_cl)

    t_out, y_out, _ = lsim(sys_cl, U=disturbance_values, T=time_values)

    plt.plot(t_out, -y_out, label=r'Requirement Representation, $K_d$ = ' + str(Kd))

plt.plot(time_values, freq_values, label='TOPS')

plt.xlabel('Time [s]')
plt.ylabel(r'$\Delta f$ [Hz]')
plt.title('Frequency Response of TOPS and Requirement Representation')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




