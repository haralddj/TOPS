import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim
import pandas as pd


# Read the CSV file
df = pd.read_csv(r'C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS\Figures\disturbance\disturbance.csv')

# Extract columns as lists
time_values = df['time'].tolist()
disturbance_values = df['disturbance'].tolist()
freq_values = df['avg_freq_dev'].tolist()



# === 1) PARAMETERS (fill these in) ===
Kp      =  3 # proportional gain
Ki      =  0.7 # integral gain
Ts      = 0.2  # servo time constant
Tw      = 1.5  # water‐ways time constant
Sn_fcr  =  600 # FCR unit gain (called S_{n–FCR} in your figure)
f0      = 50  # nominal system frequency
Sn      = 42000  # machine base power
H       = 4.5  # inertia constant
k     = 0.01  # damping coefficient (k⋅f₀)

# === 2) BUILD EACH BLOCK AS AN LTI SYSTEM ===

# 2.1 PI controller: (Kp s + Ki)/s
num_pi = [Kp, Ki]
den_pi = [1, 0]
ctrl = lti(num_pi, den_pi)

# 2.2 Servo: 1 / (Ts s + 1)
servo = lti([1], [Ts, 1])

# 2.3 Water-ways: (−Tw s + 1) / (0.5 Tw s + 1)
num_ww = [-Tw, 1]
den_ww = [0.5*Tw, 1]
waters = lti(num_ww, den_ww)

# 2.4 FCR unit path: cascade of PI → Servo → Water-ways → Sn_fcr gain
#    series multiply = np.convolve of numerators / denominators
num_1  = np.convolve(ctrl.num,   servo.num)
den_1  = np.convolve(ctrl.den,   servo.den)
num_2  = np.convolve(num_1,       waters.num) * Sn_fcr
den_2  = np.convolve(den_1,       waters.den)
fcr_unit = lti(num_2, den_2)

# 2.5 Power system: (f0/Sn) / (2 H s + kf0)
num_ps = [f0/Sn]
den_ps = [2*H, k*f0]
power_sys = lti(num_ps, den_ps)

# === 3) CLOSE THE LOOP (unity feedback) ===
#   Disturbance d adds before the power system block,
#   control FCR feedback is around power_sys.
#   So the closed‐loop TF from d → Δf is:
#
#       T_cl(s) = PowerSys(s) / [1 + PowerSys(s)·FCR_unit(s)]
#
#   Denominator:   D(s) = den_ps*den_fcr + num_ps*num_fcr
den_fb = np.polyadd(
    np.convolve(power_sys.den, fcr_unit.den),
    np.convolve(power_sys.num, fcr_unit.num)
)
num_fb = power_sys.num
sys_cl = lti(num_fb, den_fb)


# === 4) SIMULATE RESPONSE TO A STEP DISTURBANCE ===
# t = time_values       # 0…50 s
# d = [dist*10 for dist in disturbance_values]                # unit‐step disturbance
# t_out, y_out, _ = lsim(sys_cl, U=d, T=t)

# # === 5) PLOT ===
# plt.figure(figsize=(8,4))
# plt.plot(t_out, -y_out, label='Δf (to step d)')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency deviation Δf')
# plt.title('Closed‐Loop Response to disturbance')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim

# === 1) PARAMETERS (fill these in) ===
q_nl = 0.01  # Nominal load in Hz
Ppu=700/900  # Power unit in p.u.
Dt=0.01
A_t   =  1  # Gain in Gt
q0 = (Ppu + (A_t * q_nl)) / (A_t - (0 * Dt)) 
Tw=1.1
T1 = (q0 - q_nl) * Tw
T2 = q0 * Tw / 2
Dg = q0 * Dt

Tr   =  5.7  # Numerator time-const in Gc
r    =  0.65  # Scaling factor in Gc
Tf   = 0.05   # Inner pole time-const in Gc
R    = 0.12   # DC gain factor in Gc

Tg   = 0.5   # Servo motor time constant in Gs

H    =  6  # Inertia constant for G_j
Kd   =  0.01  # Damping coefficient for G_j

Dg   = q0*Dt   # Additive gain to Gp (if none, set Dg=0)

# === 2) BUILD YOUR FOUR BLOCKS AS LTI’s ===

# 2.1 Gt(s) = At * (1 – s·T1) / (1 + s·T2)
num_gt = [-A_t * T1, A_t]
den_gt = [T2, 1]
Gt = lti(num_gt, den_gt)

# 2.2 Gc(s) = (1 + s·Tr) / [ r·(1 + s·Tf)·Tr·s + R·(1 + s·Tr) ]
#    expand denominator: r·Tr·Tf·s^2 + (r·Tr + R·Tr)·s + R
num_gc = [Tr, 1]
den_gc = [r*Tr*Tf, (r*Tr + R*Tr), R]
Gc = lti(num_gc, den_gc)

# 2.3 Gs(s) = 1 / (1 + s·Tg)
num_gs = [1]
den_gs = [Tg, 1]
Gs = lti(num_gs, den_gs)

# 2.4 Gj(s) = 1 / (2 H·s + Kd)    ← your “power system” / generator
num_gj = [50/3600]
den_gj = [2*H, Kd*50]
Gj = lti(num_gj, den_gj)

# === 3) CASCADE Gt → Gc → Gs to get the “nominal” Gp_cascade ===
num_12 = np.convolve(Gt.num, Gc.num)
den_12 = np.convolve(Gt.den, Gc.den)
num_123 = np.convolve(num_12, Gs.num)
den_123 = np.convolve(den_12, Gs.den)


# === 4) ADD YOUR EXTRA GAIN Dg IN PARALLEL: Gp = Gp_cascade + Dg ===
#    Gp_cascade = num_123/den_123,  Dg = Dg/1  ⇒
#    Gp_num = num_123 + Dg·den_123,  Gp_den = den_123
num_gp = np.polyadd(num_123, Dg * den_123)*15
den_gp = den_123
Gp = lti(num_gp, den_gp)

num_fcr = 4 * num_gp
den_fcr = den_gp
Gp = lti(num_fcr, den_fcr)



# === 5) FORM THE CLOSED-LOOP: unity-feedback around Gp (forward) and Gj (plant) ===
#    CL TF  = Gj / [1 + Gp·Gj]
#    ⇒ common denom = den_gj*den_gp,  feedback adds num_gj*num_gp 
num_cl = np.convolve(Gj.num, den_fcr)
den_cl = np.polyadd(
    np.convolve(Gj.den, den_fcr),
    np.convolve(Gj.num, num_fcr)
)
sys_cl = lti(num_cl, den_cl)

# === 6) SIMULATE A STEP DISTURBANCE THROUGH THE LOOP ===
#t = time_values      # 0…50 s
t=np.arange(0, 100, 0.2)
d = np.ones_like(t)              # unit-step disturbance
t_out, y_out, _ = lsim(sys_cl, U=d, T=t)

# === 7) PLOT ===
plt.figure(figsize=(8,4))
plt.plot(t_out, -y_out, label='Δf to step d')
#plt.plot(t, freq_values, label='From TOPS')
plt.xlabel('Time [s]')
plt.ylabel('Δf [Hz]')
plt.title('Closed-Loop Response with $G_p$ in place of FCR-unit')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#Making PSS
# === X) DEFINE YOUR PSS BLOCK ===
Kpss =   50    # PSS gain
Tw   =   10    # washout time constant
T1a, T2a =0.5,0.05 ### FILL IN ###   # lead–lag #1
T1b, T2b =0.5,0.05 ### FILL IN ###   # lead–lag #2 (optional)

# 1) Washout: Tw s / (Tw s + 1)
num_wo = [Tw, 0]
den_wo = [Tw, 1]
G_wo   = lti(num_wo, den_wo)

# 2) Lead–lag #1: (T1a s + 1)/(T2a s + 1)
num_ll1 = [T1a, 1]
den_ll1 = [T2a, 1]
G_ll1   = lti(num_ll1, den_ll1)

# 3) Lead–lag #2: (T1b s + 1)/(T2b s + 1)
num_ll2 = [T1b, 1]
den_ll2 = [T2b, 1]
G_ll2   = lti(num_ll2, den_ll2)

# 4) Full PSS: cascade & gain
#    You can omit the second lead–lag if you only need one.
num_pss = np.convolve(np.convolve(G_wo.num, G_ll1.num), G_ll2.num)
den_pss = np.convolve(np.convolve(G_wo.den, G_ll1.den), G_ll2.den)
# apply the Kpss gain to the numerator
num_pss = [Kpss * c for c in num_pss]
G_pss   = lti(num_pss, den_pss)




# === 1) PARAMETERS (fill these in) ===
q_nl = 0.01  # Nominal load in Hz
Ppu=700/900  # Power unit in p.u.
Dt=0.01
A_t   =  1  # Gain in Gt
q0 = (Ppu + (A_t * q_nl)) / (A_t - (0 * Dt)) 
Tw=1.1
T1 = (q0 - q_nl) * Tw
T2 = q0 * Tw / 2
Dg = q0 * Dt

Tr   =  5.7  # Numerator time-const in Gc
r    =  0.65  # Scaling factor in Gc
Tf   = 0.05   # Inner pole time-const in Gc
R    = 0.12   # DC gain factor in Gc

Tg   = 0.5   # Servo motor time constant in Gs

H    =  6  # Inertia constant for G_j
Kd   =  0.01  # Damping coefficient for G_j

Dg   = q0*Dt   # Additive gain to Gp (if none, set Dg=0)

# === 2) BUILD YOUR FOUR BLOCKS AS LTI’s ===

# 2.1 Gt(s) = At * (1 – s·T1) / (1 + s·T2)
num_gt = [-A_t * T1, A_t]
den_gt = [T2, 1]
Gt = lti(num_gt, den_gt)

# 2.2 Gc(s) = (1 + s·Tr) / [ r·(1 + s·Tf)·Tr·s + R·(1 + s·Tr) ]
#    expand denominator: r·Tr·Tf·s^2 + (r·Tr + R·Tr)·s + R
num_gc = [Tr, 1]
den_gc = [r*Tr*Tf, (r*Tr + R*Tr), R]
Gc = lti(num_gc, den_gc)

# 2.3 Gs(s) = 1 / (1 + s·Tg)
num_gs = [1]
den_gs = [Tg, 1]
Gs = lti(num_gs, den_gs)

# 2.4 Gj(s) = 1 / (2 H·s + Kd)    ← your “power system” / generator
num_gj = [50/3600]
den_gj = [2*H, Kd*50]
Gj = lti(num_gj, den_gj)

# === 3) CASCADE Gt → Gc → Gs to get the “nominal” Gp_cascade ===
num_12 = np.convolve(Gt.num, Gc.num)
den_12 = np.convolve(Gt.den, Gc.den)
num_123 = np.convolve(num_12, Gs.num)
den_123 = np.convolve(den_12, Gs.den)


# now cascade the PSS:
num_123p = np.convolve(num_123, G_pss.num)
den_123p = np.convolve(den_123, G_pss.den)


# === 4) ADD YOUR EXTRA GAIN Dg IN PARALLEL: Gp = Gp_cascade + Dg ===
#    Gp_cascade = num_123/den_123,  Dg = Dg/1  ⇒
#    Gp_num = num_123 + Dg·den_123,  Gp_den = den_123
num_gp = np.polyadd(num_123p, Dg * den_123p)
den_gp = den_123p

num_fcr = 4 * num_gp*15
den_fcr = den_gp
Gp = lti(num_fcr, den_fcr)



# === 5) FORM THE CLOSED-LOOP: unity-feedback around Gp (forward) and Gj (plant) ===
#    CL TF  = Gj / [1 + Gp·Gj]
#    ⇒ common denom = den_gj*den_gp,  feedback adds num_gj*num_gp 
num_cl = np.convolve(Gj.num, den_fcr)
den_cl = np.polyadd(
    np.convolve(Gj.den, den_fcr),
    np.convolve(Gj.num, num_fcr)
)
sys_cl_pss = lti(num_cl, den_cl)

# === 6) SIMULATE A STEP DISTURBANCE THROUGH THE LOOP ===
t = time_values      # 0…50 s
d = disturbance_values              # unit-step disturbance
t_out_pss, y_out_pss, _ = lsim(sys_cl_pss, U=d, T=t)

# === 7) PLOT ===
plt.figure(figsize=(8,4))
plt.plot(t_out, -y_out, label='Δf to disturbance d')
plt.plot(t_out_pss, -y_out_pss, label='Δf to disturbance d with PSS', linestyle='--')
plt.plot(t, freq_values, label='From TOPS')
plt.xlabel('Time [s]')
plt.ylabel('Δf [Hz]')
plt.title('Closed-Loop Response with $G_p$ in place of FCR-unit')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


