import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time

import numpy as np
import pandas as pd
from scipy import signal

import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import importlib
from examples.user_models.user_lib.MyTools.Statistics import plot_normalDistribution, plotFFT, CompareAutocorrelations, EulerMaryama

if __name__ == '__main__':

    # Load model
    #import tops.ps_models.k2a as model_data
    import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()


    model['vsc'] = {'VSC': [
        ['name',    'T_pll',    'T_i',   'bus',  'P_K_p',    'P_K_i',    'Q_K_p',    'Q_K_i',    'P_setp',   'Q_setp' ],
        ['VSC1',      0.1,        0.1,    'B8',   0.03,         1,         0.03,         1,         0,            0,]]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.ode_fun(0, ps.x_0))))

    x0 = ps.x_0
    v0 = ps.v_0

    time_step = 0.02
    t_end = 100
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=time_step)

    # Initialize simulation
    t = 0
    res = defaultdict(list)

    omega=2*np.pi*1  # Example Frequency
    theta=0.1
    mu=0
    sigma=25

    length=0

    t_0 = time.time()
    vec=range(1,101)
    # Run simulation
    idx=0
    P=ps.vsc['VSC'].par['P_setp']
    Q=ps.vsc['VSC'].par['Q_setp']


    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        if t>vec[idx]:
        #if t>0:
            sine_disturbance=36*np.sin(omega*t)

            #Modelling load as Euler-Maryama solution of Ornstein Uhlenbeck process#

            ps.vsc['VSC'].set_input('P_setp', P, 0)
            P=EulerMaryama(theta, mu, sigma, time_step, P)
            Q=EulerMaryama(theta, mu, sigma, time_step, Q)
            #ps.loads['LoadAsCurrent'].set_input('P_setp', 1000+np.random.normal(0, 1, 1), 0)
            #ps.loads['LoadAsCurrent'].set_input('whiteNoiseP', np.random.normal(0, 500), 0)
            #ps.vsc['VSC'].set_input('Q_setp', disturbance)
            idx+=1



        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t
        v = sol.v
        #print(abs(v)[0])

        dx = ps.ode_fun(0, ps.x_0)
        length+=1

        # Store result
        res['t'].append(sol.t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50) / (2 * np.pi)) + 50.0)
        res['VSC_P'].append(ps.vsc['VSC'].P(x, v).copy()[0])
        res['VSC_P_setp'].append(ps.vsc['VSC'].P_setp(x, v).copy())

        res['VSC_Q'].append(ps.vsc['VSC'].Q(x, v).copy()[0])
        res['VSC_Q_setp'].append(ps.vsc['VSC'].Q_setp(x, v).copy())
        res['current'].append((ps.vsc['VSC'].I_inj(x, v)))

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # plt.figure()
    # plt.plot(res['t'], res['gen_speed'])
    # plt.show()

    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_2_pelton.csv')
    #data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_3.csv')
    freq2 = data['Frequency [Hz*4]'][2:30000] / 4
    time2 = data['Time [s]'][2:30000]
    avg_freq = [np.mean(freq) for freq in res['freq']]
    fig, ax = plt.subplots(4)
    ax[0].plot(res['t'], res['VSC_P_setp'], label='P setp')
    ax[0].plot(res['t'], res['VSC_P'], label='P')
    ax[0].legend()
    ax[1].plot(res['t'], res['VSC_Q_setp'], label='Q setp')
    ax[1].plot(res['t'], res['VSC_Q'], label='Q')
    ax[1].legend()
    ax[2].plot(res['t'], avg_freq, label='Simulated Frequency')
    ax[2].plot(res['t'], freq2[0:length], label='Real Data Frequency')
    ax[2].legend()
    ax[3].plot(res['t'], res['current'], label='Current')
    ax[3].legend()
    plt.show()

    ###Load Data PDF and Autocorrelation###

    plot_normalDistribution(res['VSC_P'], label='VSC_P Normal Distribution')
    fig, ax = plt.subplots()
    plot_normalDistribution(avg_freq, ax=ax, label='Simulated Frequency PDF')
    plot_normalDistribution(freq2, ax=ax, label='Real Data Frequency PDF')
    plt.legend()
    plt.show()
    CompareAutocorrelations(avg_freq, freq2, 1, label1='Simulated Frequency Autocorrelation', label2='Real Data Frequency Autocorrelation')

    #plotFFT(res['VSC_P'], timestep=5e-3, frequency_range=1)







###PeltonData Frequency PSD###
data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_2_pelton.csv')
data=pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_3.csv')
freq2 = data['Frequency [Hz*4]'][2:30000] / 4
time2 = data['Time [s]'][2:30000]
f2, Pxx2 = signal.periodogram(freq2, 50)
Pxx2_dB = 20 * np.log10(abs(Pxx2))
plt.plot(f2, Pxx2_dB, label='Real Data pelton')

###Simulated Frequency Data PSD###
f1, Pxx1 = signal.periodogram(avg_freq, 50)
Pxx1_dB = 20 * np.log10(abs(Pxx1))
plt.plot(f1, Pxx1_dB, label='Simulated Data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim(0,2)
plt.legend()
plt.show()



