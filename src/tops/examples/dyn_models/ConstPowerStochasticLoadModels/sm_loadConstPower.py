
import sys
import time
import importlib
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.dyn_models.utils import DAEModel

from tops.examples.user_models.user_lib.MyTools.Statistics import (
    plot_normalDistribution, plotFFT, CompareAutocorrelations, 
    EulerMaryama, getAutocorrelationData, MakeEulerMaryamaList
)
from tops.examples.user_models.user_lib.MyTools.Functions import plot_frequency_pdf, fitNormalDistribution,calculateSystemTf, makeFFT, makeWelchFFT


# -----------------------------
# Custom Load Class Definition
# -----------------------------
class ConstPowerLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = self.par = data
        self.n_units = len(data)
        self.bus_idx = np.zeros(self.n_units, dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.zeros(self.n_units, dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def apparent_power_injections(self, x, v):
        s_inj = -(self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], s_inj


if __name__ == '__main__':
    # ----------------------------------
    # Load model and initialize system
    # ----------------------------------
    # import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    #import tops.ps_models.k2a as model_data
    import tops.ps_models.sm_load as model_data
    importlib.reload(model_data)
    model = model_data.load()

    model['loads'] = {
        #'Load': [model['loads'][ix] for ix in [0, 2]],
        'ConstPowerLoad': [model['loads'][ix] for ix in [0, 1]]
    }

    user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    v = ps.solve_algebraic(0, ps.x0, ps.v_0)
    max(abs(v - ps.v_0))



    # ----------------------------------
    # Setup simulation parameters
    # ----------------------------------
    t_end = 100
    x_0 = ps.x_0.copy()
    time_step = 0.02
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v0, t_end, max_step=time_step)

    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
    line_mdl = ps.lines['Line']

    def p_bus_2(x, v):
        return -ps.s_n * (line_mdl.p_to(x, v)[0])
    
    def q_bus_2(x, v):
        return -ps.s_n * (line_mdl.q_to(x, v)[0])
    

    # ----------------------------------
    # Noise and stochastic config
    # ----------------------------------
    omega = 2 * np.pi * 1
    vec1 = np.arange(0, 300000, 0.02)
    idx1 = 0

    p_0 = ps.loads['ConstPowerLoad'].par['P'][0]
    q_0 = ps.loads['ConstPowerLoad'].par['Q'][0]

    theta = 0.05
    mu_p, mu_q = p_0, q_0
    sigma_p, sigma_q = 27.34, 2

    listGenInitial=list(ps.gen['GEN'].par['P'])
    print(listGenInitial)

    # ----------------------------------
    # Simulation loop
    # ----------------------------------
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / t_end * 100))


        if t > 1:
             
            ps.loads['ConstPowerLoad'].par['P'][0] = EulerMaryama(theta, mu_p, sigma_p, time_step, ps.loads['ConstPowerLoad'].par['P'][0])
            # ps.loads['ConstPowerLoad'].par['Q'][0] = EulerMaryama(theta, mu_q, sigma_q, time_step, ps.loads['ConstPowerLoad'].par['Q'][0])
            # idx1 += 1
            # print("P_m1", listGenInitial[0])
            # ps.gen['GEN'].set_input('P_m', listGenInitial[1], 1)
            # print("P_m2", listGenInitial[1])
            # ps.gen['GEN'].set_input('P_m', listGenInitial[2], 2)
            # print("P_m3", listGenInitial[2])
            # ps.gen['GEN'].set_input('P_m', listGenInitial[3], 3)
            #ps.loads['ConstPowerLoad'].par['P'][0]=p_0*1

        result = sol.step()
        x, v, t = sol.y, sol.v, sol.t

        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['v'].append(v.copy())
        res['p_bus_2'].append(p_bus_2(x, v).copy())
        res['q_bus_2'].append(q_bus_2(x, v).copy())
        res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50)) + 50)
        res['gen_power'].append(ps.gen['GEN'].P_e(x,v).copy())
        res['iterations'].append(ps.it_prev)

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # ----------------------------------
    # Data & Visualization
    # ----------------------------------
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_1.csv')
    freq2 = data['Frequency [Hz*4]'][2:(len(res['t']) + 2)] / 4
    time2 = data['Time [s]'][2:(len(res['t']) + 2)]
    avg_freq = [np.mean(freq) for freq in res['freq']]

    # Plot frequency comparison
    plt.figure()
    plt.plot(res['t'], avg_freq, label='Simulated freq.')
    plt.plot(time2, freq2, label='Measured Frequency')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot load data
    plt.figure()
    plt.plot(res['t'], np.abs(res['p_bus_2']), label="Real Power at bus 2")
    plt.plot(res['t'], np.abs(res['q_bus_2']), label="Reactive Power at bus 2")


    plt.xlabel('Real and Reactive power at bus 2')
    plt.ylabel('MW and MVAR')
    plt.legend()

    #Plot Gen Power
    plt.figure()
    plt.plot(res['t'], np.abs(res['gen_power']), label="Generator Power")
    plt.xlabel('Time [s]')
    plt.ylabel('Generator Power [MW]')
    plt.legend()


    # Plot voltage
    plt.figure()
    plt.plot(res['t'], np.abs(res['v']))
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage magnitude')

    plt.show()

run_code = input("Plot Probability density functions? (y/n): ").strip().lower()

if run_code == 'y':
    totalPSD = plot_frequency_pdf("C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData", "Frequency [Hz*4]")
    mean, stdDev = fitNormalDistribution(totalPSD)
    print("Mean and standard deviation of total frequency data:", mean, stdDev)

    '''Plotting Normal Distribution Functions'''
    fig1, ax = plt.subplots()
    x = np.linspace(mean - 3 * stdDev, mean + 3 * stdDev, 1000)
    y = (1 / (stdDev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stdDev) ** 2)

    ax.plot(x, y, label='Normal Distribution')
    plot_normalDistribution(avg_freq, ax=ax, label='Simulated Frequency PDF')
    plot_normalDistribution(freq2, ax=ax, label='Real Data Frequency PDF')
    plot_normalDistribution(totalPSD, ax=ax, label='Total Frequency PDF')
    plt.legend()
    plt.show()

else:
    print("Skipping the code execution.")


'''Plotting Autocorrelation Functions'''
run_codeAutocorrelation = input("Plot Autocorrelations? (y/n): ").strip().lower()
if run_codeAutocorrelation == 'y':
    CompareAutocorrelations(avg_freq, freq2, 10000, timestepSim=time_step, label1='Simulated Frequency Autocorrelation', label2='Real Data Frequency Autocorrelation')
else:
    print("Skipping the code execution.")

run_codeFFT = input("Plot Load FFT? (y/n): ").strip().lower()
if run_codeFFT == 'y':
        w, mag, w_k2a, mag_k2a, w_req, mag_req=calculateSystemTf(model_data)
        p_fft_freq_rad, p_fft_mag = makeFFT((np.abs(res['p_bus_2']) - np.mean(np.abs(res['p_bus_2']))) / 60, time_step)

        f_plot, fft_mag_plot = makeFFT(avg_freq - np.mean(avg_freq), time_step)
        #welch_freq_rad, welch_mag = makeWelchFFT((np.abs(res['p_bus_2']) - np.mean(np.abs(res['p_bus_2']))) / 51.4, time_step)
        plt.figure()

        def func(x,a):
            return 1/(a*x+1)
        popt_fft, pcovv_fft = scipy.optimize.curve_fit(func, p_fft_freq_rad, p_fft_mag)
        #popt_welch, pcovv_welch = scipy.optimize.curve_fit(func, welch_freq_rad, welch_mag)
        print("Timedelay constant of fitted low-pass filter, fft simulations:", popt_fft)
        #print("Timedelay constant of fitted low-pass filter, welch simulations:", popt_welch)

        plt.semilogx(w_k2a, mag_k2a, label="G_req-n magnitude K2A", color='blue')  # Magnitude in absolute terms
        plt.semilogx(w_k2a, 0.9 * mag_k2a, label="G_req-n magnitude K2A with reduction factor", color='green')  # Magnitude in absolute terms
        plt.semilogx(w_req, mag_req, label="Requirement magnitude", linestyle='dashed', color='darkred')  # Magnitude in absolute terms

        #plt.semilogx(welch_freq_rad, 1/welch_mag, label="Welch of P at bus 2", color='black')
        plt.semilogx(p_fft_freq_rad, 1/p_fft_mag, label="FFT of P at bus 2", color='red')
        #plt.semilogx(welch_freq_rad, 1/func(welch_freq_rad, *popt_welch), label="Best low-pass filter describing Welch", color='orange')
        plt.yscale('log')


        plt.title('Bode Magnitude Plot')
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Magnitude (abs)')
        plt.legend()
        plt.xlim(0.001, 10)
        plt.grid()
        plt.show()
else:
    print("Skipping the code execution until line 217.")


runPSD = input("Plot PSD? (y/n): ").strip().lower()
if runPSD == 'y':
    omega = np.linspace(-5, 10, 5000)

    f2, Pxx2 = scipy.signal.periodogram(freq2, 50)
    Pxx2_dB = 20 * np.log10(abs(Pxx2))
    plt.plot(f2, Pxx2_dB, label='Real Data pelton')

    ###Simulated Frequency Data PSD###
    f1, Pxx1 = scipy.signal.periodogram(avg_freq, 50)
    Pxx1_dB = 20 * np.log10(abs(Pxx1))
    plt.plot(f1, Pxx1_dB, label='Simulated Data')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB/Hz]')
    plt.xlim(0, 2)
    plt.legend()
    plt.show()
else:
    print("Skipping the code execution.")


