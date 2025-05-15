
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

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from tops.examples.user_models.user_lib.MyTools.Statistics import (
    plot_normalDistribution, plotFFT, CompareAutocorrelations, 
    EulerMaryama, getAutocorrelationData, MakeEulerMaryamaList
)
from tops.examples.user_models.user_lib.MyTools.Functions import plot_frequency_pdf, fitNormalDistribution,calculateSystemTf, makeFFT, makeWelchFFT, plotACFs
import wesanderson as ws

colors=ws.film_palette('Darjeeling limited')

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
    import tops.ps_models.n45_tuned as model_data

    importlib.reload(model_data)
    model = model_data.load()

    loadIndexes=[i for i in range(0, len(model['loads']))]
    #removing idx 1:
    loadIndexes.remove(29)



    model['loads'] = { # 'ConstPowerLoad': model['loads']}
        'Load': [model['loads'][ix] for ix in loadIndexes],
        'ConstPowerLoad': [model['loads'][ix] for ix in [0, 29]]}


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

    def p_bus_5321(x, v):
        return -ps.s_n * (line_mdl.p_to(x, v)[1] + line_mdl.p_from(x, v)[2] + line_mdl.p_from(x, v)[3])

    def q_bus_5321(x, v):
        return -ps.s_n * (line_mdl.q_to(x, v)[1] + line_mdl.q_from(x, v)[2] + line_mdl.q_from(x, v)[3])

    
    def p_bus_5321(x, v):
        line_p = line_mdl.p_from(x, v)
        line_p_to = line_mdl.p_to(x, v)
        return - ps.s_n*(line_p_to[44])
    
    def q_bus_5321(x, v):
        line_q = line_mdl.q_from(x, v)
        line_q_to = line_mdl.q_to(x, v)
        return - ps.s_n*(line_q_to[44])
    

    # ----------------------------------
    # Noise and stochastic config
    # ----------------------------------
    omega = 2 * np.pi * 1
    vec1 = np.arange(0, 300000, 0.02)
    idx1 = 0

    p_0 = ps.loads['ConstPowerLoad'].par['P'][0]
    q_0 = ps.loads['ConstPowerLoad'].par['Q'][0]

    theta = 0.008
    mu_p, mu_q = p_0, q_0
    sigma_p, sigma_q = 400, 2

    listGenInitial=list(ps.gen['GEN'].par['P'])
    print(ps.f_n)
    print(ps.sys_data['f_n'])
    # ----------------------------------
    # Simulation loop
    # ----------------------------------

    eventflag=True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / t_end * 100))


        if t > vec1[idx1]:
            ps.loads['ConstPowerLoad'].par['P'][0] =  EulerMaryama(theta, mu_p, sigma_p, time_step, ps.loads['ConstPowerLoad'].par['P'][0])
            idx1 += 1
            # ps.gen['GEN'].set_input('P_m', listGenInitial[0], 0)
            # ps.gen['GEN'].set_input('P_m', listGenInitial[1], 1)
            # ps.gen['GEN'].set_input('P_m', listGenInitial[2], 2)
            # ps.gen['GEN'].set_input('P_m', listGenInitial[3], 3)
            # 



        # result = sol.step()
        # x, v, t = sol.y, sol.v, sol.t
        # if idx1 == 3:
        #     oldSpeed = x[ps.gen['GEN'].state_idx['speed']].copy()

        # if t > 5:
        #     for i, idx in enumerate(ps.gen['GEN'].state_idx['speed']):
        #         x[idx] = oldSpeed[i] - 0.072


        result = sol.step()
        x, v, t = sol.y, sol.v, sol.t

        # if t>3 and eventflag:
        #     eventflag=False
        #     ps.gov['HYGOV'].int_par['bias'][0]+= 0.01
        #     ps.gov['HYGOV'].int_par['bias'][1] += 0.01
        #     ps.gov['HYGOV'].int_par['bias'][2] += 0.01
        #     ps.gov['HYGOV'].int_par['bias'][3] += 0.01

        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['v'].append(v.copy())
        res['p_bus_5321'].append(p_bus_5321(x, v).copy())
        res['q_bus_5321'].append(q_bus_5321(x, v).copy())
        res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50) / (2 * np.pi)) + 50)
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

    #Finding out how big a portion of the datapoints falls outside of 49.9-50.1 Hz
    count = 0
    for i in range(len(avg_freq)):
        if avg_freq[i] < 49.9 or avg_freq[i] > 50.1:
            count += 1
    print("Percentage of points outside 49.9-50.1 Hz:", count / len(avg_freq) * 100)

    # Calculate the center of inertia frequency for each timestep
    f_coi = []  # List to store the center of inertia frequency for each timestep
    gen_inertia = ps.gen['GEN'].par['H']  # Generator inertias

    # Iterate over each timestep
    for timestep_freqs in res['freq']:
        # Calculate the weighted sum of generator frequencies
        numerator = np.sum(gen_inertia * timestep_freqs)
        denominator = np.sum(gen_inertia)
        f_coi.append(numerator / denominator)  # Append the center of inertia frequency

    # Convert f_coi to a numpy array for easier handling
    f_coi = np.array(f_coi)

    # Plot frequency comparison
    plt.figure()
    # for gen_idx, gen_freq in enumerate(zip(*res['freq']), start=1):  # Transpose res['freq'] to iterate over generators
    #     plt.plot(res['t'], gen_freq, label=f"Generator {gen_idx}", color=colors[gen_idx - 1])
    plt.plot(res['t'], avg_freq, label='Simulated Frequency', color=colors[0])
    plt.plot(time2, freq2, label='Measured Frequency', color=colors[4])
    plt.plot(res['t'], f_coi, label='Center of Inertia Frequency', linestyle='dashed')
    # Add labels, title, and legend
    plt.title('Frequency of Each Generator')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency[Hz]')
    plt.legend()

    # Plot load data
    plt.figure()
    plt.plot(res['t'], np.abs(res['p_bus_5321']), label="Real Power at bus 7", color=colors[3])
    #plt.plot(res['t'], np.abs(res['q_bus_5321']), label="Reactive Power at bus 7")

    plt.xlabel('Real and Reactive power at bus 7')
    plt.ylabel('MW and MVAR')
    plt.legend()

    #Plot Gen Power
    plt.figure()
    plt.plot(res['t'], np.abs(res['gen_power']), label="Generator Power")
    plt.xlabel('Time [s]')
    plt.ylabel('Generator Power [MW]')
    plt.legend()

    #Extract voltage at bus 7 and 9
    # v_bus_7 = np.array([v[ps.bus_idx_red[6]] for v in res['v']])
    # v_bus_9 = np.array([v[ps.bus_idx_red[8]] for v in res['v']])

    # # Plot voltage
    # plt.figure()
    # plt.plot(res['t'], np.abs(v_bus_7), label='Bus 7 voltage', color=colors[1])
    # plt.plot(res['t'], np.abs(v_bus_9), label='Bus 9 voltage', color=colors[3])
    # #plt.plot(res['t'], np.abs(res['v']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Bus voltage magnitude')
    plt.legend()
    plt.show()

    run_code = input("Plot Probability density functions? (y/n): ").strip().lower()

    if run_code == 'y':
        totalPSD = plot_frequency_pdf("C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData", "Frequency [Hz*4]")
        mean, stdDev = fitNormalDistribution(totalPSD)
        print("Mean and standard deviation of total frequency data:", mean, stdDev)

        '''Plotting Normal Distribution Functions'''
        fig1, ax = plt.subplots()
        # x = np.linspace(mean - 3 * stdDev, mean + 3 * stdDev, 1000)
        # y = (1 / (stdDev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stdDev) ** 2)

        #ax.plot(x, y, label='Normal Distribution of total frequency data')
        plot_normalDistribution(avg_freq, ax=ax, label='Simulated Frequency PDF')
        plot_normalDistribution(freq2, ax=ax, label='Real Data Frequency PDF')
        #plot_normalDistribution(totalPSD, ax=ax, label='Total Frequency PDF')
        plt.xlabel('Frequency [Hz]')
        plt.legend()
        plt.show()

    else:
        print("Skipping the code execution.")


    '''Plotting Autocorrelation Functions'''
    run_codeAutocorrelation = input("Plot Autocorrelations? (y/n): ").strip().lower()


    if run_codeAutocorrelation == 'y':
        plotACFs(avg_freq)

    else:
        print("Skipping the code execution.")

    run_codeFFT = input("Plot Load FFT? (y/n): ").strip().lower()
    if run_codeFFT == 'y':
            # Unpack the lists returned by calculateSystemTf
        w_list, mag_list, w_k2a_list, mag_k2a_list, w_req_list, mag_req_list, w_reqs, actualReq = calculateSystemTf(model_data)

        # Compute FFT data
        p_fft_freq_rad, p_fft_mag = makeFFT(
            (np.abs(res['p_bus_5321']) - np.mean(np.abs(res['p_bus_5321'])) + (np.abs(res['p_bus_9']) - np.mean(np.abs(res['p_bus_9'])))) / 51.4,
            time_step
        )

        # Fit a low-pass filter to the FFT data
        def func(x, a):
            return 1 / (a * x + 1)

        popt_fft, pcovv_fft = scipy.optimize.curve_fit(func, p_fft_freq_rad, p_fft_mag)
        print("Timedelay constant of fitted low-pass filter, FFT simulations:", popt_fft)



        # Create a new figure for the plot
        plt.figure()

        # Plot all elements in w_k2a_list and mag_k2a_list
        for idx, (w_k2a, mag_k2a) in enumerate(zip(w_k2a_list, mag_k2a_list), start=1):
            plt.semilogx(w_k2a, mag_k2a, label=f"G_req-n magnitude K2A {idx}", color='blue', alpha=0.5)

        # Plot all elements in w_req_list and mag_req_list
        for idx, (w_req, mag_req) in enumerate(zip(w_req_list, mag_req_list), start=1):
            plt.semilogx(w_req, mag_req, label=f"Requirement magnitude {idx}", linestyle='dashed', color='darkred', alpha=0.7)

        # Plot the FFT data
        plt.semilogx(p_fft_freq_rad, 1/p_fft_mag, label="FFT of P at bus 7", color=colors[3])

        #Plot TSO filter
        plt.semilogx(w_reqs, 1/actualReq, label="1/D(s)", color=colors[4], linestyle='dashed')

        # Plot the fitted low-pass filter
        plt.semilogx(p_fft_freq_rad, 1/abs(func(p_fft_freq_rad, *popt_fft)), label="Best low-pass filter describing FFT", color=colors[2], linestyle='dotted')

        # Add labels, title, and legend
        plt.yscale('log')
        plt.title('Bode Magnitude Plot with FFT')
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


