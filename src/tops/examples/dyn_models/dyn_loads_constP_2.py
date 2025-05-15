import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
import pandas as pd
importlib.reload(dps)
import numpy as np
import scipy
from tops.examples.user_models.user_lib.MyTools.Statistics import plot_normalDistribution, plotFFT, CompareAutocorrelations, EulerMaryama, getAutocorrelationData, MakeEulerMaryamaList
from tops.examples.user_models.user_lib.MyTools.Functions import *





from tops.dyn_models.utils import DAEModel


class ConstPowerLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def apparent_power_injections(self, x, v):
        s_inj = -(self.par['P'] + 1j*self.par['Q'])/self.sys_par['s_n']
        return self.bus_idx_red['terminal'], s_inj
    

if __name__ == '__main__':

    # Load model
    #import tops.ps_models.k2a as model_data
    import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    #import src.tops.ps_models.sm_load as model_data
    #import src.tops.ps_models.n45_tuned as model_data
    importlib.reload(model_data)
    model = model_data.load()

    #Creating list with indexes for loads in n45_tuned except for the first load, but with index zero
    # loadIndexes=[i for i in range(0, len(model['loads']))]
    # #removing idx 1:
    # loadIndexes.remove(29)

    loadIndexesk2a=[i for i in range(0, len(model['loads']))]
    loadIndexesk2a.remove(1)


    #Summing up all the loads in the model
    sumLoads=0
    sumReactiveLoads=0


    # model['loads'] = { # 'ConstPowerLoad': model['loads']}
    #     'Load': [model['loads'][ix] for ix in loadIndexes],
    #     'ConstPowerLoad': [model['loads'][ix] for ix in [0, 29]]}

    model['loads'] = {  # 'ConstPowerLoad': model['loads']}
        'Load': [model['loads'][ix] for ix in loadIndexesk2a],
        'ConstPowerLoad': [model['loads'][ix] for ix in [0, 1]]}

    user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})
    hasattr(getattr(user_mdl_lib, 'loads'), 'ConstPowerLoad')

    # Power system model
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)
    #ps.loads['ConstPowerLoad'].par['Q'] = 100.0
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    v = ps.solve_algebraic(0, ps.x0, ps.v_0)
    max(abs(v - ps.v_0))

    for i in range(0, len(ps.loads['Load'].par['P'])):
        sumLoads+=ps.loads['Load'].par['P'][i]
        sumReactiveLoads+=ps.loads['Load'].par['Q'][i]

    print("Sum of all loads in the model: ", sumLoads)
    print("Sum of all reactive loads in the model: ", sumReactiveLoads)

    t_end = 100
    x_0 = ps.x_0.copy()
    time_step=0.02

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v0, t_end, max_step=time_step)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

    line_mdl = ps.lines['Line']

    length = 0
    def p_bus_7(x, v):
        line_p = line_mdl.p_from(x, v)
        line_p_to = line_mdl.p_to(x, v)


        return - ps.s_n*(line_p_to[1] + line_p[2] + line_p[3])


    def q_bus_7(x, v):
        line_q = line_mdl.q_from(x, v)
        line_q_to = line_mdl.q_to(x, v)

        return - ps.s_n*(line_q_to[1] + line_q[2] + line_q[3])

    # def p_bus_5321(x, v):
    #     line_p = line_mdl.p_from(x, v)
    #     line_p_to = line_mdl.p_to(x, v)
    #
    #     return - ps.s_n*(line_p_to[44])
    #
    # def q_bus_5321(x, v):
    #     line_q = line_mdl.q_from(x, v)
    #     line_q_to = line_mdl.q_to(x, v)
    #     return - ps.s_n*(line_q_to[44])

    omega=2*np.pi*1  # Example Frequency
    vec1=np.arange(0,300000, 0.02)
    vec2=np.arange(1, 3000, 10)
    vec3=np.arange(1, 3000, 0.1)
    idx1=0
    idx2=0
    idx3=0
    #p_bus_5321(ps.x0, ps.v0)
    p_bus_7(ps.x0, ps.v0)
    print(ps.loads['ConstPowerLoad'].par['P'])
    p_0=ps.loads['ConstPowerLoad'].par['P'][0]
    q_0=ps.loads['ConstPowerLoad'].par['Q'][0]

    967/900

    # line_mdl.par[3]
    print(ps.loads['ConstPowerLoad'].par)

    ###############################

    theta=0.05
    mu_p=p_0
    mu_q=q_0
    sigma_p=50.624       #1% of total load at the beginning
    sigma_q=10.87

    ################################
    # Run simulation

    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        if t>vec1[idx1]:
            #Setting load as sine signal
            #ps.loads['ConstPowerLoad'].par['P'][0]=p_0+600*np.sin(1/15*omega*t)
            ps.loads['ConstPowerLoad'].par['P'][0]=EulerMaryama(theta, mu_p, sigma_p, time_step, ps.loads['ConstPowerLoad'].par['P'][0])
            ps.loads['ConstPowerLoad'].par['Q'][0]=EulerMaryama(theta, mu_q, sigma_q, time_step, ps.loads['ConstPowerLoad'].par['Q'][0])
            idx1+=1


        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        # dx = ps.ode_fun(0, ps.x_0)
        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['v'].append(v.copy())
        res['p_bus_7'].append(p_bus_7(x, v).copy())
        res['q_bus_7'].append(q_bus_7(x, v).copy())
        res['load'].append(ps.loads['Load'].par['P'].copy())
        # res['p_bus_5321'].append(p_bus_5321(x, v).copy())
        # res['q_bus_5321'].append(q_bus_5321(x, v).copy())
        res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50) / (2 * np.pi)) + 50)
        res['iterations'].append(ps.it_prev)
    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_1.csv')
    # data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_3.csv')
    freq2 = data['Frequency [Hz*4]'][2:(len(res['t']) + 2)] / 4
    time2 = data['Time [s]'][2:(len(res['t']) + 2)]
    avg_freq = [np.mean(freq) for freq in res['freq']]
    # print("Simulated Mean Load: ", np.mean(res['p_bus_5321']))
    # print("Simulated Variance Load: ", np.var(res['p_bus_5321']))
    print("Simulated Mean Load: ", np.mean(res['p_bus_7']))
    print("Simulated Variance Load: ", np.var(res['p_bus_7']))
    #Finding out when exp^(-2*theta*t) is less than 0.01
    shortestTime=np.log(0.01)/(-2*theta)
    print("Shortest simulation time needed to use this theta:", shortestTime)
    LongtermCalcVar = sigma_p ** 2 / (2 * theta)
    LongTermCalcVarTimeDependent = sigma_p ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * t_end))
    print("Long term variance: ", LongtermCalcVar)
    print("Long term variance time dependent: ", LongTermCalcVarTimeDependent)


    fig=plt.figure()

    plt.plot(res['t'], avg_freq, label='Simulated freq.')
    plt.plot(time2, freq2, label="Measured Frequency")
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency')
    plt.legend()

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load']), label="Other load")
    plt.plot(res['t'], np.abs(res['p_bus_7']), label="Real Power at bus 7")
    plt.plot(res['t'], np.abs(res['q_bus_7']), label="Reactive Power at bus 7")
    # plt.plot(res['t'], np.abs(res['p_bus_5321']), label="Real Power at bus 45")
    # plt.plot(res['t'], np.abs(res['q_bus_5321']), label="Reactive Power at bus 45")
    plt.legend()
    plt.xlabel('Real and Reactive power at bus 45')
    plt.ylabel('MW and MVAR')

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['v']))
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage magnitude')


    # totalPSD=plot_frequency_pdf("C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData", "Frequency [Hz*4]")
    # mean,stdDev=fitNormalDistribution(totalPSD)
    # print("Mean and standard deviation of total frequency data:", mean, stdDev)
    #
    # '''Plotting Normal Distribution Functions'''
    #
    # fig1, ax = plt.subplots()
    # x = np.linspace(mean - 3 * stdDev, mean + 3 * stdDev, 1000)
    # y = (1 / (stdDev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stdDev) ** 2)
    #
    #
    # ax.plot(x, y, label='Normal Distribution')
    # plot_normalDistribution(avg_freq, ax=ax, label='Simulated Frequency PDF')
    # plot_normalDistribution(freq2, ax=ax, label='Real Data Frequency PDF')
    # plot_normalDistribution(totalPSD, ax=ax, label='Total Frequency PDF')
    # plt.legend()


    '''Plotting Autocorrelation Functions'''
    CompareAutocorrelations(avg_freq, freq2, 10000, timestepSim=time_step, label1='Simulated Frequency Autocorrelation', label2='Real Data Frequency Autocorrelation')


    # p_fft_freq, p_fft_mag=makeFFT((np.abs(res['p_bus_7'])-np.mean(np.abs(res['p_bus_7'])))/600, time_step)
    # p_fft_freq_rad = 2 * np.pi * p_fft_freq
    # p_fft_freq, p_fft_mag = makeFFT((np.abs(res['p_bus_5321']) - np.mean(np.abs(res['p_bus_5321']))) / 600, time_step)
    # p_fft_freq_rad = 2 * np.pi * p_fft_freq
    #
    # f_plot, fft_mag_plot=makeFFT(avg_freq-np.mean(avg_freq), time_step)
    # welch_freq, welch_mag=makeWelchFFT((np.abs(res['p_bus_5321'])-np.mean(np.abs(res['p_bus_5321'])))/600, time_step)
    # welch_freq_rad = 2 * np.pi * welch_freq

    p_fft_freq, p_fft_mag = makeFFT((np.abs(res['p_bus_7']) - np.mean(np.abs(res['p_bus_7']))) / 51.4, time_step)
    p_fft_freq_rad = 2 * np.pi * p_fft_freq

    f_plot, fft_mag_plot = makeFFT(avg_freq - np.mean(avg_freq), time_step)
    welch_freq, welch_mag = makeWelchFFT((np.abs(res['p_bus_7']) - np.mean(np.abs(res['p_bus_7']))) / 51.4, time_step)
    welch_freq_rad = 2 * np.pi * welch_freq


    def func(x,a):
        return 1/(a*x+1)
    popt_fft, pcovv_fft = scipy.optimize.curve_fit(func, p_fft_freq_rad, p_fft_mag)
    popt_welch, pcovv_welch = scipy.optimize.curve_fit(func, welch_freq_rad, welch_mag)
    print("Timedelay constant of fitted low-pass filter, fft simulations:", popt_fft)
    print("Timedelay constant of fitted low-pass filter, welch simulations:", popt_welch)


    ###PLOTTING###
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

    w, mag, w_k2a, mag_k2a, w_req, mag_req=calculateSystemTf(model_data)

    plt.figure()

    #plt.semilogx(w_k2a, mag_k2a, label="G_req-n magnitude K2A", color='blue')  # Magnitude in absolute terms
    #plt.semilogx(w_k2a, 0.9 * mag_k2a, label="G_req-n magnitude K2A with reduction factor", color='green')  # Magnitude in absolute terms
    plt.semilogx(w, mag, label="G_req-n magnitude NSA", color='blue')  # Magnitude in absolute terms
    plt.semilogx(w, 0.9 * mag, label="G_req-n magnitude NSA with reduction factor", color = 'green')  # Magnitude in absolute terms
    plt.semilogx(w_req, mag_req, label="Requirement magnitude", linestyle='dashed', color='darkred')  # Magnitude in absolute terms

    plt.semilogx(welch_freq_rad, 1/welch_mag, label="Welch of P at bus 7", color='black')
    plt.semilogx(p_fft_freq_rad, 1/p_fft_mag, label="FFT of P at bus 7", color='red')
    #plt.semilogx(welch_freq_rad, 1/welch_mag, label="FFT of P at bus 7 Welch", color='red')
    plt.semilogx(p_fft_freq_rad, 1/func(p_fft_freq_rad, *popt_welch), label="Best low-pass filter describing Welch", color='orange')
    plt.yscale('log')


    plt.title('Bode Magnitude Plot')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (abs)')
    # plt.xlim(0.01, 10)
    #plt.ylim(0, 15)
    plt.legend()
    plt.xlim(0.001, 10)
    plt.grid()
    plt.show()



    # _,systemTF=returnSystemTf(model_data)
    # frequenciesDisturbance, disturbanceMag, systemTFMag=calculateDisturbanceMagnitude(model_data, np.abs(res['p_bus_7']), avg_freq, time_step, time_step)
    # compareFreqandDisturbance(disturbanceMag, systemTFMag, frequenciesDisturbance)

