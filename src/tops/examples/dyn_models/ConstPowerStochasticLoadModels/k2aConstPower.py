
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
from tops.examples.user_models.user_lib.MyTools.Functions import *
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
    #import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.k2aTuning as model_data

    #import tops.ps_models.sm_load as model_data
    importlib.reload(model_data)
    model = model_data.load()

    model['loads'] = {
        #'Load': [model['loads'][ix] for ix in [0, 2]],
        'ConstPowerLoad': [model['loads'][ix] for ix in [0, 1, 2]]
    }


    user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    v = ps.solve_algebraic(0, ps.x0, ps.v_0)
    max(abs(v - ps.v_0))
    
    df = pd.read_csv(r'C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS\Figures\disturbance\disturbance.csv')

    disturbance_values = df['disturbance'].tolist()
    freq_values = df['avg_freq_dev'].tolist()





    # ----------------------------------
    # Setup simulation parameters
    # ----------------------------------
    t_end = 10
    x_0 = ps.x_0.copy()
    time_step = 0.02
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v0,t_end, max_step=time_step)

    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
    line_mdl = ps.lines['Line']

    def p_bus_7(x, v):
        return -ps.s_n * (line_mdl.p_to(x, v)[1] + line_mdl.p_from(x, v)[2] + line_mdl.p_from(x, v)[3])

    def q_bus_7(x, v):
        return -ps.s_n * (line_mdl.q_to(x, v)[1] + line_mdl.q_from(x, v)[2] + line_mdl.q_from(x, v)[3])
    
    def p_bus_9(x,v):
        return -ps.s_n * (line_mdl.p_to(x, v)[4] + line_mdl.p_to(x, v)[5] + line_mdl.p_from(x, v)[6])
    

    # ----------------------------------
    # Noise and stochastic config
    # ----------------------------------
    omega = 2 * np.pi * 1
    vec1 = np.arange(0, 300000, 0.02)
    idx1 = 0

    p_0 = ps.loads['ConstPowerLoad'].par['P'][0]
    #p_1=ps.loads['ConstPowerLoad'].par['P'][1]
    q_0 = ps.loads['ConstPowerLoad'].par['Q'][0]

    theta = 0.002
    mu_p, mu_q = p_0, q_0
    #mu_p1=p_1
    sigma_p, sigma_q = 1.3, 0.1

    listGenInitial=list(ps.gen['GEN'].par['P'])
    print(ps.f_n)
    print(ps.sys_data['f_n'])
    # ----------------------------------
    # Simulation loop
    # ----------------------------------
    LoadChange=40
    disturbanceList=[]

    eventflag=True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / t_end * 100))


        if t > 0:
            #step=EulerMaryama(theta, mu_p, sigma_p, time_step, ps.loads['ConstPowerLoad'].par['P'][0])
            #disturbanceList.append(step-p_0)
            ps.loads['ConstPowerLoad'].par['P'][0] = EulerMaryama(theta, mu_p, sigma_p, time_step, ps.loads['ConstPowerLoad'].par['P'][0])
            idx1+=1
            #ps.loads['ConstPowerLoad'].par['Q'][0] = EulerMaryama(theta, mu_q, sigma_q, time_step, ps.loads['ConstPowerLoad'].par['Q'][0])
        #p_0+np.sin(0.1*t)*8.4 



        result = sol.step()
        x, v, t = sol.y, sol.v, sol.t

        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['v'].append(v.copy())
        res['p_bus_7'].append(p_bus_7(x, v).copy())
        res['q_bus_7'].append(q_bus_7(x, v).copy())
        res['p_bus_9'].append(p_bus_9(x, v).copy())
        res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50)) + 50)
        res['gen_power'].append(ps.gen['GEN'].P_e(x,v).copy())
        res['gen_reactive_power'].append(ps.gen['GEN'].Q_e(x,v).copy())
        res['iterations'].append(ps.it_prev)

    print('\nSimulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # ----------------------------------
    # Data & Visualization
    # ----------------------------------
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_7.csv')


    disturbance_folder = r'C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Figures/disturbance'
    os.makedirs(disturbance_folder, exist_ok=True)
    disturbance_path = os.path.join(disturbance_folder, 'disturbance.csv')

    # Save as DataFrame
    avg_freq_dev = [(np.mean(freq)-50) for freq in res['freq']]
    import pandas as pd
    df_disturbance = pd.DataFrame({'time': res['t'][:len(disturbanceList)], 'disturbance': disturbanceList, 'avg_freq_dev': avg_freq_dev[:len(disturbanceList)]})
    df_disturbance.to_csv(disturbance_path, index=False)
    print(f"Saved disturbance data to {disturbance_path}")

    avg_freq = [np.mean(freq) for freq in res['freq']]
    freq2 = data['Frequency [Hz*4]'][2:(len(res['t']) + 2)] / 4
    time2 = data['Time [s]'][2:(len(res['t']) + 2)]

    timeOutage, freqOutage=GetFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous_(1)/Outage-anonymous.xlsx', 'FI south:Frequency')

    freqOutage=freqOutage[0:2000]
    timeOutage=timeOutage[0:2000]
    # freq1=data['Value'][156800:158100]
    # time1=[]
    # for i in range(len(freq1)):
    #     time1.append(i/10)



    #Finding out how big a portion of the datapoints falls outside of 49.9-50.1 Hz
    count = 0
    for i in range(len(avg_freq)):
        if avg_freq[i] < 49.9 or avg_freq[i] > 50.1:
            count += 1
    print("Percentage of points outside 49.9-50.1 Hz:", count / len(avg_freq) * 100)

    

    # Plot frequency comparison
    plt.figure()
    # for gen_idx, gen_freq in enumerate(zip(*res['freq']), start=1):  # Transpose res['freq'] to iterate over generators
    #     plt.plot(res['t'], gen_freq, label=f"Generator {gen_idx}", color=colors[gen_idx - 1])

    plt.plot(res['t'], avg_freq, label='Simulated Frequency', color=colors[3])
    #plt.plot(res['t'], exp_freq, label='Expected Frequency', color=colors[1])
    # plt.plot(res['t'], avg_freq, label='Simulated Frequency with PSS', color=colors[3])
    # plt.plot(res['t'][:-1], [(i+50) for i in freq_values], label='Simulated Frequency w/o PSS', color=colors[1], linestyle='dashed')
    plt.plot(time2, freq2, label='Measured Frequency', color=colors[1])
    # plt.plot(timeOutage, freqOutage, label='Measured Frequency', color=colors[1])
    # Add labels, title, and legend
    plt.title('Grid Frequency')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency[Hz]')
    plt.xlim(0, max(res['t']))
    # Add horizontal dashed lines
    plt.axhline(49.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.axhline(50.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend()
    plt.savefig('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Figures/FrequencyPlots/grid_frequency_plot.pdf', format='pdf', bbox_inches='tight')




    # Plot load data
    plt.figure()
    #plt.plot(res['t'], abs(np.asarray(res['v'])), label="v", color=colors[1])
    plt.plot(res['t'], np.abs(res['p_bus_9']), label="Real Power at bus 9", color=colors[1])
    plt.plot(res['t'], np.abs(res['p_bus_7']), label="Real Power at bus 7", color=colors[2])
    plt.plot(res['t'], np.abs(res['q_bus_7']), label="Reactive Power at bus 7", color=colors[4])

    plt.xlabel('Real and Reactive power at bus 7')
    plt.ylabel('MW and MVAR')
    plt.legend()


    #Extract voltage at bus 7 and 9
    v_bus_7 = np.array([v[ps.bus_idx_red[6]] for v in res['v']])
    v_bus_9 = np.array([v[ps.bus_idx_red[8]] for v in res['v']])

    # Plot voltage
    plt.figure()
    plt.plot(res['t'], np.abs(v_bus_7), label='Bus 7 voltage', color=colors[1])
    plt.plot(res['t'], np.abs(v_bus_9), label='Bus 9 voltage', color=colors[3])
    #plt.plot(res['t'], np.abs(res['v']))
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage magnitude')
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

        #ax.plot(x, y, label='Normal Distribution of total frequency data', color=colors[0])
        plot_normalDistribution(avg_freq, ax=ax,  label='Simulated Frequency PDF')
        plot_normalDistribution(freq2, ax=ax, label='Real Data Frequency PDF')
        #plot_normalDistribution(totalPSD,ax=ax,  label='All Frequency data PDF')
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

        # Read the CSV file
        stability_df = pd.read_csv(r'C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS\Figures\FrequencyPlots\stabilityLineK2A.csv')

        # Extract columns as numpy arrays or lists
        w_stab = stability_df['w_stab'].values  # or .tolist() if you want a list
        mag_stab = stability_df['mag_stab'].values 

        # Compute FFT data
        p_fft_freq_rad, p_fft_mag = makeFFT(((
            (res['p_bus_7']) - np.mean(res['p_bus_7']))/60), time_step)
        
        # f_fft_freq_rad, f_fft_mag= makeFFT(((
        #     (avg_freq) - np.mean(avg_freq))/0.1), time_step)
        
        # ratio_mag= p_fft_mag/f_fft_mag

        # Fit a low-pass filter to the FFT data
        def func(x, a):
            return 1 / (a * x + 1)

        # popt_fft, pcovv_fft = scipy.optimize.curve_fit(func, p_fft_freq_rad, p_fft_mag)
        # print("Timedelay constant of fitted low-pass filter, FFT simulations:", popt_fft)
        # a=popt_fft[0]

        #import numpy as np
        #import scipy.optimize

        # Define the low-pass filter model
        def lp_filter(x, a):
            return 1 / (a * x + 1)

        # Frequency threshold (e.g., 5e-2 rad)
        freq_threshold = 5e-2

        # Apply constraint only to the data above this frequency
        mask = p_fft_freq_rad > freq_threshold
        x_constrained = p_fft_freq_rad[mask]
        y_constrained = p_fft_mag[mask]

        # Objective: Minimize 'a' (or keep flat if you just want the tightest fit)
        def objective(a):
            return -a[0]  # Could be 0 if you just want a valid a

        # Constraint: Above-threshold filter output must be ≥ FFT magnitude
        slack_eps = 5e-4  # Small slack to avoid numerical issues
        def constraint(a):
            return lp_filter(x_constrained, a[0]) - y_constrained+slack_eps

        # Initial guess and bounds
        a0 = [70.0]
        cons = {'type': 'ineq', 'fun': constraint}
        bounds = [(1e-6, None)]  # Avoid divide-by-zero

        # Run optimization
        result = scipy.optimize.minimize(objective, a0, constraints=cons, bounds=bounds)

        # Result
        if result.success:
            a_fit = result.x[0]
            print("Constrained time-delay constant (above threshold):", a_fit)
        else:
            print("Optimization failed:", result.message)

        omega = np.logspace(-5, 3, 500)
        d=TransferFunction([1], [a_fit, 1])
        w_req_dist, disturbanceReq, phaseReq_dist=bode(d, w=omega)
        disturbanceReq = (10 ** (disturbanceReq / 20))




        #print("Fitted low-pass filter values:", PlotFunc)
        # Create a new figure for the plot
        plt.figure()

        plt.plot(w_k2a_list[0], mag_k2a_list[0], label="Generator 1 Magnitude", color='darksalmon', alpha=0.8) 
        plt.plot(w_k2a_list[1], mag_k2a_list[1], label="Generator 2 Magnitude", color='olive', alpha=0.8)
        plt.plot(w_k2a_list[2], mag_k2a_list[2], label="Generator 3 Magnitude", color='skyblue', alpha=0.8)
        plt.plot(w_k2a_list[3], mag_k2a_list[3], label="Generator 4 Magnitude", color='plum', alpha=0.8)  
        plt.plot(w_stab, mag_stab, label="Stability Requirement", color='blue', alpha=0.7) 


        # Plot the FFT data
        plt.semilogx(p_fft_freq_rad, 1/p_fft_mag, label="FFT of P at bus 7", color=colors[3], alpha=0.6)

        #Plot TSO filter
        plt.semilogx(w_reqs, 1/actualReq, label="1/D(s)", color=colors[4], linestyle='dashed')
        plt.semilogx(w_stab, mag_stab, label="Stability Requirement", color='maroon', alpha=0.8, linestyle='dashed')

        # Plot the fitted low-pass filter
        plt.semilogx(w_req_dist, 1/disturbanceReq, label="Best low-pass filter describing FFT", color=colors[2], linestyle='dotted')

        # Add labels, title, and legend
        plt.yscale('log')
        plt.title('Magnitude plot of tuned generator and requirement')
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Magnitude (abs)')
        plt.legend()
        plt.xlim(0.01, 1)
        plt.ylim(0.5, 12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
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
        plt.plot(f1, Pxx1_dB, label='Simulated Data', alpha=0.9)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [dB/Hz]')
        plt.xlim(0, 0.5)
        plt.legend()
        plt.show()
    else:
        print("Skipping the code execution.")


