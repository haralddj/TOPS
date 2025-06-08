
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
from pathlib import Path
from scipy.optimize import minimize

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from tops.examples.user_models.user_lib.FreqPerformanceTool.Statistics import (
    plot_normalDistribution, CompareAutocorrelations, 
    EulerMaryama, getAutocorrelationData, MakeEulerMaryamaList
)
from tops.examples.user_models.user_lib.FreqPerformanceTool.Functions_utility import *
import wesanderson as ws


###Code tidied up using AI###

#color palette
colors = ws.film_palette('Darjeeling limited')

#-----------------------------
#file and directory constants
#-----------------------------
base_dir          = Path(r"C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS")
disturbance_dir   = base_dir / "storage_simulationdata" / "disturbance"
pelton_data_dir   = base_dir / "HistoricFreqData"    / "PeltonData"
outage_file       = base_dir / "HistoricFreqData"    / "Outage-anonymous_(1)" / "Outage-anonymous.xlsx"
figures_dir       = base_dir / "Figures" / "FrequencyPlots"

#-----------------------------
#custom load class definition
#-----------------------------
class ConstPowerLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.par         = data
        self.n_units     = len(data)
        self.bus_idx     = np.zeros(self.n_units,
                                    dtype=[(k, int) for k in self.bus_ref_spec().keys()])
        self.bus_idx_red = self.bus_idx.copy()
        self.sys_par     = sys_par

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def apparent_power_injections(self, x, v):
        s_inj = -(self.par['P'] + 1j*self.par['Q']) / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], s_inj

#-----------------------------
#main: load model & simulate
#-----------------------------
if __name__ == "__main__":
    import tops.examples.user_models.user_lib.k2aTunedToGrid.k2aTuning as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'ConstPowerLoad': [model['loads'][i] for i in (0,1,2)]}

    user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)
    ps.init_dyn_sim()

    print("Initial derivative max:", max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))
    _ = ps.solve_algebraic(0, ps.x_0, ps.v_0)

    disturbance_file = disturbance_dir / "disturbance.csv"
    df = pd.read_csv(disturbance_file)
    disturbance_values = df['disturbance'].tolist()
    freq_values       = df['avg_freq_dev'].tolist()

    t_end     = 25
    time_step = 0.02
    sol       = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives, ps.solve_algebraic,
        0, ps.x_0.copy(), ps.v0, t_end,
        max_step=time_step
    )

    res = defaultdict(list)
    t = 0
    start_time = time.time()
    line_mdl = ps.lines['Line']

    def p_bus_7(x, v):
        p = line_mdl.p_to(x, v)[1] + line_mdl.p_from(x, v)[2] + line_mdl.p_from(x, v)[3]
        return -ps.s_n * p

    def q_bus_7(x, v):
        q = line_mdl.q_to(x, v)[1] + line_mdl.q_from(x, v)[2] + line_mdl.q_from(x, v)[3]
        return -ps.s_n * q

    def p_bus_9(x, v):
        p = line_mdl.p_to(x, v)[4] + line_mdl.p_to(x, v)[5] + line_mdl.p_from(x, v)[6]
        return -ps.s_n * p

    theta          = 0.002
    mu_p, mu_q     = ps.loads['ConstPowerLoad'].par['P'][0], ps.loads['ConstPowerLoad'].par['Q'][0]
    sigma_p, sigma_q = 1.25, 0.1

    while t < t_end:
        sys.stdout.write(f"\r{t/t_end*100:.0f}%")
        if t > 0:
            curr_p = ps.loads['ConstPowerLoad'].par['P'][0]
            ps.loads['ConstPowerLoad'].par['P'][0] = EulerMaryama(
                theta, mu_p, sigma_p, time_step, curr_p
            )
        sol.step()
        x, v, t = sol.y, sol.v, sol.t

        res['t'].append(t)
        res['v'].append(v.copy())
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['p_bus_7'].append(p_bus_7(x, v))
        res['q_bus_7'].append(q_bus_7(x, v))
        res['p_bus_9'].append(p_bus_9(x, v))
        res['freq'].append(ps.gen['GEN'].speed(x, v)*50 + 50.04)
        res['gen_power'].append(ps.gen['GEN'].P_e(x, v).copy())
        res['gen_reactive_power'].append(ps.gen['GEN'].Q_e(x, v).copy())
        res['iterations'].append(ps.it_prev)

    print(f"\nSimulation completed in {time.time()-start_time:.2f}s")

    #-----------------------------
    #load real-data CSVs
    #-----------------------------
    csv_paths = [pelton_data_dir / f"data_set_{i}.csv" for i in range(1, 9)]
    dfs       = [pd.read_csv(p) for p in csv_paths]

    avg_freq = [np.mean(f) for f in res['freq']]
    n_pts    = len(res['t'])
    measured = [df['Frequency [Hz*4]'][2:2+n_pts]/4 for df in dfs]
    time2    = dfs[0]['Time [s]'][2:2+n_pts]

    time_outage, freq_outage = GetFreqData(str(outage_file), 'FI south:Frequency')
    time_outage, freq_outage = time_outage[:2000], freq_outage[:2000]

    #-----------------------------
    #check violation of MoNB
    #-----------------------------
    outside     = np.sum((np.array(avg_freq) < 49.9) | (np.array(avg_freq) > 50.1))
    pct_outside = outside/len(avg_freq)*100
    print(f"Pct outside 49.9–50.1 Hz: {pct_outside:.2f}%")

    #-----------------------------
    #plot frequency, load and voltage data
    #-----------------------------
    fig, ax = plt.subplots()
    ax.plot(res['t'], avg_freq, label='Simulated', color=colors[3])
    arr      = np.vstack(measured)
    ax.fill_between(time2, arr.min(0), arr.max(0), label='Measured range',
                    color='gray', alpha=0.2)
    ax.axhline(49.9, linestyle='--', alpha=0.5)
    ax.axhline(50.1, linestyle='--', alpha=0.5)
    ax.set(xlabel='Time [s]', ylabel='Freq [Hz]', title='Grid Frequency')
    ax.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(res['t'], avg_freq, label='Simulated', color=colors[3])
    plt.plot(time2, measured[0], label='Measured Stochastic', color=colors[4])
    #plt.plot(time_outage, freq_outage, label='Measured outage', color=colors[1])
    plt.axhline(49.9, linestyle='--', alpha=0.5)
    plt.axhline(50.1, linestyle='--', alpha=0.5)
    plt.xlabel('Time [s]'); plt.ylabel('Freq [Hz]')
    plt.title('Grid Frequency'); plt.legend()

    plt.figure()
    plt.plot(res['t'], np.abs(res['p_bus_9']), label='P bus 9', color=colors[1])
    plt.plot(res['t'], res['p_bus_7'],       label='P bus 7', color=colors[3])
    plt.xlabel('Time [s]'); plt.ylabel('MW'); plt.title('Load demand'); plt.legend()

    v_bus_7 = np.abs([v[ps.bus_idx_red[6]] for v in res['v']])
    v_bus_9 = np.abs([v[ps.bus_idx_red[8]] for v in res['v']])
    plt.figure()
    plt.plot(res['t'], v_bus_7, label='Bus 7 V', color=colors[1])
    plt.plot(res['t'], v_bus_9, label='Bus 9 V', color=colors[3])
    plt.xlabel('Time [s]'); plt.ylabel('Voltage'); plt.legend()
    plt.show()

    #-----------------------------
    #interactive further analysis
    #-----------------------------
    if input("Plot PDFs? (y/n): ").strip().lower() == 'y':
        #Plot PDFs of simulated and measured frequency data
        total_psd = plot_frequency_pdf(str(pelton_data_dir), "Frequency [Hz*4]")
        mean, std = fitNormalDistribution(total_psd)
        print("Mean, std:", mean, std)

        fig, ax = plt.subplots()
        plot_normalDistribution(avg_freq, ax=ax, label='Sim PDF', zorder=2, linewidth=2)
        for i, freq in enumerate(measured):
            plot_normalDistribution(
                freq, ax=ax,
                label='Real PDFs' if i==0 else None,
                color='gray', alpha=0.4
            )
        ax.set(xlabel='Frequency [Hz]', title='PDF of Simulated vs Real')
        ax.legend(); plt.show()

    if input("Plot ACFs? (y/n): ").strip().lower() == 'y':
        plotACFs(avg_freq)

    if input("Plot Requirement Check? (y/n): ").strip().lower() == 'y':

        #Plot Magnitude of system amplification transfer function alongside requirement and FFT of P bus 7
        w_list, mag_list, w_k2a, mag_k2a, w_req, mag_req, w_reqs, actual_req, fcr_gens = computeTransferFunctionMagnitudes(model_data)
        total_fcr = sum(fcr_gens)
        p_fft_w, p_fft_m = makeFFT(((np.array(res['p_bus_7']) - np.mean(res['p_bus_7']))/total_fcr), time_step)
        print("Mean P bus 7:", np.mean(res['p_bus_7']))

       # pick the region where we want to bound
        mask = p_fft_w > 5e-3

        # slack to avoid numerical issues
        slack_eps = 5e-4

        # vector‐valued constraint: for each ω in mask,
        #  1/(a*ω + 1) − Pₘ(ω) + slack ≥ 0
        def ub_constraint(a):
            return 1/(a * p_fft_w[mask] + 1) - p_fft_m[mask] + slack_eps

        # we want to *maximize* a → minimize −a
        res_opt = minimize(
            fun    = lambda a: -a[0],
            x0     = [70.0],
            bounds = [(1e-6, None)],
            constraints = {'type': 'ineq', 'fun': ub_constraint},
            method = 'SLSQP',
        )

        a_fit = res_opt.x[0] if res_opt.success else None
        print("Fitted a (tightest upper bound):", a_fit if a_fit else res_opt.message)

        omega = np.logspace(-5,3,500)
        tf = TransferFunction([1], [a_fit, 1])
        w_req_dist, dist_req, _ = bode(tf, w=omega)
        dist_req = 10**(dist_req/20)

        plt.figure()
        for idx in range(len(mag_k2a)):
            plt.semilogx(w_k2a[idx], mag_k2a[idx], label=f"Gen {idx+1}", alpha=1)
        plt.semilogx(p_fft_w, 1/p_fft_m, label="FFT P bus 7", color=colors[3], alpha=0.6)
        plt.semilogx(w_reqs, 1/actual_req, label="1/D(s)", linestyle='--')
        plt.semilogx(w_req_dist, 1/dist_req, label=f"LPF, T={a_fit:.1f}", linestyle=':')
        plt.yscale('log'); plt.xlabel('rad/s'); plt.ylabel('Mag'); plt.title('Generator tuning')
        plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.show()

    if input("Plot PSD? (y/n): ").strip().lower() == 'y':

        #Plot power spectral density (PSD) of simulated and measured data
        all_meas   = np.hstack(measured)
        f_meas, P_meas = periodogram(all_meas, 50)
        P_meas_db    = 20*np.log10(np.abs(P_meas))
        f_sim, P_sim = periodogram(avg_freq, 50)
        P_sim_db     = 20*np.log10(np.abs(P_sim))

        plt.figure()
        plt.plot(f_meas, P_meas_db, label='Measured', alpha=0.7)
        plt.plot(f_sim, P_sim_db, label='Simulated', alpha=0.9)
        plt.xlim(0,0.5); plt.xlabel('Hz'); plt.ylabel('PSD [dB/Hz]'); plt.legend(); plt.show()






