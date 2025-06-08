
import sys
import time
import importlib
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from pathlib import Path
import scipy.signal as sig

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.dyn_models.utils import DAEModel

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from tops.examples.user_models.user_lib.FreqPerformanceTool.Statistics import (
    plot_normalDistribution, CompareAutocorrelations, 
    EulerMaryama, getAutocorrelationData, MakeEulerMaryamaList
)
from tops.examples.user_models.user_lib.FreqPerformanceTool.Functions_utility import plot_frequency_pdf, fitNormalDistribution,computeTransferFunctionMagnitudes, makeFFT, makeWelchFFT, plotACFs
import wesanderson as ws



###Starting concept for doing stochastic analysis on n45###
###Code is tidied up using AI###

colors=ws.film_palette('Darjeeling limited')

#color palette
colors = ws.film_palette('Darjeeling limited')

#directories
base_dir     = Path(r"C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS")
pelton_dir   = base_dir/"HistoricFreqData"/"PeltonData"
disturb_file = base_dir/"storage_simulationData"/"disturbance"/"disturbance.csv"

#custom load class
class ConstPowerLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.par         = data
        self.n_units     = len(data)
        self.bus_idx     = np.zeros(self.n_units,
                                    dtype=[(k,int) for k in self.bus_ref_spec()])
        self.bus_idx_red = self.bus_idx.copy()
        self.sys_par     = sys_par

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def apparent_power_injections(self, x, v):
        s = -(self.par['P'] + 1j*self.par['Q']) / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], s

if __name__ == "__main__":
    #load and init model
    import tops.ps_models.n45_tuned as model_data
    importlib.reload(model_data)
    model = model_data.load()
    idxs  = list(range(len(model['loads'])))
    model['loads'] = {'ConstPowerLoad': [model['loads'][i] for i in idxs]}

    user_lib = type('',(),{'loads': type('',(),{'ConstPowerLoad': ConstPowerLoad})})
    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_lib)
    ps.init_dyn_sim()
    print("Initial derivative max:",
          max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    #simulation setup
    t_end, dt = 50.0, 0.02
    solver = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives, ps.solve_algebraic,
        0, ps.x_0.copy(), ps.v0, t_end,
        max_step=dt
    )

    #helpers for bus 5321
    line = ps.lines['Line']
    def p_bus(x,v): return -ps.s_n * line.p_to(x,v)[44]
    def q_bus(x,v): return -ps.s_n * line.q_to(x,v)[44]

    #load disturbance data
    df_dist    = pd.read_csv(disturb_file)
    disturbance = df_dist['disturbance'].tolist()
    freq_vals   = df_dist['avg_freq_dev'].tolist()

    #run simulation
    res, t = defaultdict(list), 0.0
    start   = time.time()
    p0      = ps.loads['ConstPowerLoad'].par['P'][28]

    while t < t_end:
        sys.stdout.write(f"\r{t/t_end*100:.0f}%")
        if t > 0:
            ps.loads['ConstPowerLoad'].par['P'][28] = p0 + 300
        solver.step()
        x, v, t = solver.y, solver.v, solver.t

        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x,v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x,v).copy())
        res['v'].append(v.copy())
        res['p_bus_5321'].append(p_bus(x,v))
        res['q_bus_5321'].append(q_bus(x,v))
        res['freq'].append(ps.gen['GEN'].speed(x,v)*50 + 50)
        res['gen_power'].append(ps.gen['GEN'].P_e(x,v).copy())
        res['iterations'].append(ps.it_prev)

    print(f"\nSimulation completed in {time.time()-start:.2f}s")

    #load measured Pelton data
    df = pd.read_csv(pelton_dir/"data_set_1.csv")
    freq2    = df['Frequency [Hz*4]'][2:2+len(res['t'])] / 4
    time2    = df['Time [s]'][2:2+len(res['t'])]
    avg_freq = [np.mean(f) for f in res['freq']]

    #compute center-of-inertia frequency
    H     = ps.gen['GEN'].par['H']
    f_coi = np.dot(H, np.vstack(res['freq']).T) / H.sum()

    #print deviation stat
    pct = np.mean((np.array(avg_freq)<49.9)|(np.array(avg_freq)>50.1)) * 100
    print(f"Pct outside 49.9–50.1 Hz: {pct:.2f}%")

    #plot frequencies
    plt.figure()
    plt.plot(res['t'], avg_freq, label='Simulated', color=colors[0])
    plt.plot(time2, freq2,       label='Measured',  color=colors[4])
    plt.plot(res['t'], f_coi,     label='COI', linestyle='--', color='k')
    plt.xlabel('Time [s]'); plt.ylabel('Freq [Hz]'); plt.title('Generator Frequencies')
    plt.legend()

    #plot bus 5321 power
    plt.figure()
    plt.plot(res['t'], np.abs(res['p_bus_5321']), label='P 5321', color=colors[3])
    plt.plot(res['t'], np.abs(res['q_bus_5321']), label='Q 5321', color=colors[1])
    plt.xlabel('Time [s]'); plt.ylabel('MW/MVar'); plt.legend()

    #plot generator power
    plt.figure()
    plt.plot(res['t'], np.abs(res['gen_power']), label='Gen Power')
    plt.xlabel('Time [s]'); plt.ylabel('MW'); plt.legend()

    plt.show()

    #interactive analyses
    if input("Plot PDFs? (y/n): ").strip().lower()=='y':
        psd = plot_frequency_pdf(str(pelton_dir), "Frequency [Hz*4]")
        m,s = fitNormalDistribution(psd)
        print("Mean, std:", m, s)
        fig,ax = plt.subplots()
        plot_normalDistribution(avg_freq, ax=ax, label='Sim PDF')
        plot_normalDistribution(freq2,     ax=ax, label='Real PDF')
        ax.set(xlabel='Freq [Hz]', title='PDF Comparison')
        ax.legend(); plt.show()

    if input("Plot ACFs? (y/n): ").strip().lower()=='y':
        plotACFs(avg_freq)

    if input("Plot PSD? (y/n): ").strip().lower()=='y':
        f_sim, P_sim = sig.periodogram(avg_freq, 50)
        f_mea, P_mea = sig.periodogram(freq2,    50)
        plt.figure()
        plt.plot(f_mea, 20*np.log10(P_mea), label='Measured')
        plt.plot(f_sim, 20*np.log10(P_sim), label='Simulated')
        plt.xlim(0,2); plt.xlabel('Hz'); plt.ylabel('PSD [dB/Hz]'); plt.legend(); plt.show()


