
import sys
from multiprocessing import Pool
from collections import defaultdict
import time
import itertools
import tops.dynamic as dps
import numpy as np
import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.k2aTuning as model_data
import pandas as pd
import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.dyn_models.utils import DAEModel
import importlib
import os

output_folder = r"C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS\Figures\SimulationStorageFreqAndLoad"
os.makedirs(output_folder, exist_ok=True)


def EulerMaryama(theta, mu, sigma, dt, last_value):
    dW = np.random.normal(0, np.sqrt(dt))
    return last_value+theta*(mu-last_value)*dt+sigma*dW




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


def simulate_sys(sim_i, H, R, r, Tf, Tr, Tg, Tw):
    model = model_data.load()
    #Set overnor parameters:
    for gen in model['generators']['GEN'][1:]:
        gen[6] = H
        #gen[7] = D
    #Set the time constants of the governors
    for gov in model['gov']['HYGOV'][1:]:
        gov[2]=R
        gov[3]=r
        gov[4]=Tf
        gov[5]=Tr
        gov[6]=Tg
        gov[8]=Tw


    # ----------------------------------
    # Load model and initialize system
    # ----------------------------------

    model['loads'] = {
        'ConstPowerLoad': [model['loads'][ix] for ix in [0, 1, 2]]
    }


    user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})

    ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)
    ps.init_dyn_sim()


    # ----------------------------------
    # Setup simulation parameters
    # ----------------------------------
    t_end = 2000
    x_0 = ps.x_0.copy()
    time_step = 0.02
    t_save=0.02
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v0,t_end, max_step=time_step)

    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
    line_mdl = ps.lines['Line']

    def p_bus_7(x, v):
        return -ps.s_n * (line_mdl.p_to(x, v)[1] + line_mdl.p_from(x, v)[2] + line_mdl.p_from(x, v)[3])
    

    # ----------------------------------
    # Noise and stochastic config
    # ----------------------------------


    p_0 = ps.loads['ConstPowerLoad'].par['P'][0]
    theta = 0.002
    mu_p = p_0
    sigma_p = 1.25

    # ----------------------------------
    # Simulation loop
    # ----------------------------------
    eventflag=True
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / t_end * 100))


        if t > 0:
            ps.loads['ConstPowerLoad'].par['P'][0] = EulerMaryama(theta, mu_p, sigma_p, time_step, ps.loads['ConstPowerLoad'].par['P'][0])                



        result = sol.step()
        x, v, t = sol.y, sol.v, sol.t

        res['t'].append(t)
        res['p_bus_7'].append(p_bus_7(x, v).copy())
        freqs=(((ps.gen['GEN'].speed(x, v).copy() * 50)) + 50)
        [res['f%d' % i].append(freq) for i, freq in enumerate(freqs)]
        k=int(t_save/time_step)
        filename = f"res_sim_{sim_i}_H_{H}_R_{R}_r_{r}_Tf_{Tf}_Tr_{Tr}_Tg_{Tg}_Tw_{Tw}.parquet"
        pd.DataFrame(data=res).iloc[::, :].to_parquet(os.path.join(output_folder, filename))

n_sim = 1
sim_i = np.arange(n_sim)


case1=[6, 0.12, 0.65, 0.05, 5.7, 0.5, 1.1]
case2=[6, 0.12, 0.65, 0.05, 10.7, 0.5, 1.1]
case3=[6, 0.12, 0.65, 0.05, 5.7, 0.5, 3]
case4=[6, 0.12, 1, 0.05, 5.7, 0.5, 1.1]
case5=[4.5, 0.12, 0.65, 0.05, 5.7, 0.5, 1.1]

cases = [case1, case2, case3, case4, case5]

# build a list of (sim_i, *case) for each sim_i in 0..n_sim-1 and each case
items = [
    (sim_i, *case)
    for sim_i in range(n_sim)
    for case in cases
]


items = list(itertools.product(sim_i ))

# Number of processes
n_processes = 2

if __name__ == '__main__':
    start = time.time()

    cases = [case1, case2, case3, case4, case5]
    items = [(i, *case) for i in range(n_sim) for case in cases]

    with Pool(n_processes) as pool:
        pool.starmap(simulate_sys, items)

    end = time.time()
    total_runs = len(items)
    print(f"Ran {total_runs} cases in {end - start:.1f} seconds")   



    




