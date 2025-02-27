import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time

import numpy as np

import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import importlib

if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    #import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    dt=5e-3


    model['loads'] = {'LoadAsCurrent': [
        ['name', 'bus',     'P_setp',  'Q_setp',     'T_i',    'T_v',  'v_n', 'model'],
        ['L1',    'B7',        967,       100,       0.1,         0.1,      230,     'Z'],
        ['L2',    'B9',       1767,       100,       0.1,         0.1,      230,     'Z'],
    ]}


    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.ode_fun(0, ps.x_0))))

    x0 = ps.x_0
    v0 = ps.v_0

    time_step = 5e-3
    t_end = 50
    x_0 = ps.x_0.copy()
    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=time_step)
    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # if t > 0:
        #     ps.loads['LoadAsCurrent'].set_input('P_setp', 967, 0)
        # if t > 0:
        #     ps.loads['LoadAsCurrent'].set_input('Q_setp', 100, 0)

        # Simulate next step
        result = sol.step()
        x = sol.y
        t = sol.t
        v = sol.v


        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(sol.t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['gen_production'].append(ps.gen['GEN'].p_e(x, v).copy())
        res['p_loads'].append(ps.loads['LoadAsCurrent'].P(x, v))
        res['q_loads'].append(ps.loads['LoadAsCurrent'].Q(x, v))
        res['freq'].append((ps.gen['GEN'].speed(x, v).copy()*50/(2*np.pi))+50.0)
        res['current'].append(abs((ps.loads['LoadAsCurrent'].i_inj(x, v))))
        res['voltage'].append(abs(v[6]))
        res['voltages_real'].append(ps.loads['LoadAsCurrent'].voltageReal(x,v))
        res['voltages_imag'].append(ps.loads['LoadAsCurrent'].voltageImag(x,v))
        vstate = ps.loads['LoadAsCurrent'].voltageReal(x,v)+1j*ps.loads['LoadAsCurrent'].voltageImag(x,v)
        res['vstate'].append(abs(vstate))





    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))
    avg_freq = [np.mean(freq) for freq in res['freq']]
    p_load1=[]
    p_load2=[]
    for elem in res['p_loads']:
        p_load1.append(elem[0])
        p_load2.append(elem[1])

    # plt.figure()
    # plt.plot(res['t'], res['gen_speed'])
    # plt.show()

    fig, ax = plt.subplots(4, sharex=True)

    ax[0].plot(res['t'], res['voltage'], label='Voltage at bus 7 (absolute value p.u)')
    ax[0].plot(res['t'], res['vstate'], label='vstates (absolute value p.u)')

    ax[0].legend()
    ax[1].plot(res['t'], avg_freq, label='Avg. freq.')
    ax[1].legend()
    ax[2].plot(res['t'], res['current'], label='Injected currents (p.u.)')
    ax[2].legend()
    ax[3].plot(res['t'], p_load1, label='P Load 1')
    ax[3].plot(res['t'], p_load2, label='P Load 2')
    ax[3].legend()
    plt.show()