import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
from tops.solvers_sde import EulerDAE_SDE
import numpy as np

def run_simulation_spranglast(model):
    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    load_state_idx_g = ps.loads['DynamicLoadFiltered'].lpf_g.state_idx_global['x']
    load_state_idx_b = ps.loads['DynamicLoadFiltered'].lpf_b.state_idx_global['x']

    t_end = 10
    # Solver
    sol = EulerDAE_SDE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, max_step=5e-3, dim_w=2)

    def b_func(t, x, v):
        mat = np.zeros((len(sol.x), sol.dim_w))
        mat[load_state_idx_g[0], 0] = 0.1
        mat[load_state_idx_b[0], 0] = 0.1
        mat[load_state_idx_g[1], 1] = 0.1
        mat[load_state_idx_b[1], 1] = 0.1
        return mat

    sol.b = b_func

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t / t_end * 100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        dw = sol.dw  # random variable
        t = sol.t

        if t > 5:
            ps.loads['DynamicLoadFiltered'].set_input('g_setp', 2.08076099 * 0.6, 1)

        res['time'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['v'].append(v.copy())
        res['P_load'].append(ps.loads['DynamicLoadFiltered'].p(x, v))
    return res

if __name__ == '__main__':
    # Load model
    import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    model = model_data.load()
    model['loads'] = {'DynamicLoadFiltered':  [
        model['loads'][0] + ['T_g', 'T_b'],
        *[row + [0.1, 0.1] for row in model['loads'][1:]]
    ]}

    # Run simulation with original values
    res_original = run_simulation_spranglast(model)
    orgH=model['generators']['GEN'][1][6]

    # Modify inertia constant of G1 and run simulation again
    model['generators']['GEN'][1][6] = 100  # Change inertia constant H of G1 to 7
    highH=model['generators']['GEN'][1][6]
    res_modifiedhigh = run_simulation_spranglast(model)
    model['generators']['GEN'][1][6]=3
    lowH=model['generators']['GEN'][1][6]
    res_modifiedlow = run_simulation_spranglast(model)


    #Show difference in generator speed
    deviationhigh = []
    deviationlow = []
    for i in range(len(res_original['gen_speed'])):
        deviationhigh.append(res_original['gen_speed'][i][0] - res_modifiedhigh['gen_speed'][i][0])
        deviationlow.append(res_original['gen_speed'][i][0] - res_modifiedlow['gen_speed'][i][0])
    # Plot generator speed for both simulations
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(res_original['time'], [speed[0] for speed in res_original['gen_speed']], label=f'Original H={orgH}')
    ax[0].plot(res_modifiedhigh['time'], [speed[0] for speed in res_modifiedhigh['gen_speed']], label=f'Modified H={highH}')
    ax[0].plot(res_modifiedlow['time'], [speed[0] for speed in res_modifiedlow['gen_speed']], label=f'Modified H={lowH}')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Generator 1 Speed')
    ax[0].legend()
    ax[1].plot(res_original['time'], deviationhigh, label='Deviation with high H')
    ax[1].plot(res_original['time'], deviationlow, label='Deviation with low H')
    ax[1].legend()
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Deviation')
    plt.show()

    # Plot load power for both simulations
    fig, ax = plt.subplots(1)
    ax.plot(res_original['time'], res_original['P_load'], label='Original')
    ax.plot(res_modifiedhigh['time'], res_modifiedhigh['P_load'], label='Modified H=10')
    ax.plot(res_modifiedlow['time'], res_modifiedlow['P_load'], label='Modified H=3')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Load Power [p.u.]')
    ax.legend()
    plt.show()


