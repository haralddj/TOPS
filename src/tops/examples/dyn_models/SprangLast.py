import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
from tops.solvers_sde import EulerDAE_SDE
import numpy as np

if __name__ == '__main__':

    # Load model
    #import examples.user_models.user_lib.min_sm_ib as model_data
    import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    #import tops.ps_models.k2a as model_data

    model = model_data.load()
    model['loads'] = {'DynamicLoadFiltered':  [# model['loads']}
        model['loads'][0] + ['T_g', 'T_b'],    #Change loads to dynamicfiltered, adds two elements per load representing
        *[row + [0.1, 0.1] for row in model['loads'][1:] #time constants for the real and imaginary part
    ]]}                                                      # of the low pass filters



    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    load_state_idx_g = ps.loads['DynamicLoadFiltered'].lpf_g.state_idx_global['x']
    load_state_idx_b = ps.loads['DynamicLoadFiltered'].lpf_b.state_idx_global['x']

    t_end = 20
    # Solver
    sol = EulerDAE_SDE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, max_step=5e-3, dim_w = 2)

    def b_func(t, x, v):
        mat = np.zeros((len(sol.x), sol.dim_w))
        mat[load_state_idx_g[0], 0] = 0.1
        mat[load_state_idx_b[0], 0] = 0.1
        mat[load_state_idx_g[1], 1] = 0.1
        mat[load_state_idx_b[1], 1] = 0.1
        return mat

    # plt.imshow(b_func(*[None]*3))
    # plt.show()

    sol.b = b_func

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        dw = sol.dw  # random variable
        t = sol.t

        if t > 5:
            ps.loads['DynamicLoadFiltered'].set_input('g_setp', 5, 1) #

        res['time'].append(t)
        res['p_load'].append(ps.loads['DynamicLoadFiltered'].p(x, v))
        res['g_load'].append(ps.loads['DynamicLoadFiltered'].g_load(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['v'].append(v.copy())
        res['dw'].append(sol.dw)
        res['freq'].append((ps.gen['GEN'].speed(x, v).copy()/(2*np.pi))+50)

    gen_1_freq = [freq[0] for freq in res['freq']]
    gen_1_speed = [speed[0] for speed in res['gen_speed']]
    fig2, ax1 = plt.subplots(2, sharex=True)
    ax1[0].plot(res['time'], gen_1_speed)
    ax1[0].set_xlabel('Time [s]')
    ax1[0].set_ylabel('Generator 1 Speed')
    ax1[1].plot(res['time'], gen_1_freq)
    ax1[1].set_xlabel('Time [s]')
    ax1[1].set_ylabel('Frequency [Hz]')
    ax1[1].set_ylim([49.999, 50.002])
    plt.show()




    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(res['time'], res['gen_speed'])
    ax[0].set_ylabel('Gen speed')
    ax[0].set_xlabel('Time [s]')
    ax[1].plot(res['time'], sol.dt*np.cumsum(np.array(res['dw']), axis=0))
    ax[1].set_ylabel('W')
    ax[2].plot(res['time'], res['p_load'])
    ax[2].set_ylabel('P Load [p.u.]')
    plt.show()

    gen_angle_np = np.array(res['gen_angle']).T
    gen_speed_np = np.array(res['gen_speed']).T

    angle_speed_mat = np.vstack([gen_angle_np, gen_speed_np])

    cov_mat = np.cov(angle_speed_mat)
    cov_mat_angle = np.cov(gen_angle_np[[0, 2], :])
    np.linalg.det(cov_mat_angle)

    # np.linalg.inv()

    np.savetxt('p_load.csv', np.array(res['p_load']))
    np.savetxt('v_abs.csv', abs(np.array(res['v'])))
    np.savetxt('gen_speed.csv', np.array(res['gen_speed']))

    np.array(res['v'])
    import pandas as pd



    pd.DataFrame()

    gen_speed = np.loadtxt('v_abs.csv')
    plt.plot(gen_speed)
    plt.show()