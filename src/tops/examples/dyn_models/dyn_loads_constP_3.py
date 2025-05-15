import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np
from examples.user_models.user_lib.MyTools.Statistics import plot_normalDistribution, plotFFT, CompareAutocorrelations, EulerMaryama



if __name__ == '__main__':

    # Load model
    import tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'ConstPowerLoad': model['loads']}
        # 'Load': [model['loads'][ix] for ix in [0, 2]],
        # 'ConstPowerLoad': [model['loads'][ix] for ix in [0, 1]]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.loads['ConstPowerLoad'].par['Q'] = 0
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    v = ps.solve_algebraic(0, ps.x0, ps.v_0)
    # v = ps.solve_algebraic(0, ps.x0)
    max(abs(v - ps.v_0))

    print(max(abs(ps.ode_fun(0, ps.x_0, ps.v0))))
    time_step = 5e-3
    t_end = 40
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v0, t_end, max_step=time_step)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

    p_0 = ps.loads['ConstPowerLoad'].par['P'][0]
    print(p_0)
    print(ps.loads['ConstPowerLoad'].par['P'])
    theta = 0.1
    mu = p_0
    sigma = 10

    length = 0


    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        # print(t)

        if 0 <= t:  #  < 1.1:
            ps.loads['ConstPowerLoad'].par['P'][0]=EulerMaryama(theta, mu, sigma, time_step, ps.loads['ConstPowerLoad'].par['P'][0])

            #Q=EulerMaryama(theta, mu, sigma, time_step, Q)

            #ps.loads['ConstPowerLoad'].par['P'] += 0 # np.random.randn(ps.loads['ConstPowerLoad'].n_units)
        # else:
            # ps.loads['ConstPowerLoad'].par['P'][0] = p_0
            
        # if 1.1 <= t:
            # ps.loads['ConstPowerLoad'].par['P'][0] = 967
        # print(ps.loads['ConstPowerLoad'].par['P'][0])
        # Short circuit
        # if t >= 1 and t <= 1.05:
            # ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e-1
        # else:
            # ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        # dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['v'].append(v.copy())
        res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50) / (2 * np.pi)) + 50.0)
        # res['q_bus_7'].append(q_bus_7(x, v).copy())
        res['load_P'].append(ps.loads['ConstPowerLoad'].P(x, v).copy())
        # res['load_Q'].append(ps.loads['ConstPowerLoad'].Q(x, v).copy())
        res['iterations'].append(ps.it_prev)

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))


    figfig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

    ax1.plot(res['t'], res['gen_speed'])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Gen speed')

    ax2.plot(res['t'], res['gen_angle'])
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Gen angle')

    ax3.plot(res['t'], res['freq'])
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Frequency')

    ax4.plot(res['t'], res['load_P'])
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('P\_load')

    
    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['v']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('Bus voltage magnitude')
    #
    # fig = plt.figure()
    # plt.plot(res['t'], np.angle(np.array(res['v'])), color='g')
    # plt.plot(res['t'], res['gen_angle'], color='r')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Bus voltage angle')
    #
    # fig = plt.figure()
    # plt.plot(res['t'], res['iterations'])
    
    fig = plt.figure()
    
    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['p_bus_7']))
    # plt.xlabel('Power at bus 7')
    # plt.ylabel('MW')

    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['q_bus_7']))
    # plt.xlabel('Reactive power at bus 7')
    # plt.ylabel('MW')

    # fig = plt.figure()
    # plt.plot(res['t'], np.abs(res['load_Q']))
    # plt.xlabel('Time [s]')
    # plt.ylabel('MVA')
    
    plt.show()
    